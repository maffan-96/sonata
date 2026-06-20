# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Feature-gradient guided hole completion for a QEM mesh, as a POST-PROCESS.

Pipeline:
    1. Load a mesh (or point cloud) .ply -- e.g. the vertices/mesh exported by
       the incremental QEM mesher.
    2. Attach a per-vertex Sonata feature (computed on the vertices directly,
       transferred from a dense cloud via nearest neighbour, or loaded from a
       .npy). Features are PCA-reduced and L2-normalised.
    3. Build a feature-gradient / conductance field over the mesh edges and
       auto-calibrate sigma_f from the interior (closed) edges.
    4. Extract open boundary loops, gate each loop by size / planarity /
       feature coherence, and complete the accepted ones with a feature-
       weighted Laplacian (anisotropic-diffusion) membrane solve.
    5. Save the completed mesh (and an optional feature-gradient debug ply).

This is unsupervised: no semantic labels are used, only the continuous
encoder features. The feature gradient decides WHICH gaps (and which spans of
a gap) are genuine same-surface holes vs real object boundaries, and sets the
conductance that SHAPES the filled patch so it stops at boundaries.

Examples:
    export PYTHONPATH=./
    # features from the dense cloud (recommended), transferred to vertices
    python demo/6_feature_hole_fill.py \
        --input  mesh_vertices.ply \
        --dense  dense_cloud.ply \
        --output mesh_completed.ply --save-debug

    # features straight from the vertices (no dense cloud handy)
    python demo/6_feature_hole_fill.py -i mesh_vertices.ply -o out.ply

    # iterate on the meshing logic without a GPU / full sonata env
    python demo/6_feature_hole_fill.py -i mesh_vertices.ply -o out.ply --no-sonata
"""

import argparse
import os

import numpy as np

try:
    import open3d as o3d
except ImportError:  # IO needs open3d; the algorithm core does not
    o3d = None

try:
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import Delaunay, cKDTree
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This script needs scipy (sparse solve + Delaunay). "
        "Install with `pip install scipy`.\n" + str(e)
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def sonata_features(coord, color, normal, feat_dim, grid_size=None, device=None):
    """Run the Sonata encoder on a point set and return per-point features
    reduced to `feat_dim` dims and L2-normalised. Imported lazily so the rest
    of the script (mesh logic) runs without the heavy sonata/spconv env."""
    import torch
    import sonata

    try:
        import flash_attn
    except ImportError:
        flash_attn = None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sonata.utils.set_seed(53124)

    if flash_attn is not None and device == "cuda":
        model = sonata.load("sonata", repo_id="facebook/sonata").to(device)
    else:
        cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=cfg
        ).to(device)
    model.eval()

    transform = sonata.transform.default()
    if grid_size is not None:
        for t in transform.transforms:
            if type(t).__name__ == "GridSample":
                t.grid_size = grid_size
    point = {
        "coord": coord.astype(np.float32),
        "color": color.astype(np.float32),   # 0-255
        "normal": normal.astype(np.float32),
    }
    point = transform(point)
    with torch.inference_mode():
        for k in point.keys():
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].to(device, non_blocking=True)
        point = model(point)
        # upcast encoder features back to the first (grid) level
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        feat = point.feat                       # grid-level features
        # PCA-reduce for a compact, discriminative descriptor
        q = min(feat_dim, feat.shape[1])
        u, s_val, v = torch.pca_lowrank(feat, center=True, q=q, niter=5)
        feat = feat @ v[:, :q]
        feat = feat[point.inverse]              # back to every input point
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype(np.float32)


def attach_features(verts, vcolor, vnormal, args):
    """Return (N, D) L2-normalised per-vertex features for the mesh vertices."""
    if args.features:
        print(f"  loading precomputed features: {args.features}")
        feat = np.load(args.features).astype(np.float32)
        assert feat.shape[0] == verts.shape[0], (
            f"feature rows ({feat.shape[0]}) != vertices ({verts.shape[0]})"
        )
    elif args.no_sonata:
        print("  [--no-sonata] using vertex normals as a stand-in feature "
              "(meshing-logic debug only)")
        feat = vnormal.copy()
    elif args.dense:
        print(f"  computing Sonata features on dense cloud: {args.dense}")
        dpcd = o3d.io.read_point_cloud(args.dense)
        dcoord = np.asarray(dpcd.points, np.float32)
        dcolor = (np.asarray(dpcd.colors) * 255.0 if dpcd.has_colors()
                  else np.full_like(dcoord, 128.0))
        if dpcd.has_normals():
            dnormal = np.asarray(dpcd.normals)
        else:
            dpcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            dnormal = np.asarray(dpcd.normals)
        dfeat = sonata_features(dcoord, dcolor, dnormal, args.feat_dim,
                                grid_size=args.grid_size)
        print("  transferring dense features to mesh vertices (nearest neighbour)")
        _, idx = cKDTree(dcoord).query(verts, k=1)
        feat = dfeat[idx]
    else:
        print("  [warn] computing Sonata features on DECIMATED vertices; "
              "feature quality is better with --dense <dense_cloud.ply>")
        feat = sonata_features(verts, vcolor, vnormal, args.feat_dim,
                               grid_size=args.grid_size)

    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.clip(norm, 1e-9, None)


# --------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------- #
def edge_face_counts(faces):
    """Map sorted (a,b) -> incident face count."""
    counts = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def boundary_loops(faces):
    """Chain open (single-incidence) edges into ordered vertex loops."""
    counts = edge_face_counts(faces)
    adj = {}
    for (a, b), c in counts.items():
        if c == 1:                       # open edge
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    loops, visited = [], set()
    for start in list(adj.keys()):
        if start in visited or not adj.get(start):
            continue
        loop, prev, cur = [start], None, start
        visited.add(start)
        while True:
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            if nxt in visited:           # non-manifold tangle: stop this loop
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def calibrate_sigma_f(faces, feat):
    """sigma_f = median feature difference over interior (closed) edges -- this
    makes 'what counts as a boundary' scene-relative."""
    counts = edge_face_counts(faces)
    diffs = [np.linalg.norm(feat[a] - feat[b])
             for (a, b), c in counts.items() if c >= 2]
    if not diffs:
        diffs = [np.linalg.norm(feat[a] - feat[b]) for (a, b) in counts.keys()]
    # floor avoids a degenerate gate on near-uniform-feature scenes, where a
    # zero median would otherwise reject (or over-tighten) every hole.
    return max(float(np.median(diffs)) if diffs else 1.0, 1e-3)


def point_in_polygon(pts, poly):
    """Vectorised even-odd test. pts:(M,2) poly:(K,2) ordered. -> (M,) bool."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
        inside ^= cond
        j = i
    return inside


# --------------------------------------------------------------------------- #
# Feature-gated membrane hole fill
# --------------------------------------------------------------------------- #
def loop_is_fillable(loop, verts, feat, sigma_f, med_edge, args):
    """Geometry + feature-coherence gate. Returns (ok, plane_basis)."""
    P = verts[loop]
    # size cap
    if len(loop) > args.max_hole_edges:
        return False, None
    diam = np.linalg.norm(P.max(0) - P.min(0))
    if diam > args.max_hole_diam_vox * med_edge:
        return False, None
    # best-fit plane via PCA; reject if too non-planar
    c = P.mean(0)
    cov = np.cov((P - c).T)
    evals, evecs = np.linalg.eigh(cov)            # ascending
    n = evecs[:, 0]
    planarity = evals[0] / max(evals.sum(), 1e-12)
    if planarity > args.max_planarity:
        return False, None
    # feature coherence: max feature gradient across the loop must stay below
    # a multiple of sigma_f, else the loop straddles a real boundary.
    F = feat[loop]
    # sample chords (opposing vertices) + consecutive steps
    k = len(loop)
    chord = max([np.linalg.norm(F[i] - F[(i + k // 2) % k]) for i in range(k)])
    if chord > args.coherence_sigma_scale * sigma_f:
        return False, None
    u = evecs[:, 2]
    u = u - n * (u @ n)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return True, (c, u, v, n)


def fill_loop(loop, verts, feat, basis, sigma_f, med_edge):
    """Feature-weighted Laplacian membrane fill of one boundary loop.

    Returns (new_xyz (M,3), new_feat (M,D), tris (T,3) indexing into the
    concatenation [loop_verts ; new_xyz]). Empty arrays if nothing added.
    """
    c, u, v, n = basis
    D = feat.shape[1]

    def _empty():
        return (np.zeros((0, 3)), np.zeros((0, D)),
                np.zeros((0, 3), int), np.zeros(0, int), 0, n)

    P = verts[loop]
    rel = P - c
    uvB = np.stack([rel @ u, rel @ v], axis=1)         # boundary 2D
    hB = rel @ n                                        # boundary heights
    FB = feat[loop]

    # interior grid samples inside the loop polygon
    lo, hi = uvB.min(0), uvB.max(0)
    step = max(med_edge, 1e-6)
    gx = np.arange(lo[0] + step, hi[0], step)
    gy = np.arange(lo[1] + step, hi[1], step)
    if len(gx) == 0 or len(gy) == 0:
        grid = np.zeros((0, 2))
    else:
        gxx, gyy = np.meshgrid(gx, gy)
        grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1)
        grid = grid[point_in_polygon(grid, uvB)]

    uv = np.vstack([uvB, grid]) if len(grid) else uvB.copy()
    nB = len(uvB)
    nI = len(grid)

    # triangulate in 2D, keep triangles whose centroid is inside the polygon
    if len(uv) < 3:
        return _empty()
    try:
        tri = Delaunay(uv)
    except Exception:
        return _empty()
    cent = uv[tri.simplices].mean(axis=1)
    keep = tri.simplices[point_in_polygon(cent, uvB)]
    if len(keep) == 0:
        return _empty()

    # interior features via inverse-distance interpolation from the boundary
    if nI > 0:
        d, idx = cKDTree(uvB).query(grid, k=min(6, nB))
        w = 1.0 / (d + 1e-6)
        w /= w.sum(1, keepdims=True)
        FI = np.einsum("ij,ijk->ik", w, FB[idx])
        FI /= np.clip(np.linalg.norm(FI, axis=1, keepdims=True), 1e-9, None)
    else:
        FI = np.zeros((0, FB.shape[1]))
    Fall = np.vstack([FB, FI]) if nI else FB

    # feature-weighted Laplacian over the triangulation edges
    edges = set()
    for t in keep:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.add((a, b) if a < b else (b, a))
    rows, cols, vals = [], [], []
    nV = len(uv)
    deg = np.zeros(nV)
    for a, b in edges:
        wij = np.exp(-(np.linalg.norm(Fall[a] - Fall[b]) ** 2) /
                     (2.0 * sigma_f * sigma_f + 1e-12))
        rows += [a, b]; cols += [b, a]; vals += [-wij, -wij]
        deg[a] += wij; deg[b] += wij
    rows += list(range(nV)); cols += list(range(nV)); vals += list(deg)
    L = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(nV, nV)))

    if nI == 0:
        # nothing interior to solve; just stitch the boundary fan
        new_xyz = np.zeros((0, 3))
        new_feat = np.zeros((0, feat.shape[1]))
    else:
        I = np.arange(nB, nV)
        B = np.arange(0, nB)
        L_II = L[I][:, I]
        L_IB = L[I][:, B]
        hI = spsolve(L_II.tocsc(), -(L_IB @ hB))
        hI = np.atleast_1d(np.asarray(hI))
        new_uv = uv[nB:]
        new_xyz = c + np.outer(new_uv[:, 0], u) + np.outer(new_uv[:, 1], v) \
            + np.outer(hI, n)
        new_feat = FI

    # remap triangle indices: boundary -> original loop ids; interior -> new
    loop = np.asarray(loop)
    remap = np.empty(nV, dtype=int)
    remap[:nB] = loop
    remap[nB:] = -1  # filled below by caller offset

    return new_xyz, new_feat, keep, remap, nB, n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Feature-gradient guided hole completion (post-process).")
    ap.add_argument("--input", "-i", required=True,
                    help="input mesh/point-cloud .ply (QEM vertices/mesh)")
    ap.add_argument("--output", "-o", default=None,
                    help="output .ply (default: <input>_filled.ply)")
    ap.add_argument("--dense", default=None,
                    help="dense cloud .ply to compute features on and transfer "
                         "(recommended over features on decimated vertices)")
    ap.add_argument("--features", default=None,
                    help="precomputed per-vertex features .npy (N x D)")
    ap.add_argument("--no-sonata", action="store_true",
                    help="use vertex normals as a stand-in feature (no GPU)")
    ap.add_argument("--feat-dim", type=int, default=32,
                    help="PCA dim of the Sonata feature (default 32)")
    ap.add_argument("--grid-size", type=float, default=None,
                    help="Sonata internal GridSample size (m). Default 0.02; "
                         "set ~ your QEM voxel size for sparse LiDAR vertices")
    ap.add_argument("--save-features", default=None,
                    help="cache the computed per-vertex features to this .npy "
                         "(reuse later with --features to skip re-inference)")
    ap.add_argument("--knn", type=int, default=16,
                    help="kNN for base meshing if input has no faces")
    ap.add_argument("--max-hole-edges", type=int, default=80,
                    help="skip boundary loops longer than this")
    ap.add_argument("--max-hole-diam-vox", type=float, default=10.0,
                    help="skip holes wider than this * median edge length")
    ap.add_argument("--max-planarity", type=float, default=0.05,
                    help="reject loops whose plane-fit residual ratio exceeds this")
    ap.add_argument("--coherence-sigma-scale", type=float, default=1.5,
                    help="fill only if max loop feature gradient < scale*sigma_f")
    ap.add_argument("--save-debug", action="store_true",
                    help="also write <output>_featgrad.ply colored by gradient")
    args = ap.parse_args()

    if o3d is None:
        raise SystemExit("open3d is required for mesh I/O: pip install open3d")

    out_ply = args.output or (os.path.splitext(args.input)[0] + "_filled.ply")

    # ---- load ----
    print(f"Loading: {args.input}")
    mesh = o3d.io.read_triangle_mesh(args.input)
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.triangles, np.int64)

    if verts.shape[0] == 0:
        raise SystemExit("No vertices found in input.")

    if faces.shape[0] == 0:
        print("  input has no faces -> building a base mesh "
              "(ball-pivoting) before hole filling")
        pcd = o3d.io.read_point_cloud(args.input)
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        d = np.asarray(pcd.compute_nearest_neighbor_distance())
        r = 3.0 * float(np.median(d))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, 2 * r, 4 * r]))
        verts = np.asarray(mesh.vertices, np.float32)
        faces = np.asarray(mesh.triangles, np.int64)
        print(f"  base mesh: {len(verts)} verts, {len(faces)} faces")

    mesh.compute_vertex_normals()
    vnormal = np.asarray(mesh.vertex_normals)
    vcolor = (np.asarray(mesh.vertex_colors) * 255.0
              if mesh.has_vertex_colors() else np.full_like(verts, 128.0))

    # ---- features ----
    print("Attaching per-vertex features ...")
    feat = attach_features(verts, vcolor, vnormal, args)
    if args.save_features:
        np.save(args.save_features, feat.astype(np.float32))
        print(f"  cached per-vertex features -> {args.save_features}")

    # ---- gradient calibration ----
    sigma_f = calibrate_sigma_f(faces, feat)
    edge_lens = []
    for t in faces[:: max(1, len(faces) // 5000 + 1)]:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_lens.append(np.linalg.norm(verts[a] - verts[b]))
    med_edge = float(np.median(edge_lens)) if edge_lens else 0.05
    print(f"  sigma_f = {sigma_f:.4f}   median edge = {med_edge:.4f} m")

    # ---- boundary loops ----
    loops = boundary_loops(faces)
    print(f"Found {len(loops)} open boundary loop(s)")

    new_xyz_all, new_feat_all, new_tris_all = [], [], []
    base_n = len(verts)
    filled, skipped = 0, 0
    for loop in loops:
        ok, basis = loop_is_fillable(loop, verts, feat, sigma_f, med_edge, args)
        if not ok:
            skipped += 1
            continue
        res = fill_loop(loop, verts, feat, basis, sigma_f, med_edge)
        new_xyz, new_feat, keep, remap, nB, plane_n = res
        if len(keep) == 0:
            skipped += 1
            continue
        offset = base_n + sum(len(x) for x in new_xyz_all)
        full_remap = remap.copy()
        full_remap[nB:] = offset + np.arange(len(new_xyz))
        tris = full_remap[keep]
        # orient new faces consistently with the loop plane normal
        for tri in tris:
            p0, p1, p2 = (verts[t] if t < base_n else
                          new_xyz[t - offset] for t in tri)
            fn = np.cross(p1 - p0, p2 - p0)
            if fn @ plane_n < 0:
                tri[1], tri[2] = tri[2], tri[1]
        new_xyz_all.append(new_xyz)
        new_feat_all.append(new_feat)
        new_tris_all.append(tris)
        filled += 1

    print(f"  filled {filled} loop(s), skipped {skipped} "
          f"(real boundaries / too large / non-planar)")

    # ---- assemble output ----
    if new_xyz_all:
        add_xyz = np.vstack(new_xyz_all)
        add_tris = np.vstack(new_tris_all)
        out_verts = np.vstack([verts, add_xyz])
        out_faces = np.vstack([faces, add_tris])
    else:
        out_verts, out_faces = verts, faces

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(out_verts.astype(np.float64))
    out.triangles = o3d.utility.Vector3iVector(out_faces.astype(np.int32))
    out.remove_duplicated_vertices()
    out.remove_degenerate_triangles()
    out.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, out)
    print(f"Saved completed mesh -> {out_ply}  "
          f"({len(out_verts)} verts, {len(out_faces)} faces)")

    # ---- debug: feature-gradient coloring ----
    if args.save_debug:
        counts = edge_face_counts(faces)
        gmax = np.zeros(len(verts))
        for (a, b) in counts.keys():
            g = np.linalg.norm(feat[a] - feat[b]) / max(
                np.linalg.norm(verts[a] - verts[b]), 1e-6)
            gmax[a] = max(gmax[a], g)
            gmax[b] = max(gmax[b], g)
        g = gmax / (np.percentile(gmax, 95) + 1e-9)
        g = np.clip(g, 0, 1)
        col = np.stack([g, np.zeros_like(g), 1 - g], axis=1)  # blue->red
        dbg = o3d.geometry.PointCloud()
        dbg.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
        dbg.colors = o3d.utility.Vector3dVector(col)
        dbg_path = os.path.splitext(out_ply)[0] + "_featgrad.ply"
        o3d.io.write_point_cloud(dbg_path, dbg)
        print(f"Saved feature-gradient debug -> {dbg_path}")


if __name__ == "__main__":
    main()
