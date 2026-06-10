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
PCA feature visualization on a custom point cloud file with Sonata (PTv3),
saving the PCA-colored point cloud to disk.

Unlike semantic segmentation, this is unsupervised: it colors each point by a
PCA projection of the learned Sonata features, so visually-similar structures
get similar colors. It works on any point cloud (indoor or outdoor) since it
does not rely on a dataset-specific classification head.

Example:
    export PYTHONPATH=./
    python demo/5_custom_pca.py \
        --input  /path/to/your_scene.ply \
        --output /path/to/your_scene_pca.ply

Supported input formats:
    * .ply / .pcd / .xyz / .pts  (read via open3d; color optional, normals
      are estimated automatically when missing)
    * .npz / .npy                (must provide a "coord" array, and optionally
      "color" in 0-255 and "normal")

Outputs (next to --output):
    * <output>.ply   point cloud colored by PCA of Sonata features
    * <output>.npz   {coord, color} of the saved cloud
"""

import argparse
import os

import numpy as np
import open3d as o3d
import sonata
import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


def load_point_cloud(path):
    """Load a custom point cloud into the dict format Sonata expects.

    Returns a dict with keys: coord (N,3 float32), color (N,3 float32 in
    0-255), normal (N,3 float32). Color/normal are estimated/filled when the
    source file does not provide them.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".npz", ".npy"):
        raw = np.load(path)
        raw = dict(raw) if hasattr(raw, "keys") else {"coord": raw}
        assert "coord" in raw, ".npz/.npy input must contain a 'coord' array"
        coord = np.asarray(raw["coord"], dtype=np.float32)
        color = raw.get("color", None)
        normal = raw.get("normal", None)
        pcd = None
    else:
        pcd = o3d.io.read_point_cloud(path)
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors) * 255.0 if pcd.has_colors() else None
        normal = np.asarray(pcd.normals) if pcd.has_normals() else None

    assert coord.shape[0] > 0, f"No points found in {path}"

    if color is None:
        print("  [info] no color in input -> using mid-gray (128) for all points")
        color = np.full_like(coord, 128.0)
    color = np.asarray(color, dtype=np.float32)

    if normal is None:
        print("  [info] no normals in input -> estimating with open3d")
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        normal = np.asarray(pcd.normals)
    normal = np.asarray(normal, dtype=np.float32)

    return {"coord": coord, "color": color, "normal": normal}


def main():
    parser = argparse.ArgumentParser(
        description="Run Sonata PCA feature visualization on a custom point cloud."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="path to input point cloud file"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="path to output .ply (default: <input>_pca.ply)",
    )
    parser.add_argument(
        "--grid-size", type=float, default=0.02,
        help="grid sample size in meters (default: 0.02, matches pre-training)",
    )
    parser.add_argument(
        "--brightness", type=float, default=1.2,
        help="PCA color brightness scale (default: 1.2)",
    )
    args = parser.parse_args()

    out_ply = args.output or (os.path.splitext(args.input)[0] + "_pca.ply")
    out_npz = os.path.splitext(out_ply)[0] + ".npz"

    # (random seed affects pca color)
    sonata.utils.set_seed(53124)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Model ----
    if flash_attn is not None and device == "cuda":
        model = sonata.load("sonata", repo_id="facebook/sonata").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).to(device)

    # ---- Transform pipeline (grid size configurable) ----
    transform = sonata.transform.default()
    if args.grid_size != 0.02:
        for t in transform.transforms:
            if type(t).__name__ == "GridSample":
                t.grid_size = args.grid_size

    # ---- Data ----
    print(f"Loading point cloud: {args.input}")
    point = load_point_cloud(args.input)
    original_coord = point["coord"].copy()
    print(f"  {original_coord.shape[0]} points loaded")
    point = transform(point)

    # ---- Inference ----
    model.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(device, non_blocking=True)
        point = model(point)
        # upcast encoder features back to the first (grid) level
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        # PCA on grid-level features
        pca_color = get_pca_color(point.feat, brightness=args.brightness, center=True)

    # map grid-level colors back to every original input point
    original_pca_color = pca_color[point.inverse].cpu().detach().numpy()

    # ---- Save ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(original_pca_color)
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved PCA-colored point cloud -> {out_ply}")

    np.savez(
        out_npz,
        coord=original_coord,
        color=(original_pca_color * 255.0).astype(np.uint8),
    )
    print(f"Saved PCA color array         -> {out_npz}")


if __name__ == "__main__":
    main()
