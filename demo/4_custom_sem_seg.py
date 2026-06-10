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
Semantic segmentation on a custom point cloud file with Sonata (PTv3) + a
ScanNet linear-probed head, saving the segmented point cloud to disk.

Example:
    export PYTHONPATH=./
    python demo/4_custom_sem_seg.py \
        --input  /path/to/your_scene.ply \
        --output /path/to/your_scene_seg.ply

Supported input formats:
    * .ply / .pcd / .xyz / .pts  (read via open3d; color optional, normals
      are estimated automatically when missing)
    * .npz / .npy                (must provide a "coord" array, and optionally
      "color" in 0-255 and "normal")

Outputs (next to --output):
    * <output>.ply   colored point cloud (one color per predicted class)
    * <output>.npz   {coord, color, normal, segment(label id), label(name)}
"""

import argparse
import os

import numpy as np
import open3d as o3d
import sonata
import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ----------------------------------------------------------------------------
# ScanNet-20 meta data (must match the linear-probed head we load below)
# ----------------------------------------------------------------------------
VALID_CLASS_IDS_20 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
)

CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    16: (219.0, 219.0, 141.0),
    24: (255.0, 127.0, 14.0),
    28: (158.0, 218.0, 229.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    36: (227.0, 119.0, 194.0),
    39: (82.0, 84.0, 163.0),
}

CLASS_COLOR_20 = np.array([SCANNET_COLOR_MAP_20[i] for i in VALID_CLASS_IDS_20])


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


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
        description="Run Sonata semantic segmentation on a custom point cloud."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="path to input point cloud file"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="path to output .ply (default: <input>_seg.ply)",
    )
    parser.add_argument(
        "--grid-size", type=float, default=0.02,
        help="grid sample size in meters (default: 0.02, matches pre-training)",
    )
    args = parser.parse_args()

    out_ply = args.output or (
        os.path.splitext(args.input)[0] + "_seg.ply"
    )
    out_npz = os.path.splitext(out_ply)[0] + ".npz"

    sonata.utils.set_seed(24525867)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Model + linear-probed ScanNet head ----
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

    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])

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
    num_points = original_coord.shape[0]
    print(f"  {num_points} points loaded")
    point = transform(point)

    # ---- Inference ----
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(device, non_blocking=True)
        point = model(point)
        # unpool encoder features back to the first (grid) level
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        seg_logits = seg_head(point.feat)
        pred_grid = seg_logits.argmax(dim=-1)
        # map grid-level predictions back to every original input point
        pred = pred_grid[point.inverse].cpu().numpy()

    seg_color = CLASS_COLOR_20[pred]

    # ---- Report class distribution ----
    print("Predicted class distribution:")
    ids, counts = np.unique(pred, return_counts=True)
    for cid, cnt in sorted(zip(ids, counts), key=lambda x: -x[1]):
        print(f"  {CLASS_LABELS_20[cid]:>15s}: {cnt:>8d} ({100.0 * cnt / num_points:5.1f}%)")

    # ---- Save ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(seg_color / 255.0)
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved segmented point cloud -> {out_ply}")

    np.savez(
        out_npz,
        coord=original_coord,
        color=seg_color.astype(np.uint8),
        segment=pred.astype(np.int32),
        label=np.array([CLASS_LABELS_20[i] for i in pred]),
        valid_class_ids=np.array(VALID_CLASS_IDS_20),
        class_labels=np.array(CLASS_LABELS_20),
    )
    print(f"Saved per-point labels      -> {out_npz}")


if __name__ == "__main__":
    main()
