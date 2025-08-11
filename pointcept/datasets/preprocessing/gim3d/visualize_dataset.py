"""
Given a split, open3d visualizations with color & normals.
Remark open3d shortcuts:
> press `1` to show colors
> press `9` to show normals as colors
> press `n` to toggle normals as lines
"""

from os import path
import argparse
import json

import numpy as np
import open3d as o3d

COLOR_MAP = np.array(
    [
        [0.5, 0.5, 0.5],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root")
    parser.add_argument("--split")
    args = parser.parse_args()

    print(__doc__)

    with open(path.join(args.dataset_root, args.split + ".json")) as f:
        data = json.load(f)

    for case in data:
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            np.load(path.join(args.dataset_root, case, "coord.npy"))
        )
        pcd.normals = o3d.utility.Vector3dVector(
            np.load(path.join(args.dataset_root, case, "normal.npy"))
        )
        labels = np.load(path.join(args.dataset_root, case, "segment.npy")).astype(int)
        pcd.colors = o3d.utility.Vector3dVector(COLOR_MAP[labels])

        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
