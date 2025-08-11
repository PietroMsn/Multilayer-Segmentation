from os import path
import argparse

import open3d as o3d
import numpy as np
import colorsys
import tqdm
import matplotlib.pyplot as plt

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets.builder import build_dataset

COLOR_MAP = np.array(
    [
        [0.5, 0.5, 0.5],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)

RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
GRAY = [0.5, 0.5, 0.5]
WHITE = [1.0, 1.0, 1.0]
YELLOW = [1.0, 1.0, 0.0]
PURPLE = [0.5, 0, 1]
ORANGE = [1.0, 0.5, 0.0]
BLACK = [0.0, 0.0, 0.0]
MAGENTA = [1.0, 0.0, 1.0]
LIGHT_BLUE = [0.7, 0.7, 1.0]
LIME = [0.7, 1.0, 0.0]
PINK = [1.0, 0.7, 0.8]

COLOR_EXPERIMENTS = [
    np.array(x)
    for x in [
        [GRAY, YELLOW],  # type 2
        [GRAY, RED],
        [GRAY, BLUE],
        [GRAY, YELLOW],  # type 3
        [GRAY, RED, BLUE],
        [GRAY, BLUE],
        [GRAY, YELLOW],  # type 4
        [GRAY, RED, ORANGE, YELLOW],
        [GRAY, BLUE, LIGHT_BLUE, GREEN],
        [GRAY, YELLOW],  # type 5
        [GRAY, ORANGE, LIGHT_BLUE, BLUE, RED, YELLOW, GREEN],
        [GRAY, GREEN, LIGHT_BLUE, BLUE],
    ]
]


def labels_to_colors(labels, exp=None):
    if exp is None:
        _, indices = np.unique(labels, return_inverse=True)
        num_unique = len(np.unique(labels))
        normalized = indices / max(1, num_unique - 1)
        cmap = plt.get_cmap("tab20")
        return cmap(normalized)[:, :3]
    else:
        return COLOR_EXPERIMENTS[exp][labels]


def visualize_segmentations(coords, normals, ground_truth, predicted):
    assert len(ground_truth) == len(predicted)

    pcds = []
    spread = np.ptp(coords, axis=0)  # (3,)

    for i, (gt, pred) in enumerate(zip(ground_truth, predicted)):
        # Create two point cloud objects for ground truth and prediction
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pred = o3d.geometry.PointCloud()

        gt_coords = coords.copy()
        gt_coords[:, 0] += spread[0] * 1.5 * i

        pred_coords = coords.copy()
        pred_coords[:, 0] += spread[0] * 1.5 * i
        pred_coords[:, 1] += spread[1] * 1.5

        # Assign coordinates and normals to both point clouds
        pcd_gt.points = o3d.utility.Vector3dVector(gt_coords)
        pcd_gt.normals = o3d.utility.Vector3dVector(normals)
        # pcd_gt.colors = o3d.utility.Vector3dVector(COLOR_MAP[gt].astype(np.float32))
        pcd_gt.colors = o3d.utility.Vector3dVector(labels_to_colors(gt))

        pcd_pred.points = o3d.utility.Vector3dVector(pred_coords)
        pcd_pred.normals = o3d.utility.Vector3dVector(normals)
        # pcd_pred.colors = o3d.utility.Vector3dVector(COLOR_MAP[pred].astype(np.float32))
        pcd_pred.colors = o3d.utility.Vector3dVector(labels_to_colors(pred))

        pcds.extend((pcd_gt, pcd_pred))

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Segmentation Visualization", width=1200, height=800)
    for g in pcds:
        vis.add_geometry(g)

    # Start visualization
    vis.run()
    vis.destroy_window()


def visualize_segmentations_no_gt(coords, normals, predicted, name=None, skip=[]):
    pcds = []
    spread = np.ptp(coords, axis=0)  # (3,)

    for i, pred in enumerate(predicted):
        if i not in skip:
            pcd = o3d.geometry.PointCloud()

            xyz = coords.copy()
            xyz[:, 0] += spread[0] * 1.5 * i

            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(labels_to_colors(pred, exp=i))

            pcds.append(pcd)

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Segmentation Visualization" if name is None else name,
        width=1200,
        height=800,
    )
    for g in pcds:
        vis.add_geometry(g)

    # Start visualization
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d")
    parser.add_argument("--exp_name", "-n", nargs="+")
    parser.add_argument("--show_gt", default=False, action="store_true")
    parser.add_argument("--skip", type=int, nargs="*", default=[])
    parser.add_argument("--name", type=str, nargs="?")
    args = parser.parse_args()

    datasets = []
    results_dirs = []
    for exp in args.exp_name:
        cfg = default_config_parser(
            path.join("exp", args.dataset_name, exp, "config.py"), None
        )
        results_dirs.append(path.join("exp", args.dataset_name, exp, "result"))
        datasets.append(build_dataset(cfg.data.test))

    if args.name:
        i = datasets[0].data_list.index(path.join("data", args.dataset_name, args.name))

        data = [d.get_data(i) for d in datasets]
        colors = [
            np.load(path.join(res_dir, data[0]["name"] + "_pred.npy"))
            for res_dir in results_dirs
        ]
        colors = [color[:, i] for color in colors for i in range(color.shape[1])]

        if args.show_gt:
            segments = [d["segments"] for d in data]
            visualize_segmentations(
                data[0]["coord"], data[0]["normal"], segments, colors
            )
        else:
            visualize_segmentations_no_gt(
                data[0]["coord"],
                data[0]["normal"],
                colors,
                name=data[0]["name"],
                skip=args.skip,
            )
    else:
        for i in tqdm.trange(len(datasets[0])):
            data = [d.get_data(i) for d in datasets]
            colors = [
                np.load(path.join(res_dir, data[0]["name"] + "_pred.npy"))
                for res_dir in results_dirs
            ]
            colors = [color[:, i] for color in colors for i in range(color.shape[1])]

            if args.show_gt:
                segments = [d["segments"] for d in data]
                visualize_segmentations(
                    data[0]["coord"], data[0]["normal"], segments, colors
                )
            else:
                visualize_segmentations_no_gt(
                    data[0]["coord"],
                    data[0]["normal"],
                    colors,
                    name=data[0]["name"],
                    skip=args.skip,
                )
