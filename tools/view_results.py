import argparse
import sys
from os import path

import numpy as np
import tqdm
import open3d as o3d

from pointcept.utils.visualization import get_point_cloud
from pointcept.engines.defaults import default_config_parser
from pointcept.datasets.builder import build_dataset


COLORS = dict(
    RED=[1.0, 0.0, 0.0],
    GREEN=[0.0, 1.0, 0.0],
    BLUE=[0.0, 0.0, 1.0],
    GRAY=[0.5, 0.5, 0.5],
    WHITE=[1.0, 1.0, 1.0],
    YELLOW=[1.0, 1.0, 0.0],
    PURPLE=[0.5, 0, 1],
    ORANGE=[1.0, 0.5, 0.0],
    BLACK=[0.0, 0.0, 0.0],
    MAGENTA=[1.0, 0.0, 1.0],
    LIGHT_BLUE=[0.7, 0.7, 1.0],
    LIME=[0.7, 1.0, 0.0],
    PINK=[1.0, 0.7, 0.8],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d")
    parser.add_argument("--exp_name", "-n")
    parser.add_argument("--layer", "-l", nargs="?", type=int)
    parser.add_argument("--name", nargs="?")
    parser.add_argument("--save_path", nargs="?")
    parser.add_argument("--colors", nargs="+")
    parser.add_argument("--noshow", action="store_true", default=False)
    parser.add_argument("--category", type=int, nargs="?")
    args = parser.parse_args()

    cfg = default_config_parser(
        path.join("exp", args.dataset_name, args.exp_name, "config.py"), None
    )
    res_dir = path.join("exp", args.dataset_name, args.exp_name, "result")
    test_dataset = build_dataset(cfg.data.test)

    if args.colors:
        COLOR_MAP = np.array([COLORS[c.upper()] for c in args.colors])
    else:
        COLOR_MAP = np.array([v for _, v in COLORS.items()])

    if args.name:
        idx = test_dataset.data_list.index(
            path.join("data", args.dataset_name, args.name)
        )
        data = test_dataset.get_data(idx)
        if args.layer is not None:
            color = np.load(path.join(res_dir, data["name"] + "_pred.npy"))[
                :, args.layer
            ]
        else:
            color = np.load(path.join(res_dir, data["name"] + "_pred.npy"))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["coord"])
        pcd.colors = o3d.utility.Vector3dVector(COLOR_MAP[color].astype(np.float32))
        pcd.normals = o3d.utility.Vector3dVector(data["normal"])

        if not args.noshow:
            o3d.visualization.draw_geometries([pcd])

        if args.save_path:
            o3d.io.write_point_cloud(args.save_path, pcd)
            print(f"Saved point cloud to {args.save_path}.")

    else:
        for i in (t := tqdm.trange(len(test_dataset))):
            data = test_dataset.get_data(i)

            cat = np.load(
                path.join("data", args.dataset_name, data["name"], "category.npy")
            ).item()

            if args.category is not None and cat != args.category:
                continue

            t.write(f'name: {data["name"]}')
            if args.layer is not None:
                color = np.load(path.join(res_dir, data["name"] + "_pred.npy"))[
                    :, args.layer
                ]
            else:
                color = np.load(path.join(res_dir, data["name"] + "_pred.npy"))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["coord"])
            pcd.colors = o3d.utility.Vector3dVector(COLOR_MAP[color].astype(np.float32))
            pcd.normals = o3d.utility.Vector3dVector(data["normal"])

            o3d.visualization.draw_geometries([pcd])

            # get_point_cloud(data["coord"], COLOR_MAP[color].astype(np.float32), verbose=True)
