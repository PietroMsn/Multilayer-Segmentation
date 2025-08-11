import os
from os import path
import argparse
import multiprocessing as mp
import glob
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import re

import open3d as o3d
import numpy as np
from tqdm.contrib.concurrent import process_map

from utils import remove_unconnected_parts, create_point_cloud_from_vertices

RED, GREEN, BLUE = [1, 0, 0], [0, 1, 0], [0, 0, 1]


def save_pc(fn, out_dir, num_vert):
    frame_type = re.findall(r"^(Gim3d(?:Fr\d+)?).*", fn.split(path.sep)[-3])[0]
    name = f"{path.basename(fn)[:5]}_{frame_type}"

    noisy_mesh = o3d.io.read_triangle_mesh(fn)
    mesh = remove_unconnected_parts(noisy_mesh)
    pcd = create_point_cloud_from_vertices(
        np.asarray(mesh.vertices), mesh.vertex_normals, list(mesh.vertex_colors)
    )
    reduced_pcd = pcd.farthest_point_down_sample(num_vert)
    colors = np.asarray(reduced_pcd.colors)
    segment = np.zeros(colors.shape[0])
    for c, label in zip((RED, GREEN, BLUE), (1, 2, 3)):
        mask = np.all(colors == c, axis=1)
        segment[mask] = label

    os.makedirs(path.join(out_dir, name), exist_ok=True)
    np.save(path.join(out_dir, name, "coord.npy"), np.asarray(reduced_pcd.points))
    np.save(path.join(out_dir, name, "normal.npy"), np.asarray(reduced_pcd.normals))
    np.save(path.join(out_dir, name, "segment.npy"), segment)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the gim3d dataset root (where `Gim3d*_scanned_aligned_on_smpl` folders are).",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where files will be located.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--num_verts",
        default=6000,
        type=int,
        help="Number of vertex to subsample the point cloud. Default: 6000.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    fns = glob.glob(
        path.join(args.dataset_root, "Gim3d*_scanned_aligned_on_smpl", "**/*.ply"),
        recursive=True,
    )
    print(f"Found {len(fns)} files in total.")
    print(f"Using {args.num_workers} workers ({os.cpu_count()=} & {mp.cpu_count()=}).")

    print("Processing files...")
    if os.name == "posix":
        pool = ProcessPoolExecutor(
            max_workers=args.num_workers, mp_context=mp.get_context("forkserver")
        )
        _ = list(
            pool.map(
                save_pc,
                fns,
                repeat(args.output_root),
                repeat(args.num_verts),
            )
        )
    else:
        _ = process_map(
            save_pc,
            fns,
            repeat(args.output_root),
            repeat(args.num_verts),
            max_workers=args.num_workers,
        )
    print("Done creating files")
