import os
from os import path
import glob
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import repeat
import argparse

import plyfile
import open3d as o3d
import numpy as np
from tqdm.contrib.concurrent import process_map
import fpsample

"""
For segmentation, we follow this convention:
0 | tshirt      | orange
1 | shorts      | light blue
2 | long pants  | blue
3 | long shirts | red
4 | top         | yellow
5 | skirt       | green
6 | empty       | black
"""

RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
GRAY = [128, 128, 128]
LIGHTBLUE = [179, 179, 255]
ORANGE = [255, 128, 0]
YELLOW = [255, 255, 0]

LABELS = [ORANGE, LIGHTBLUE, BLUE, RED, YELLOW, GREEN, GRAY]

# classification labels
cls = {
    "long_short_skirt": 0,
    "longlong": 1,
    "short_short_skirt": 2,
    "shortshort": 3,
    "topshort": 4,
}


def process(fn, out_dir, num_vert):
    with open(fn, "rb") as f:
        plydata = plyfile.PlyData.read(f)

    coord = np.column_stack(
        (
            plydata["vertex"]["x"],
            plydata["vertex"]["y"],
            plydata["vertex"]["z"],
        )
    )

    color = np.column_stack(
        (
            plydata["vertex"]["red"],
            plydata["vertex"]["green"],
            plydata["vertex"]["blue"],
        )
    )

    faces = np.stack(plydata["face"].data["vertex_indices"])

    # compute normals with open3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(coord)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    normal = np.asarray(mesh.vertex_normals)

    idx = fpsample.bucket_fps_kdline_sampling(coord, 6000, h=7)

    fps_coord = coord[idx]
    fps_color = color[idx]
    fps_normal = normal[idx]

    segment = np.zeros(fps_color.shape[0])
    segment.fill(-1)
    for i, label in enumerate(LABELS):
        mask = np.all(fps_color == label, axis=1)
        segment[mask] = i
    if np.any(segment == -1):
        # there's a gray at [125, 125, 125]
        mask = np.all(fps_color == [125, 125, 125], axis=1)
        segment[mask] = 6

    if np.any(segment == -1):
        return fn

    cat = cls[fn.split(path.sep)[-2]]
    name = path.splitext(path.split(fn)[1])[0]

    os.makedirs(path.join(out_dir, name), exist_ok=True)
    np.save(path.join(out_dir, name, "coord.npy"), fps_coord)
    np.save(path.join(out_dir, name, "normal.npy"), fps_normal)
    np.save(path.join(out_dir, name, "segment.npy"), segment)
    np.save(path.join(out_dir, name, "category.npy"), cat)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the close dataset root (should contain `<cat>/<mesh>.ply` files).",
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
        help="Num workers for preprocessing. Default: `mp.cpu_count()`.",
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
        path.join(args.dataset_root, "**/*.ply"),
        recursive=True,
    )
    fns = [fn for fn in fns if fn.split(path.sep)[-2] != "hoodies"]

    print(f"Found {len(fns)} files in total.")
    print(f"Using {args.num_workers} workers ({os.cpu_count()=} & {mp.cpu_count()=}).")

    print("Processing files...")

    if os.name == "posix":
        pool = ProcessPoolExecutor(
            max_workers=args.num_workers, mp_context=mp.get_context("forkserver")
        )
        data = list(
            pool.map(
                process,
                fns,
                repeat(args.output_root),
                repeat(args.num_verts),
            )
        )
    else:
        data = process_map(
            process,
            fns,
            repeat(args.output_root),
            repeat(args.num_verts),
            max_workers=args.num_workers,
        )
    print("Done creating files")

    data = [fn for fn in data if fn != 0]
    print("The following files have failed during processing:")
    for fn in data:
        print(fn)


if __name__ == "__main__":
    main()
