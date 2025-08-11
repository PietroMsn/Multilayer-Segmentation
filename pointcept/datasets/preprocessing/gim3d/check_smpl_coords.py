from os import path
import glob
import argparse
import multiprocessing as mp

import numpy as np
from tqdm.contrib.concurrent import process_map


def test_dir(dirname):
    if not path.exists(path.join(dirname, "coord.npy")):
        return -1
    if not path.exists(path.join(dirname, "coord_smpl.npy")):
        return -2

    coords = np.load(path.join(dirname, "coord.npy"))
    coords2 = np.load(path.join(dirname, "coord_smpl.npy"))
    if not np.allclose(coords, coords2):
        return -3
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    args = parser.parse_args()

    dirs = glob.glob(path.join(args.dataset_path, "*Gim3d*"))

    results = process_map(
        test_dir,
        dirs,
        max_workers=args.num_workers,
    )

    results = np.array(results)

    idxs = np.argwhere(results == -1).reshape(-1)
    print("No `coord.npy`:")
    for i in range(idxs.shape[0]):
        print(path.basename(dirs[idxs[i]]))

    idxs = np.argwhere(results == -2).reshape(-1)
    print("No `coord_smpl.npy`:")
    for i in range(idxs.shape[0]):
        print(path.basename(dirs[idxs[i]]))

    idxs = np.argwhere(results == -3).reshape(-1)
    print("Coords not matching:")
    for i in range(idxs.shape[0]):
        print(path.basename(dirs[idxs[i]]))
