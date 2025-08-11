import argparse

import tqdm
import numpy as np

from pointcept.engines.defaults import (
    default_config_parser,
)
from pointcept.datasets.builder import build_dataset
from pointcept.utils.visualization import get_point_cloud

COLORS = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "black": [0, 0, 0],
    "white": [1, 1, 1],
    "gray": [0.5, 0.5, 0.5],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--colors", type=str, nargs="+")
    args = parser.parse_args()
    cfg = default_config_parser(args.config_file, None)

    dataset = build_dataset(cfg.data.train)

    if args.colors:
        COLOR_MAP = np.array([COLORS[c] for c in args.colors])
    else:
        COLOR_MAP = np.array(COLORS.values())

    for i in tqdm.trange(len(dataset)):
        data = dataset.get_data(i)
        get_point_cloud(
            data["coord"], COLOR_MAP[data["segments"][args.layer]], verbose=True
        )


if __name__ == "__main__":
    main()
