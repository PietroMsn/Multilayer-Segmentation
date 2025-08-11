import argparse
import glob
from os import path
import re
import json

import numpy as np
from tqdm import tqdm


def main():
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
    args = parser.parse_args()

    original_fns = glob.glob(
        path.join(args.dataset_root, "Gim3d*_scanned_aligned_on_smpl", "**/*.ply"),
        recursive=True,
    )

    with open("pointcept/datasets/preprocessing/gim3d/categories.json") as f:
        categories_mapping = json.load(f)

    for fn in tqdm(original_fns):
        comps = fn.split(path.sep)
        fn_id = comps[-1][:5]
        cat = comps[-2]
        frame_type = re.findall(r"^(Gim3d(?:Fr\d+)?).*", comps[-3])[0]
        name = f"{fn_id}_{frame_type}"

        assert cat in categories_mapping

        np.save(
            path.join(args.output_root, name, "category.npy"),
            np.array(categories_mapping[cat]).reshape(1),
        )


if __name__ == "__main__":
    main()
