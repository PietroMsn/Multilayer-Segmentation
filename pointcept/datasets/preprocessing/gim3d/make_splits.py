import os
from os import path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_folder",
        help="Original folder containing `{train/test/val}.json` files",
    )
    parser.add_argument(
        "--output_folder", help="Output folder where to save split files."
    )
    args = parser.parse_args()
    for split in ("train", "test", "val"):
        with open(path.join(args.split_folder, split + ".json")) as f:
            data = json.load(f)
        names = [n.replace("_multilayer_smpl", "") for n in data]
        with open(path.join(args.output_folder, split + ".json")) as f:
            json.dump(names, f)
    print(f"Done saving splits to {args.output_folder}.")


if __name__ == "__main__":
    main()
