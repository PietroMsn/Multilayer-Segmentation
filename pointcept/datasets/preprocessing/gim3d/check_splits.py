import os
from os import path
import argparse
import json
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        help="Folder containing dataset and `{train/test/val}.json` files",
    )
    args = parser.parse_args()

    missing = 0
    counts = dict()

    for split in ("train", "test", "val"):
        with open(path.join(args.dataset_folder, split + ".json")) as f:
            data = json.load(f)
        tot = 0
        for fn in data:
            if not path.exists(path.join(args.dataset_folder, fn)):
                print(f"Error: {fn} folder not found!")
                missing += 1
            else:
                tot += 1
        counts[split] = tot

    print(f"Found {missing} missing folders.")
    print(f"Dataset splits:")
    pprint(counts)


if __name__ == "__main__":
    main()
