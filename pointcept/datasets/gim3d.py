import os
from copy import deepcopy

import numpy as np

from pointcept.utils.cache import shared_dict

from .defaults import DefaultDataset
from .builder import DATASETS

CATEGORIES = {
    "long_short_skirt": 0,
    "longlong": 1,
    "longshort": 2,
    "short_short_skirt": 3,
    "shortlong": 4,
    "shortshort": 5,
    "top_short_skirt": 6,
    "toplong": 7,
    "topshort": 8,
}

CATEGORIES_NAMES = {
    0: "long_short_skirt",
    1: "longlong",
    2: "longshort",
    3: "short_short_skirt",
    4: "shortlong",
    5: "shortshort",
    6: "top_short_skirt",
    7: "toplong",
    8: "topshort",
}

SEG_CAT = {
    "other": 0,
    "t-shirt": 1,
    "shorts": 2,
    "long pants": 3,
    "long shirt": 4,
    "top": 5,
    "skirt": 6,
}

CAT_TO_UP_SEG = {
    "long_short_skirt": SEG_CAT["long shirt"],
    "longlong": SEG_CAT["long shirt"],
    "longshort": SEG_CAT["long shirt"],
    "short_short_skirt": SEG_CAT["t-shirt"],
    "shortlong": SEG_CAT["t-shirt"],
    "shortshort": SEG_CAT["t-shirt"],
    "top_short_skirt": SEG_CAT["top"],
    "toplong": SEG_CAT["top"],
    "topshort": SEG_CAT["top"],
}

CAT_TO_DOWN_SEG = {
    "long_short_skirt": SEG_CAT["skirt"],
    "longlong": SEG_CAT["long pants"],
    "longshort": SEG_CAT["shorts"],
    "short_short_skirt": SEG_CAT["skirt"],
    "shortlong": SEG_CAT["long pants"],
    "shortshort": SEG_CAT["shorts"],
    "top_short_skirt": SEG_CAT["skirt"],
    "toplong": SEG_CAT["long pants"],
    "topshort": SEG_CAT["shorts"],
}


@DATASETS.register_module()
class Gim3DClassificationDataset(DefaultDataset):
    VALID_ASSETS = DefaultDataset.VALID_ASSETS + ["category"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def build_fn(cfg):
    def gen_segment(smpl, segment, cat):
        out = np.zeros_like(segment) - 1
        for line in reversed(cfg):
            # smpl
            match line[0]:
                case "smpl":
                    smpl_mask = smpl == 1.0
                case "no-smpl":
                    smpl_mask = smpl == 0.0
                case -1:
                    smpl_mask = np.ones(segment.shape[0], dtype=bool)
                case x:
                    raise ValueError(f"Invalid smpl layers config: `{x}`")

            # seg
            match line[1]:
                case "body":
                    seg_mask = segment == 0
                case "upper":
                    seg_mask = segment == 1
                case "overlap":
                    seg_mask = segment == 2
                case "lower":
                    seg_mask = segment == 3
                case -1:
                    seg_mask = np.ones(segment.shape[0], dtype=bool)
                case x:
                    raise ValueError(f"Invalid seg layers config: `{x}`")

            mask = smpl_mask & seg_mask

            if isinstance(line[3], int) and isinstance(line[2], int):
                if line[2] >= 0 and line[2] != cat:
                    continue
                else:
                    segval = line[3]
            elif isinstance(line[3], str) and isinstance(line[2], str):
                if line[3].startswith("upcls"):
                    segval = CAT_TO_UP_SEG[CATEGORIES_NAMES[cat.item()]]
                elif line[3].startswith("downcls"):
                    segval = CAT_TO_DOWN_SEG[CATEGORIES_NAMES[cat.item()]]
                else:
                    raise ValueError(f"Unknown in segmentation label: {line[3]}")
            elif isinstance(line[3], int) and isinstance(line[2], str):
                if line[2] not in CATEGORIES:
                    raise ValueError(
                        f"Invalid config, `{line[2]}` is not a valid category."
                    )
                if CATEGORIES[line[2]] != cat:
                    continue
                else:
                    segval = line[3]
            else:
                raise ValueError(
                    f"Incompatible category and segmentation label: `{line[2]}` and `{line[3]}`"
                )

            out[mask] = segval

        if np.any(out == -1):
            raise ValueError(
                f"Unexhaustive pattern when generating layers ({len(cfg)=})."
            )

        return out

    return gen_segment


@DATASETS.register_module()
class Gim3DMultiHeadSegmentationDataset(DefaultDataset):
    VALID_ASSETS = DefaultDataset.VALID_ASSETS + ["category", "smpl"]

    def __init__(self, *args, layers_cfg, first_n=None, **kwargs):
        self.first_n = first_n
        super().__init__(*args, **kwargs)
        self.layers_fns = []
        for cfg in layers_cfg:
            self.layers_fns.append(build_fn(cfg))

    def get_data_list(self):
        data_list = super().get_data_list()
        if self.first_n:
            return data_list[: self.first_n]
        else:
            return data_list

    def get_data(self, idx):
        # TODO: cache
        data_dict = super().get_data(idx)
        segments = []
        for fn in self.layers_fns:
            segments.append(
                fn(data_dict["smpl"], data_dict["segment"], data_dict["category"])
            )
        data_dict["segments"] = segments
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segments=data_dict.pop("segments"), name=data_dict.pop("name")
        )
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict


@DATASETS.register_module()
class Gim3DSegmentationDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "smpl",
        "category",
    ]

    def __init__(
        self, *args, seg_key="segment", custom_layer_cfg=None, first_n=None, **kwargs
    ):
        self.first_n = first_n

        super().__init__(*args, **kwargs)

        if not seg_key in ["segment", "smpl"]:
            raise ValueError("`seg_key` should be either `segment` or `smpl`")

        if seg_key != "segment" and custom_layer_cfg is not None:
            raise ValueError(
                "`seg_key` must be `segment` if you set `custom_layer_cfg`."
            )

        self.seg_fn = (
            build_fn(custom_layer_cfg) if custom_layer_cfg is not None else None
        )

        self.seg_key = seg_key

    def get_data_list(self):
        data_list = super().get_data_list()
        if self.first_n:
            return data_list[: self.first_n]
        else:
            return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "smpl" in data_dict.keys():
            data_dict["smpl"] = data_dict["smpl"].reshape([-1]).astype(np.int32)
        else:
            data_dict["smpl"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if self.seg_fn is not None:
            data_dict["segment"] = self.seg_fn(
                data_dict["smpl"], data_dict["segment"], data_dict["category"]
            )
        else:
            data_dict["segment"] = data_dict[self.seg_key]

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict
