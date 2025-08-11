from copy import deepcopy

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS

LABELS = [
    "tshirt",
    "shorts",
    "long_pants",
    "long_shirts",
    "top",
    "skirt",
    "empty",
]


@DATASETS.register_module()
class CloSeDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "category",
    ]

    def __init__(self, *args, map_labels=None, first_n=None, **kwargs):
        self.first_n = first_n

        super().__init__(*args, **kwargs)

        if map_labels is not None:
            if not all(lab in LABELS for lab in map_labels.keys()):
                raise ValueError("All labels must be present in `map_labels`")
            self.map_labels = np.array([map_labels[k] for k in LABELS])
        else:
            self.map_labels = None

    def get_data_list(self):
        data_list = super().get_data_list()
        if self.first_n:
            return data_list[: self.first_n]
        else:
            return data_list

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        if self.map_labels is not None:
            data_dict["segment"] = self.map_labels[data_dict["segment"]]
        return data_dict


@DATASETS.register_module()
class CloSeDatasetMultiHead(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "category",
    ]

    def __init__(self, *args, map_labels=[], first_n=None, **kwargs):
        self.first_n = first_n

        super().__init__(*args, **kwargs)

        self.map_labels = []
        for ml in map_labels:
            if not all(lab in LABELS for lab in ml.keys()):
                raise ValueError("All labels must be present in `map_labels`")
            self.map_labels.append(np.array([ml[k] for k in LABELS]))

    def get_data_list(self):
        data_list = super().get_data_list()
        if self.first_n:
            return data_list[: self.first_n]
        else:
            return data_list

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        segments = []
        for ml in self.map_labels:
            segments.append(ml[data_dict["segment"]])
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
