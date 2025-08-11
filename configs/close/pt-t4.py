_base_ = ["../gim3d/pt-t4.py"]

weight = "exp/gim3d/pt-t4/model/model_best.pth"
save_path = "exp/close/pt-t4"

map_labels = [
    dict(tshirt=0, shorts=0, long_pants=0, long_shirts=0, top=0, skirt=0, empty=0),
    dict(tshirt=0, shorts=0, long_pants=0, long_shirts=0, top=0, skirt=0, empty=0),
    dict(tshirt=0, shorts=0, long_pants=0, long_shirts=0, top=0, skirt=0, empty=0),
]

data = dict(
    test=dict(
        _delete_=True,
        type="CloSeDatasetMultiHead",
        map_labels=map_labels,
        split="all.json",
        data_root="data/close",
        transform=None,
        test_mode=True,
        test_cfg=dict(
            voxelize=None,
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "normal"),
                ),
            ],
            aug_transform=[  # do nothing
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=0,
                    )
                ]
            ],
        ),
    )
)
