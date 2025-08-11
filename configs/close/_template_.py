_base_ = ["../gim3d/X.py"]

weight = "exp/gim3d/X/model/model_best.pth"
save_path = "exp/close/X"

data = dict(
    test=dict(
        type="CloSeDataset",
        map_labels=dict(
            tshirt=1,
            shorts=3,
            long_pants=3,
            long_shirts=1,
            top=1,
            skirt=3,
            empty=0,
        ),
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
