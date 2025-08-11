_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 100  # bs: total bs in all gpus
batch_size_val = 100
mix_prob = 0.0  # dunno yet what it does... something in dataloader collate_fn
empty_cache = False
enable_amp = True
amp_dtype = "float16"
enable_wandb = True

# dataset settings
heads = [2, 2, 2]
heads_names = ["smpl", "upper", "lower"]
heads_class_names = [
    ["no-smpl", "smpl"],
    ["other", "upper body"],
    ["other", "lower body"],
]
dataset_type = "Gim3DMultiHeadSegmentationDataset"
data_root = "data/gim3d"
# fmt: off
layers_cfg = [
    [
        ["no-smpl", -1, -1, 0],
        [   "smpl", -1, -1, 1]
    ],
    [
        [-1,   "upper", -1, 1],
        [-1, "overlap", -1, 1],
        [-1,        -1, -1, 0],
    ],
    [
        [-1,   "lower", -1, 1],
        [-1, "overlap", -1, 1],
        [-1,        -1, -1, 0],
    ],
]
# fmt: on


# model settings
model = dict(
    type="MultiHeadDefaultSegmentor",
    backbone=dict(
        type="PointNetSeg",
        feat_dim=3,
        use_segm_head=False,
    ),
    heads=[
        dict(
            type="DefaultSegmentationHead",
            backbone_embed_dim=128,
            layers=[256, 256],
            num_classes=heads[0],
        ),
        dict(
            type="DefaultSegmentationHead",
            backbone_embed_dim=128,
            layers=[256, 256],
            num_classes=heads[1],
        ),
        dict(
            type="DefaultSegmentationHead",
            backbone_embed_dim=128,
            layers=[256, 256],
            num_classes=heads[2],
        ),
    ],
    criteria=[
        dict(
            type="MultiHeadLoss",
            losses=[
                dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
                dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
                dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
            ],
        )
    ],
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="MultiHeadSemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

default_tx = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment", "segments"),
        feat_keys=("normal",),
    ),
]

# Tester
test = dict(type="MultiHeadSemSegTester", verbose=True)

# dataset settings (continue)
data = dict(
    heads=heads,
    heads_names=heads_names,
    heads_class_names=heads_class_names,
    train=dict(
        type=dataset_type,
        split="train.json",
        data_root=data_root,
        transform=default_tx,
        test_mode=False,
        layers_cfg=layers_cfg,
    ),
    val=dict(
        type=dataset_type,
        split="val.json",
        data_root=data_root,
        transform=default_tx,
        test_mode=False,
        layers_cfg=layers_cfg,
    ),
    test=dict(
        type=dataset_type,
        split="test.json",
        data_root=data_root,
        transform=None,
        test_mode=True,
        layers_cfg=layers_cfg,
        test_cfg=dict(
            voxelize=None,
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("normal",),
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
    ),
)
