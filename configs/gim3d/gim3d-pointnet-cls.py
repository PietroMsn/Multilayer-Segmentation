_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 200
batch_size_val = 200
batch_size_test = 32
mix_prob = 0.0
empty_cache = False
enable_amp = True
amp_dtype = "float16"
enable_wandb = True

# Tester
test = dict(type="ClsTester")

# dataset settings
dataset_type = "Gim3DClassificationDataset"  # TODO
data_root = "data/gim3d"
class_names = [
    "long_short_skirt",
    "longlong",
    "longshort",
    "short_short_skirt",
    "shortlong",
    "shortshort",
    "top_short_skirt",
    "toplong",
    "topshort",
]
num_classes = len(class_names)

# model settings
model = dict(
    type="DefaultClassifier",
    backbone=dict(
        type="PointNetCls",
        feat_dim=3,
        embed_dim=1024,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    num_classes=num_classes,
    backbone_embed_dim=1024,
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
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

default_tx = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "category"),
        feat_keys=("normal",),
    ),
]

# dataset settings (continue)
data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train.json",
        data_root=data_root,
        transform=default_tx,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val.json",
        data_root=data_root,
        transform=default_tx,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test.json",
        data_root=data_root,
        transform=default_tx,
        test_mode=False,
        # test_cfg=dict(
        #     voxelize=None,
        #     crop=None,
        #     post_transform=default_tx,
        #     aug_transform=[  # do nothing
        #         [
        #             dict(
        #                 type="RandomRotateTargetAngle",
        #                 angle=[0],
        #                 axis="z",
        #                 center=[0, 0, 0],
        #                 p=0,
        #             )
        #         ]
        #     ],
        # ),
    ),
)
