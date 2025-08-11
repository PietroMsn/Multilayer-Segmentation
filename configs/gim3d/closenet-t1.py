_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 40  # bs: total bs in all gpus
batch_size_val = 20
mix_prob = 0.0  # dunno yet what it does... something in dataloader collate_fn
empty_cache = False
enable_amp = True
amp_dtype = "float16"
enable_wandb = True

# dataset settings
dataset_type = "DefaultDataset"
data_root = "data/gim3d"
class_names = ["body (gray)", "upper (red)", "overlap (green)", "lower (blue)"]
num_classes = len(class_names)

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="CloSeNet",
        inp_dim=6,
        pts_emb_dim=1024,
        k=20,
        use_tnet=True,
        n_classes=num_classes,
        dropout=0.5,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
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
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

default_tx = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
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
