_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 80  # bs: total bs in all gpus
batch_size_val = 80
mix_prob = 0.0  # dunno yet what it does... something in dataloader collate_fn
empty_cache = False
enable_amp = True
amp_dtype = "float16"
enable_wandb = True
use_step_logging = True
log_every = 500

# dataset settings
dataset_type = "Gim3DSegmentationDataset"
seg_key = "smpl"
data_root = "data/gim3d"
class_names = ["no-smpl", "smpl"]
num_classes = len(class_names)

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=num_classes,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 100
eval_epoch = 100

# optimizer = dict(type="SGD", lr=0.5)
# final_lr = 9
# milestones = [0.05*i for i in range(1, 20)]
# gamma = (final_lr / optimizer["lr"]) ** (
#     1 / len(milestones)
# )  # `start_lr * x**len(milestones) = end_lr` <-> `x = (end_lr/start_lr) ** len(milestones)`
# scheduler = dict(
#     type="MultiStepWithWarmupLR",
#     total_steps=33 * epoch,
#     gamma=gamma,
#     milestones=milestones,
# )

optimizer = dict(type="AdamW", lr=1.0, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=100.0,
    final_div_factor=10_000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="SemSegEvaluatorTrain", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# dataset settings (continue)
data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        seg_key=seg_key,
        split="train.json",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeCoord"),
            # dict(type="RandomJitter"), # noise
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="z", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", always_apply=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.5, 0.5), (-0.2, 0.2))),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        seg_key=seg_key,
        split="val.json",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        seg_key=seg_key,
        split="test.json",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeCoord"),
        ],
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
    ),
)
