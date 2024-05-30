_base_ = [
    "mmlab/mmpretrain/configs/_base_/models/efficientnet_b0.py",
    "mmlab/mmpretrain/configs/_base_/default_runtime.py",
]

# Model settings
model = dict(
    head=dict(
        num_classes=10,  # MNIST has 10 classes
        topk=(1,),
    ),
)

load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth"


# dataset settings
data_preprocessor = dict(
    mean=[0.1307],
    std=[0.3081],
    # convert image from BGR to RGB
    to_rgb=True,
)


# Data pipelines
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(28, 28)),  # Resize to MNIST dimensions
    dict(type="Normalize", **data_preprocessor),  # Normalize with MNIST stats
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(28, 28)),  # Resize to MNIST dimensions
    dict(type="Normalize", **data_preprocessor),  # Normalize with MNIST stats
    dict(type="PackInputs"),
]

# Dataloaders
train_dataloader = dict(
    batch_size=256,  # Adjust batch size as per your system capabilities
    shuffle=True,
    dataset=dict(
        type="CustomDataset",
        data_prefix="data/train",
        with_label=True,
        pipeline=train_pipeline,
    ),
    num_workers=8,  # Adjust num_workers as per your system capabilities
)

val_dataloader = dict(
    batch_size=256,  # Adjust batch size as per your system capabilities
    shuffle=False,
    dataset=dict(
        type="CustomDataset",
        data_prefix="data/val",
        with_label=True,
        pipeline=test_pipeline,
    ),
    num_workers=8,  # Adjust num_workers as per your system capabilities
)

# Training settings
train_cfg = dict(
    by_epoch=True,
    max_epochs=6,  # Increase the number of epochs if needed
    val_interval=2,
)
val_cfg = dict()
val_evaluator = dict(type="Accuracy", topk=(1,))

# Testing settings
test_cfg = None

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=0.001, weight_decay=0.0001)
)

# Checkpoint settings
checkpoint_config = dict(
    interval=-1,  # Save checkpoint at the end of the training
    save_last=True,  # Save the last checkpoint
    save_best="auto",  # Automatically save the best checkpoint
    rule="less",  # Rule for select best checkpoint: less for loss,
    max_keep_ckpts=1,  # Maximum number of checkpoints to keep
)

# Add checkpoint hook
custom_hooks = [
    dict(type='CheckpointHook', **checkpoint_config),
]