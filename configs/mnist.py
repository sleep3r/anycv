_base_ = [
    "mmlab/mmpretrain/configs/_base_/models/resnet18_cifar.py",
    "mmlab/mmpretrain/configs/_base_/default_runtime.py",
]

# Model settings
model = dict(
    head=dict(
        num_classes=10,  # MNIST has 10 classes
        topk=(1,),
    ),
)

load_from = None


# dataset settings
data_preprocessor = dict()

# Data pipelines
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
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
    max_epochs=2,  # Increase the number of epochs if needed
    val_interval=1,
)
val_cfg = dict()
val_evaluator = dict(type="Accuracy", topk=(1,))

# Testing settings
test_cfg = dict()

test_dataloader = dict(
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

test_evaluator = dict(type="Accuracy", topk=(1,))

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