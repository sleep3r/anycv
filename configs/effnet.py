_base_ = [
    'mmlab/mmpretrain/configs/_base_/models/efficientnet_b0.py',
    'mmlab/mmpretrain/configs/_base_/default_runtime.py',
]

# Model settings
model = dict(
    head=dict(
        num_classes=2,
        topk=(1,),
    ),
)

load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth"

# dataset settings
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),  # Resize the image to 224x224
    dict(type='Normalize', **data_preprocessor),  # Normalizing the images
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),  # Resize the image to 224x224
    dict(type='Normalize', **data_preprocessor),  # Normalizing the images
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,  # Adjust batch size as per your system capabilities
    shuffle=True,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/train',
        with_label=True,
        pipeline=train_pipeline,
    ),
    num_workers=8,  # Adjust num_workers as per your system capabilities
)

val_dataloader = dict(
    batch_size=8,  # Adjust batch size as per your system capabilities
    shuffle=False,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/val',
        with_label=True,
        pipeline=test_pipeline,
    ),
    num_workers=8,  # Adjust num_workers as per your system capabilities
)


# Training settings
train_cfg = dict(
    by_epoch=True,
    max_epochs=50,  # Increase the number of epochs for better training
    val_interval=2,
)
val_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1,))

# Testing settings
test_cfg = None

optim_wrapper = dict(
    # Use SGD optimizer to optimize parameters.
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001)
)