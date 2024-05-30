<div align="center">
<h1>Any CV</h1>
</div>

----

## Install:

```bash
make install
```

## Train:
```bash
CONFIG=configs/effnet.py \
WORKDIR=test_exp \
PROJECT=mmpretrain \
GPUS=0 \
make train
```

## Convert:

```bash
DEPLOY_CFG_PATH=mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py \
MODEL_CFG_PATH=mount/test_exp/effnet.py \
MODEL_CHECKPOINT_PATH=mount/test_exp/epoch_38.pth \
INPUT_IMG=mount/data/train/class_norma/508_lat.png \
TEST_IMG=mount/data/train/class_pneumonia/369_pa.png \
WORK_DIR=mount/test_exp \
DEVICE=cpu \
PROJECT=mmpretrain \
make convert
```

## MNIST Example:

```bash
# download mnist data
python anycv/load_mnist.py

# train model
CONFIG=configs/effnet_b0_mnist.py \
WORKDIR=test_exp \
PROJECT=mmpretrain \
GPUS=0 \
make train

# convert model
DEPLOY_CFG_PATH=mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py \
MODEL_CFG_PATH=mount/test_exp/effnet_b0_mnist.py \
MODEL_CHECKPOINT_PATH=mount/test_exp/epoch_2.pth \
INPUT_IMG=mount/data/train/class_0/1.png \
TEST_IMG=mount/data/train/class_1/3.png \
WORK_DIR=mount/test_exp \
DEVICE=cpu \
PROJECT=mmpretrain \
make convert
```