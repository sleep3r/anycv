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
MODEL_CFG_PATH=data/test_exp/effnet.py \
MODEL_CHECKPOINT_PATH=data/test_exp/epoch_38.pth \
INPUT_IMG=data/data/train/class_norma/508_lat.png \
TEST_IMG=data/data/train/class_pneumonia/369_pa.png \
WORK_DIR=data/test_exp \
DEVICE=cpu \
PROJECT=mmpretrain \
make convert
```