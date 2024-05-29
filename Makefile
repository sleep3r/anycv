.PHONY: format test train install help convert

install:  ## Install the requirements
	pip install -e .
	mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

train:  ## Train the model
	PYTHONPATH=$$PWD:$$PYTHONPATH mim train $(PROJECT) $(CONFIG) --gpus $(GPUS) --work-dir $(WORKDIR)

# Define variables at the beginning
DEPLOY_CFG_PATH=mmdeploy/configs/mmpretrain/classification_onnxruntime_static.py
MODEL_CFG_PATH=data/test_exp/effnet.py
MODEL_CHECKPOINT_PATH=data/test_exp/epoch_38.pth
INPUT_IMG=data/data/train/class_norma/508_lat.png
TEST_IMG=data/data/train/class_pneumonia/369_pa.png
WORK_DIR=data/test_exp
DEVICE=cpu  # or 'cuda' if using GPU
PROJECT=mmpretrain
convert: ## Convert the model
	docker run --rm -it \
		-v $$(pwd):/root/workspace/data \
		-e DEPLOY_CFG_PATH=$(DEPLOY_CFG_PATH) \
		-e MODEL_CFG_PATH=$(MODEL_CFG_PATH) \
		-e MODEL_CHECKPOINT_PATH=$(MODEL_CHECKPOINT_PATH) \
		-e INPUT_IMG=$(INPUT_IMG) \
		-e TEST_IMG=$(TEST_IMG) \
		-e WORK_DIR=$(WORK_DIR) \
		-e DEVICE=$(DEVICE) \
		openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy \
		bash -c "mim install $(PROJECT) && python3 mmdeploy/tools/deploy.py \
		$(DEPLOY_CFG_PATH) \
		$(MODEL_CFG_PATH) \
		$(MODEL_CHECKPOINT_PATH) \
		$(INPUT_IMG) \
		--test-img $(TEST_IMG) \
		--work-dir $(WORK_DIR) \
		--device $(DEVICE) \
		--log-level INFO \
		--show \
		--dump-info"

test:  ## Run the tests
	pytest -s tests

format:  ## Format configs
	@isort ./configs --skip configs/mmlab
	@black ./configs --exclude configs/mmlab

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)