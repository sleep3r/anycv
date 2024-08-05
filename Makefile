.PHONY: format train install help convert test infer

install:  ## Install the requirements
	git submodule update --init --recursive
	poetry install
	poetry run mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

train:  ## Train the model
	PYTHONPATH=$$PWD:$$PYTHONPATH poetry run mim train $(PROJECT) $(CONFIG) --gpus $(GPUS) --work-dir $(WORKDIR)

test: ## Evaluate the model
	PYTHONPATH=$$PWD:$$PYTHONPATH poetry run mim test $(PROJECT) $(CONFIG) --checkpoint $(CHECKPOINT) --gpus $(GPUS) --work-dir $(WORKDIR)

convert: ## Convert the model
	docker run --rm -it \
		-v $$(pwd):/root/workspace/mount \
		openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy \
		bash -c "pip install onnx onnxruntime && \
        pip install -U openmim && mim install $(PROJECT) && \
        cp $(DEPLOY_CFG_PATH) $(WORK_DIR)/ && \
		python3 mmdeploy/tools/deploy.py \
		$(DEPLOY_CFG_PATH) \
		$(MODEL_CFG_PATH) \
		$(MODEL_CHECKPOINT_PATH) \
		$(INPUT_IMG) \
		--work-dir $(WORK_DIR) \
		--device $(DEVICE) \
		--log-level INFO"

infer: ## Run inference using the ONNX model
	poetry run python anycv/inference.py \
		--project=$${PROJECT} \
		--config_path=$${CONFIG_PATH} \
		--model_path=$${MODEL_PATH} \
		--image_path=$${IMAGE_PATH}

format:  ## Format configs
	@isort ./configs --skip configs/mmlab
	@black ./configs --exclude configs/mmlab

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)