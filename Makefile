APP_NAME=sleep3r/anycv
CONTAINER_NAME=anycv
PROJECT_NAME=/anycv

.PHONY: format test train install help

install:  ## Install the requirements
	pip install -e .
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

train:  ## Train the model
	PYTHONPATH=$PWD:$PYTHONPATH mim train $(PROJECT) $(CONFIG) --gpus $(GPUS) --work-dir $(WORKDIR)

test:  ## Run the tests
	pytest -s tests

format:  ## Format configs
	@isort ./configs
	@black ./configs

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
