################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = Model Metrics Library
PYTHON_INTERPRETER = python3


ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

### general usage notes
### 2>&1 | tee ==>pipe operation to save model output from terminal to .txt file

################################################################################
# COMMANDS                                                                     #
################################################################################

################################################################################
############## Setting up a Virtual Environment and Dependencies ###############
################################################################################
# virtual environment set-up (local)
venv:
	$(PYTHON_INTERPRETER) -m venv metrics_venv
	source metrics_venv/bin/activate

## Install Python Dependencies
requirements_local:	
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel

venv_dep_setup_local: venv requirements_local	# for local set-up
venv_dep_setup_gpu: venv requirements_gpu     # for server/gpu set-up

################################################################################
###########################  Dataset Script Generation #########################
################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf public_data/
	rm -rf data/
	rm -rf data_output/
	rm -rf images/

.PHONY: create_folders

create_folders:
	mkdir -p model_files/results
	mkdir -p model_files/single_model_lr_results
	mkdir -p model_files/images/png_images
	mkdir -p model_files/images/svg_images
	mkdir -p model_files/single_model_lr_results/images/png_images
	mkdir -p model_files/single_model_lr_results/images/svg_images
################################################################################
##########################  Modeling Script Generation #########################
################################################################################

.PHONY: single_model_logistic_reg
single_model_logistic_reg: 
	$(PYTHON_INTERPRETER) \
	py_scripts/single_model_classification.py \
	model_files/single_model_classification_results/ \

.PHONY: single_model_lasso_reg
single_model_lasso_reg: 
	$(PYTHON_INTERPRETER) \
	py_scripts/single_model_regression.py \
	model_files/single_model_regression_results/ \

################################################################################

## Make Logistic Regression
.PHONY: logistic_regression
logistic_regression: 
	$(PYTHON_INTERPRETER) \
	py_scripts/train_adult_income.py \
	--model-type lr \
	2>&1 | tee model_files/results/logistic_regression.txt

## Make Decision Tree
.PHONY: decision_tree
decision_tree: 
	$(PYTHON_INTERPRETER) \
	py_scripts/train_adult_income.py \
	--model-type dt \
	2>&1 | tee model_files/results/decision_tree.txt

## Make Random Forest Classifier
.PHONY: random_forest
random_forest: 
	$(PYTHON_INTERPRETER) \
	py_scripts/train_adult_income.py \
	--model-type rf \
	2>&1 | tee model_files/results/random_forest.txt

all_models: single_model_lasso_reg single_model_logistic_reg \
            logistic_regression decision_tree random_forest  

################################################################################
############################## Model Evaluation ################################
################################################################################

.PHONY: model_evaluation
model_evaluation: 
	$(PYTHON_INTERPRETER) \
	py_scripts/single_model_evaluation.py \
	model_files/single_model_lr_results/ \

################################################################################
############################## Model Explanation ###############################
################################################################################

.PHONY: model_explanation
model_explanation: 
	$(PYTHON_INTERPRETER) \
	py_scripts/model_explanation.py \
	model_files/single_model_lr_results/ \
