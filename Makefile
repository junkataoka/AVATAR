
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = s3://
PROFILE = default
PROJECT_NAME = da_bfd
ENVNAME = mlenv
PYTHON_INTERPRETER = python3
SHELL := /usr/bin/bash
VENV           = $(HOME)/$(ENVNAME)
VENV_PYTHON    = $(VENV_PYTHON)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
# If virtualenv exists, use it. If not, find python using PATH
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON)


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
activate_env:
	$(HOME)/$(ENVNAME)/bin/activate
	pip3 freeze > requirements.txt  # Python3


build_environment:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf ./data/processed/*

## Lint using flake8
lint:
	flake8 src

.PHONY: loadmodule
loadmodule: 
	module load cuda11.1/toolkit/11.1.1; \

.PHONY: test
test: loadmodule
	export PYTHONPATH=$(PROJECT_DIR)/src; \
	pytest -s tests -rv  --durations 5





##$(PYTHON) -m pytest -s tests/ -rv  --durations 5

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
## Pretrain
.PHONY: pretrain_office31
pretrain_office31: loadmodule
	sbatch pretrain_avatar_office31.sh amazon webcam; \
	sbatch pretrain_avatar_office31.sh amazon dslr; \
	sbatch pretrain_avatar_office31.sh dslr amazon; \
	sbatch pretrain_avatar_office31.sh dslr webcam; \
	sbatch pretrain_avatar_office31.sh webcam amazon; \
	sbatch pretrain_avatar_office31.sh webcam dslr; \

.PHONY: train_office31
train_office31: loadmodule
	sbatch train_avatar_office31.sh amazon webcam; \
	#sbatch train_avatar_office31.sh amazon dslr; \
	sbatch train_avatar_office31.sh dslr amazon; \
	sbatch train_avatar_office31.sh dslr webcam; \
	sbatch train_avatar_office31.sh webcam amazon; \
	sbatch train_avatar_office31.sh webcam dslr; \

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
