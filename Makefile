# Courtesy of luminovo.ai
# Automates your development workflows

# ----------Environment Variables---------

# set environment variables to be used in the commands here

# ----------------Commands----------------

# change the 20 value in printf to adjust width
# Use ' ## some comment' behind a command and it will be added to the help message automatically
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the docker container
	docker build --file docker/Dockerfile --tag luminovo/vinsight:latest .

check: ## Run all static checks (like pre-commit hooks)
	pre-commit run --all-files

docs: ## Build all docs
	docker run --rm -it -v `pwd`/docs:/docs squidfunk/mkdocs-material build

test: ## Run all tests
	docker run -v `pwd`:/home/vinsight luminovo/vinsight:latest pytest tests/


# --------------Configuration-------------

.ONESHELL: ; # recipes execute in same shell
.NOTPARALLEL: ; # wait for this target to finish
.EXPORT_ALL_VARIABLES: ; # send all vars to shell

.PHONY: docs all # All targets are accessible for user
.DEFAULT: help # Running Make will run the help target

MAKEFLAGS += --no-print-directory # dont add message about entering and leaving the working directory