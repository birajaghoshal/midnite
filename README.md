<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# midnite
This is a framework to see into the dark boxes that are deep neural networks.

## Getting Started

### Installation
From package index: `pip install midnite` (not yet published!)
or from source:
```
git clone https://gitlab.com/luminovo/midnite.git
cd midnite
poetry build
pip install dist/midnite-*.whl
```

### Overview
The project is split into the following parts:
 - **uncertainty** <sub>[notebook](notebooks/uncertainty.ipynb), [doc](TODO)</sub>: confidence of predictions
 - **visualization** of network properties.
    - **base** building blocks <sub>[notebook](notebooks/building_blocks.ipynb), [doc](TODO)</sub> in a consistent and flexible interface
    - **compound** methods <sub>[notebook](notebooks/compound.ipynb), [doc](TODO)</sub> implementing 'standard' visualization methods

To follow the examples interactively, make sure you have cloned the repo and run `poetry install`.
Then start jupyter with `poetry run jupyter notebook`.

### Implemented methods
#### Uncertainty measures
 - predictive entropy (total uncertainty). <sub>[doc](TODO), [reference](TODO)</sub>
 - mutual information (model uncertainty). <sub>[doc](TODO), [reference](TODO)</sub>

#### Visualization
 - pixel activation optimization <sub>[building block](TODO), [doc](TODO), [reference](TODO)</sub>
 - (guided) backpropagation<sub> [building block](TODO), [doc](TODO), [reference](TODO)</sub>
 - (guided) gradCAM<sub> [building block](TODO), [doc](TODO), [reference](TODO)</sub>
 - occlusion<sub> [building block](TODO), [doc](TODO), [reference](TODO)</sub>

## Development
We value clean interfaces and well-tested code. If you want to contribute, please open an issue first.

### Depenency management
We use [poetry](https://github.com/sdispater/poetry) to manage dependencies.

### Pre-commit
Please make sure to have the pre-commit hooks installed.
Install [pre-commit](https://pre-commit.com/) and then run `pre-commit install` to register the hooks with git.

### Makefile
We use [make](https://www.gnu.org/software/make/) to streamline our development workflow.
Run `make help` to see all available commands.

<!-- START makefile-doc -->
```
$ make help 
help                 Show this help message
build                Build the docker container
check                Run all static checks (like pre-commit hooks)
docs                 Build all docs
test                 Run all tests 
```
<!-- END makefile-doc -->

## Acknowledgement
Contributors:
- Fabian Huch
- Christina Aigner

Papers:
- Alex Kendall, Yarin Gal, ... +Links
