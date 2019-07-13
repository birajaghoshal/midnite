# midnite
<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</p>

![](./docs/source/assets/images/logo.svg)

This is a framework to gain insight into [pytorch](https://pytorch.org/) deep neural networks for visual recognition tasks.

The `README.md` contains only a **very brief** description - for details, visit the

--> **[full documentation and apidoc](https://luminovo.gitlab.io/midnite/)** <--

## Overview
The project is split into the following parts, with the linked jupyter notebooks explaining them:
 - [uncertainty](notebooks/Monte-Carlo Uncertainty.ipynb): confidence of predictions
 - visualization of network properties.
    - base building blocks in a consistent and flexible interface
    - [compound](notebooks/Compound Visualization.ipynb) methods implementing 'standard' visualization methods

To follow the examples interactively, clone the repository and run `poetry install`.
Then start jupyter with `poetry run jupyter notebook`.

### Implemented methods
- [uncertainties](https://arxiv.org/abs/1506.02142)
    - predictive entropy (total uncertainty).
    - mutual information (model uncertainty).
- [building blocks](https://distill.pub/2018/building-blocks/)
- visualization
    - [pixel activation optimization](https://distill.pub/2017/feature-visualization/)
    - [(guided) backpropagation](https://arxiv.org/abs/1312.6034)
    - [(guided) gradCAM](https://arxiv.org/abs/1611.07450)
    - [occlusion](https://arxiv.org/abs/1311.2901)

## Installation
From package index: `pip install "midnite"` or from source:
```
git clone https://gitlab.com/luminovo/midnite.git
cd midnite
poetry build
pip install dist/midnite-*.whl
```

## Development
We value clean interfaces and well-tested code. If you want to contribute, usually it's best to open an issue first.

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
check                Run all static checks (like pre-commit hooks)
docs                 Build all docs 
```
<!-- END makefile-doc -->

## Acknowledgement
Scientific sources: see [reference doc page](https://luminovo.gitlab.io/midnite/references.html).

Contributors:
- luminovo.ai
- Fabian Huch
- Christina Aigner
