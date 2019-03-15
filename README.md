<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# interpretability-framework
This is a framework for measuring and visualizing uncertainty in CNNs.

Monte Carlo dropout is used to collect an ensemble of predictions, which are used as a basis for measuring data and model uncertainty.

## Motivation

There are two major types of uncertainty one can model. Data (aleatoric) uncertainty captures noise inherent in the observations. On the other hand, model (epistemic) uncertainty accounts for uncertainty in the model – uncertainty which can be explained away given enough data.

We study the benefits of measuring and visualizing uncertainty for image classification and image segmentation and how we can leverage this insight to improve the model.
For this, we want to gain insight on the expected uncertainty (confidence) of the predictions and identify irreducible uncertainties of the model.

For measuring uncertainty we used two main applications:

 * Image Classification
 
    applying <name of our tool> measures the data and model uncertainty of an image.

 * Image Segmentation
 
    applying <name of our tool> measures the data and model uncertainty for each pixel.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The model that needs to be tested, needs to be trained with Dropout. 

What things you need to install the software and how to install them

```
poetry install
```

### API References

todo

## Deployment and usage

In the notebook folder you find tutorial implementations, which show how to use the framework. 

Run the notebooks via poetry.

```
cd notebooks
poetry run jupyter-notebook
```

### Image Classification

Open Uncertainty Tutorial 1 - Image Classification Example using AlexNet.ipynb for further details.

### Image Segmentation

Open Uncertainty Tutorial 2 - Image Segmentation Example using FCN32.ipynb for furhter details. 

## Running the tests

In the terminal, go to your project directory, then execute the following:

```
cd tests
pytest
```

## Development

### Pre-commit
Please make sure to have the pre-commit hooks installed.
Install [pre-commit](https://pre-commit.com/) and then run `pre-commit install` to register the hooks with git.

### Poetry
Use [poetry](https://poetry.eustace.io/) to manage your dependencies.
Please make sure it is installed for your current python version.
Then start by adding your dependencies:
```console
poetry add torch
```

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

## Credits

Alex Kendall, Yarin Gal, ... +Links

## Authors

* Fabian Huch
* Christina Aigner
