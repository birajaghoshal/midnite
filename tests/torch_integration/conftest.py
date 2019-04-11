from pathlib import Path

import pytest
from torchvision import models

import data_utils
from data_utils import DataConfig


@pytest.fixture(scope="module")
def net():
    """Fixture for pre-trained AlexNet"""
    return models.alexnet(pretrained=True)


@pytest.fixture(scope="module")
def img():
    """Fixture for exmample image."""
    file = Path(__file__).parents[2].joinpath("data/imagenet_example_283.jpg")
    return data_utils.get_example_from_path(file, DataConfig.ALEX_NET)


@pytest.fixture(scope="module")
def random_img():
    """Fixture for random image."""
    return data_utils.get_random_example(DataConfig.ALEX_NET)
