from pathlib import Path

import pytest
from assertpy import assert_that
from torch.nn import Sequential
from torch.nn import Softmax
from torchvision import models

import data_utils
from data_utils import DataConfig
from midnite.visualization.base import Backpropagation
from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector


@pytest.fixture(scope="module")
def network():
    return models.alexnet(pretrained=True)


@pytest.fixture(scope="module")
def img():
    file = Path(__file__).parents[2].joinpath("data/imagenet_example_283.jpg")
    return data_utils.get_example_from_path(file, DataConfig.ALEX_NET)


def test_spatial_backpropagation(network, img):
    """Test guided backpropagation with a spatial split."""
    result = (
        Backpropagation(
            Sequential(network, Softmax(dim=1)),
            SplitSelector(NeuronSplit(), [283]),
            SpatialSplit(),
        )
        .visualize(img)
        .cpu()
    )

    assert_that(img.size()).is_length(4)
    assert_that(result.size()).is_length(2)
    assert_that(result.size()).is_equal_to(img.size()[2:])


def test_partial_backpropagation(network, img):
    """Test guided backpropagation over the partial network."""
    result = (
        Backpropagation(
            network.features, SplitSelector(ChannelSplit(), [60]), SpatialSplit()
        )
        .visualize(img)
        .cpu()
    )

    assert_that(result.size()).is_length(2)
    assert_that(result.size()).is_equal_to(img.size()[2:])
