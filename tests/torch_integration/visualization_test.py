"""Integration test of visualization methods against torch."""
from pathlib import Path

import numpy as np
import pytest
from assertpy import assert_that
from torch.nn import Softmax
from torchvision import models

import data_utils
from data_utils import DataConfig
from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector
from model_utils import Flatten


@pytest.fixture(scope="module")
def network():
    """Fixture for pre-trained AlexNet"""
    return models.alexnet(pretrained=True)


@pytest.fixture(scope="module")
def img():
    """Fixture for exmample image."""
    file = Path(__file__).parents[2].joinpath("data/imagenet_example_283.jpg")
    return data_utils.get_example_from_path(file, DataConfig.ALEX_NET)


def test_spatial_guided_backpropagation(network, img):
    """Test guided backpropagation with a spatial split."""
    result = (
        GuidedBackpropagation(
            [network, Softmax(dim=1)],
            SplitSelector(NeuronSplit(), [283]),
            SpatialSplit(),
        )
        .visualize(img)
        .cpu()
    )

    assert_that(img.size()).is_length(4)
    assert_that(result.size()).is_length(2)
    # Should have input dimensions
    assert_that(result.size()).is_equal_to(img.size()[2:])
    assert np.all(result.numpy() >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_partial_guided_backpropagation(network, img):
    """Test guided backpropagation over the partial network."""
    result = (
        GuidedBackpropagation(
            network.features, SplitSelector(ChannelSplit(), [60]), SpatialSplit()
        )
        .visualize(img)
        .cpu()
    )

    assert_that(result.size()).is_length(2)
    # Should have input dimensions
    assert_that(result.size()).is_equal_to(img.size()[2:])
    assert np.all(result.numpy() >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def get_gradam(network, split):
    return GradAM(
        [Flatten(), network.classifier],
        SplitSelector(NeuronSplit(), [283]),
        [network.features, network.avgpool],
        split,
    )


def test_gradam_spatial(network, img):
    """Test GradAM over spatial split."""
    result = get_gradam(network, SpatialSplit()).visualize(img).cpu()

    size = result.size()
    assert_that(size).is_length(2)
    # 6 x 6 spatial positions
    assert_that(size).is_equal_to((6, 6))
    assert np.all(result.numpy() >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradam_channel(network, img):
    """Test GradAM over channel split."""
    result = get_gradam(network, ChannelSplit()).visualize(img).cpu()

    # 256 channels
    assert_that(result.size()).is_equal_to((256,))
    assert np.all(result.numpy() >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradam_neuron(network, img):
    """Test GradAM over neuron split."""
    result = get_gradam(network, NeuronSplit()).visualize(img).cpu()

    # 256 x 6 x 6 neurons
    assert_that(result.size()).is_equal_to((256, 6, 6))
    assert np.all(result.numpy() >= 0)
    assert_that(result.sum().item()).is_greater_than(0)
