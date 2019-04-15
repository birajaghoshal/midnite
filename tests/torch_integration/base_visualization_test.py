"""Integration test of base visualization methods against torch."""
from typing import List

import pytest
import torch
from assertpy import assert_that
from torch.nn import Module
from torch.nn import Sequential

from midnite.common import Flatten
from midnite.visualization.base import Backpropagation
from midnite.visualization.base import BilateralTransform
from midnite.visualization.base import BlurTransform
from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import PixelActivation
from midnite.visualization.base import RandomTransform
from midnite.visualization.base import ResizeTransform
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector
from midnite.visualization.base import TVRegularization
from midnite.visualization.base import WeightDecay


@pytest.fixture
def net_layers(net) -> List[Module]:
    """Layers of alexnet."""
    return (
        list(net.features.children())
        + [net.avgpool, Flatten()]
        + list(net.classifier.children())
    )


def test_channel_backpropagation(net_layers, img, correct_class_selector):
    """Test backpropagation with channel split."""
    result = (
        Backpropagation(Sequential(*net_layers), correct_class_selector, ChannelSplit())
        .visualize(img)
        .cpu()
    )

    assert_that(result.size()).is_length(1)
    assert_that(result.size(0)).is_equal_to(3)


def test_spatial_guided_backpropagation(net_layers, img, correct_class_selector):
    """Test guided backpropagation with a spatial split."""
    result = (
        GuidedBackpropagation(net_layers, correct_class_selector, SpatialSplit())
        .visualize(img)
        .cpu()
    )

    assert_that(result.size()).is_length(2)
    # Should have input dimensions
    assert_that(result.size()).is_equal_to(img.size()[2:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_partial_guided_backpropagation(net, img):
    """Test guided backpropagation over the partial network."""
    result = (
        GuidedBackpropagation(
            net.features, SplitSelector(ChannelSplit(), [60]), SpatialSplit()
        )
        .visualize(img)
        .cpu()
    )

    assert_that(result.size()).is_length(2)
    # Should have input dimensions
    assert_that(result.size()).is_equal_to(img.size()[2:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def _get_gradam(network, split, correct_class_selector):
    return GradAM(
        Sequential(Flatten(), network.classifier),
        correct_class_selector,
        Sequential(network.features, network.avgpool),
        split,
    )


def test_gradam_spatial(net, img, correct_class_selector):
    """Test GradAM over spatial split."""
    result = (
        _get_gradam(net, SpatialSplit(), correct_class_selector).visualize(img).cpu()
    )

    size = result.size()
    assert_that(size).is_length(2)
    # 6 x 6 spatial positions
    assert_that(size).is_equal_to((6, 6))
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradam_channel(net, random_img, correct_class_selector):
    """Test GradAM over channel split, with random image.

    Uses random image as in the example, there is no channel which is important over
    enough spaital positions to be >0.

    """
    result = (
        _get_gradam(net, ChannelSplit(), correct_class_selector)
        .visualize(random_img)
        .cpu()
    )

    # 256 channels
    assert_that(result.size()).is_equal_to((256,))
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradam_neuron(net, random_img, correct_class_selector):
    """Test GradAM over neuron split, with random image."""
    result = (
        _get_gradam(net, NeuronSplit(), correct_class_selector)
        .visualize(random_img)
        .cpu()
    )

    # 256 x 6 x 6 neurons
    assert_that(result.size()).is_equal_to((256, 6, 6))
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_pixel_activation_simple(net, correct_class_selector):
    """Test pixel activation without any transformations/regularizations."""
    opt = PixelActivation(net, correct_class_selector, iter_n=2, opt_n=2)
    out = opt.visualize()
    assert_that(out.size()).is_length(3)
    assert_that(out.size(0)).is_equal_to(3)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_pixel_activation_complex(net, correct_class_selector):
    """Test pixel activation with all transformatinons/regularizers activated."""
    opt = PixelActivation(
        net,
        correct_class_selector,
        iter_n=2,
        opt_n=2,
        init_size=50,
        clamp=False,
        transform=RandomTransform()
        + BlurTransform()
        + BilateralTransform()
        + ResizeTransform(),
        regularization=[TVRegularization(), WeightDecay(1e-12)],
    )
    out = opt.visualize(torch.ones((3, 50, 50)))

    assert_that(out.size()).is_length(3)
    assert_that(out.size(0)).is_equal_to(3)
    assert torch.any(out > 1)
