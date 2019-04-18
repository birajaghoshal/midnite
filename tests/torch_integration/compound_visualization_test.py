"""Integration test of compound visualization methods against torch."""
from unittest.mock import PropertyMock

import pytest
import torch
from assertpy import assert_that
from torch.nn import Sequential

from midnite.common import Flatten
from midnite.uncertainty import StochasticDropouts
from midnite.visualization import gradcam
from midnite.visualization.base import Occlusion
from midnite.visualization.base import PixelActivation
from midnite.visualization.compound_methods import class_visualization
from midnite.visualization.compound_methods import graduam
from midnite.visualization.compound_methods import guided_backpropagation
from midnite.visualization.compound_methods import guided_gradcam
from midnite.visualization.compound_methods import occlusion


@pytest.fixture(scope="module")
def single_img(img):
    """Fixture for image without minibatch dimensions."""
    return img.squeeze(0)


def test_guided_backpropagation(net, single_img):
    """Run guided backpropagation and check output size."""
    result = guided_backpropagation([net], single_img)

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradcam(net, single_img):
    """Run gradcam and check size, all >=0."""
    result = gradcam(
        Sequential(net.features, net.avgpool),
        Sequential(Flatten(), net.classifier),
        single_img,
    )

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_graduam(net, single_img):
    """Run graduam and check sizes + sufficiently large gradients"""
    result = graduam(
        StochasticDropouts(Sequential(net.features, net.avgpool)),
        StochasticDropouts(Sequential(Flatten(), net.classifier)),
        single_img,
        5,
    )

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0.0001)


def test_guided_gradcam(net, single_img):
    """Run guided gradcam and check size, all >=0."""
    result = guided_gradcam(
        list(net.features.children()) + [net.avgpool],
        [Flatten()] + list(net.classifier.children()),
        single_img,
    )

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_occlusion(mocker, net, single_img):
    """Run occlusion and check that result is a valid image."""
    # Limit the amount of occlusion passes
    mocker.patch.object(
        Occlusion,
        "stride",
        create=True,
        new_callable=PropertyMock,
        return_value=[1, 100, 100],
    )
    mocker.patch.object(
        Occlusion,
        "chunk_size",
        create=True,
        new_callable=PropertyMock,
        return_value=[1, 2, 2],
    )

    result = occlusion(net, single_img)

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result > 0)


def test_class_visualization(mocker, net):
    """Run class visualization and check that result is an image."""
    # Limit number of iterations
    mocker.patch.object(
        PixelActivation, "opt_n", create=True, new_callable=PropertyMock, return_value=2
    )
    mocker.patch.object(
        PixelActivation,
        "iter_n",
        create=True,
        new_callable=PropertyMock,
        return_value=2,
    )
    result = class_visualization(net, 283)
    assert_that(result.size()).is_length(3)
    assert_that(result.size(0)).is_equal_to(3)
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)
