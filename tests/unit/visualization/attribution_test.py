"""Unit tests for the attribution methods."""
import numpy as np
import pytest
import torch
from assertpy import assert_that
from torch import Size
from torch.nn import Dropout2d
from torch.nn import Module

from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import LayerSplit
from midnite.visualization.base import methods
from midnite.visualization.base import NeuronSelector
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import SpatialSplit


@pytest.fixture
def backprop_mock_setup(mocker):
    """Setup img, layers, and selectors."""
    img = torch.zeros((1, 3, 4, 4))

    # We want to have the exact same img after these operations because that makes
    # testing for equality in assert_called_with easy
    img.clone = mocker.Mock(return_value=img)
    img.detach = mocker.Mock(return_value=img)
    img.to = mocker.Mock(return_value=img)
    img.requires_grad_ = mocker.Mock(return_value=img)

    out = torch.ones((1, 3, 4, 4))
    out1 = torch.ones((1, 3, 4, 4))
    out2 = torch.ones((1, 3, 4, 4))
    out2.detach = mocker.Mock(return_value=out2)
    out3 = torch.ones((1, 3, 4, 4))
    out3.squeeze = mocker.Mock(return_value=out3)
    out4 = torch.ones((1, 4, 4))

    layers = [
        mocker.MagicMock(spec=Module, return_value=img),
        mocker.MagicMock(spec=Module, return_value=out),
    ]

    for layer in layers:
        layer.to = mocker.Mock(return_value=layers)

    mocker.patch(
        "midnite.visualization.base.methods.get_single_mean", return_value=out1
    )
    mocker.patch("torch.autograd.grad", return_value=[out2])
    mocker.patch("torch.nn.functional.relu", return_value=out3)

    top_layer_sel = mocker.Mock(spec=NeuronSelector)

    bottom_split = mocker.Mock(spec=LayerSplit)
    bottom_split.get_mean = mocker.Mock(return_value=out4)

    return img, layers, top_layer_sel, (out, out1, out2, out3, out4), bottom_split


@pytest.fixture
def gradcam_mock_setup(mocker):
    """Patch splits and setup img and base layers"""
    img = torch.zeros((1, 3, 4, 4))
    base_layers = [Dropout2d(), Dropout2d()]
    mocker.spy(base_layers[0], "forward")
    mocker.spy(base_layers[1], "forward")

    return img, base_layers


def test_guided_backpropagation(backprop_mock_setup):
    """Check that layers are in correct mode and the gradient is calculated properly."""
    img, layers, top_layer_sel, out, bottom_split = backprop_mock_setup
    sut = GuidedBackpropagation(layers, top_layer_sel, bottom_split)

    result = sut.visualize(img)

    # Check preprocessing
    img.clone.assert_called_once()
    img.detach.assert_called_once()
    img.to.assert_called_once()
    img.requires_grad_.assert_called_once()

    layers[0].train.assert_called_once_with(False)
    layers[1].train.assert_called_once_with(False)

    # Check forward pass
    layers[0].assert_called()
    layers[1].assert_called()

    # Check gradient calculation
    # 1. Output mean
    methods.get_single_mean.assert_called_with(out[0], top_layer_sel)
    # 2. Take gradient
    torch.autograd.grad.assert_called_once_with(out[1], img)
    out[2].detach.assert_called_once()
    # 3. Relu
    torch.nn.functional.relu.assert_called_once_with(out[2])
    # 4. Split mean
    bottom_split.get_mean.assert_called_once_with(out[3])
    # 5. Check result
    assert_that(result).is_same_as(out[4])


def test_gradcam_basics(mocker, gradcam_mock_setup):
    img, base_layers = gradcam_mock_setup

    mocker.patch(
        "midnite.visualization.base.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(3),
    )
    GradAM(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        SpatialSplit(),
    ).visualize(img)

    base_layers[0].forward.assert_called()
    base_layers[1].forward.assert_called()
    assert_that(base_layers[0].training).is_false()
    assert_that(base_layers[1].training).is_false()


def test_saliency_visualize_spatial(mocker, gradcam_mock_setup):
    """test the correct computation and dimensions of the saliency for a spatial split"""
    img, base_layers = gradcam_mock_setup

    mocker.patch(
        "midnite.visualization.base.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(3),
    )
    sal_map = GradAM(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        SpatialSplit(),
    ).visualize(img)

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([4, 4]))


def test_saliency_visualize_channel(mocker, gradcam_mock_setup):
    """test the correct computation and dimensions of the saliency for a channel split"""
    img, base_layers = gradcam_mock_setup

    mocker.patch(
        "midnite.visualization.base.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones((4, 4)),
    )
    sal_map = GradAM(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        ChannelSplit(),
    ).visualize(img)

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([3]))


def test_saliency_visualize_neuron(mocker, gradcam_mock_setup):
    """test the correct computation and dimensions of the saliency for a neuron split"""
    img, base_layers = gradcam_mock_setup

    mocker.patch(
        "midnite.visualization.base.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(()),
    )
    sal_map = GradAM(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        NeuronSplit(),
    ).visualize(img)

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([3, 4, 4]))
