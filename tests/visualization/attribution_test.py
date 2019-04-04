"""Unit tests for the attribution methods."""
import numpy as np
import pytest
import torch
from assertpy import assert_that
from torch import Size
from torch.nn import Dropout2d
from torch.nn import Module

from midnite.visualization import ChannelSplit
from midnite.visualization import GradCAM
from midnite.visualization import GuidedBackpropagation
from midnite.visualization import Identity
from midnite.visualization import LayerSplit
from midnite.visualization import NeuronSelector
from midnite.visualization import NeuronSplit
from midnite.visualization import SpatialSplit


@pytest.fixture
def backprop_mock_setup(mocker):
    """Setup img, layers, and selectors."""
    img = torch.zeros((1, 3, 4, 4))
    layers = [Dropout2d(), Dropout2d()]

    mocker.spy(layers[0], "forward")
    mocker.spy(layers[1], "forward")
    mocker.patch("torch.autograd.grad", return_value=torch.randn(1, 3, 4, 4))

    top_layer_sel = mocker.Mock(spec=NeuronSelector)
    top_layer_sel.get_mask = mocker.Mock(return_value=torch.ones((3, 4, 4)))

    return img, layers, top_layer_sel


@pytest.fixture
def gradcam_mock_setup(mocker):
    """Patch splits and setup img and base layers"""
    img = torch.zeros((1, 3, 4, 4))
    base_layers = [Dropout2d(), Dropout2d()]
    mocker.spy(base_layers[0], "forward")
    mocker.spy(base_layers[1], "forward")

    return img, base_layers


def test_guided_backprop_basics(backprop_mock_setup):
    """Check that layers are in correct mode and forward passes are made."""
    img, layers, top_layer_sel = backprop_mock_setup
    backprop_spatial = GuidedBackpropagation(layers, top_layer_sel, SpatialSplit())

    assert_that(layers[0].training).is_true()
    assert_that(layers[1].training).is_true()

    backprop_spatial.visualize(img)

    assert_that(layers[0].training).is_false()
    assert_that(layers[1].training).is_false()
    # Check forward pass
    layers[0].forward.assert_called()
    layers[1].forward.assert_called()


def test_guided_backprop_splits(mocker, backprop_mock_setup):
    """tests the correct computation of guided gradients for the different splits"""
    img, layers, top_layer_sel = backprop_mock_setup

    grad = GuidedBackpropagation(layers, top_layer_sel, SpatialSplit()).visualize(img)
    assert_that(np.all(grad.numpy() >= 0)).is_true()
    assert_that(grad.size()).is_equal_to(Size([4, 4]))

    grad = GuidedBackpropagation(layers, top_layer_sel, ChannelSplit()).visualize(img)
    assert_that(np.all(grad.numpy() >= 0)).is_true()
    assert_that(grad.size()).is_equal_to(Size([3]))

    grad = GuidedBackpropagation(layers, top_layer_sel, NeuronSplit()).visualize(img)
    assert_that(np.all(grad.numpy() >= 0)).is_true()
    assert_that(grad.size()).is_equal_to(Size([3, 4, 4]))

    grad = GuidedBackpropagation(layers, top_layer_sel, Identity()).visualize(img)
    assert_that(grad.size()).is_equal_to(Size([]))


def test_guided_backprop_scores_1d(mocker):
    """Check that the output score is correctly computed for 1d top layer dimension."""
    # 1 dimensional selection, example neuron split
    top_layer_selector_1d = mocker.Mock(spec=NeuronSelector)
    top_layer_selector_1d.get_mask = mocker.Mock(return_value=torch.ones((50,)))

    out = torch.ones((50,))

    backprop = GuidedBackpropagation(
        [mocker.Mock(spec=Module)], top_layer_selector_1d, mocker.Mock(spec=LayerSplit)
    )

    score = backprop.select_top_layer_score(out)

    assert_that(score.size()).is_equal_to(Size([]))
    assert_that(score).is_equal_to(1)


def test_guided_backprop_scores_3d(mocker):
    """tests if the top-layer output score is correctly computed
    for 3d top layer dimension."""

    # 3 dimensional selection, example neuron split
    top_layer_selector_3d = mocker.Mock(spec=NeuronSelector)
    top_layer_selector_3d.get_mask = mocker.Mock(return_value=torch.ones((4, 3, 3)))

    out = torch.ones((1, 4, 3, 3))

    backprop = GuidedBackpropagation(
        [mocker.Mock(spec=Module)], top_layer_selector_3d, mocker.Mock(spec=LayerSplit)
    )

    score = backprop.select_top_layer_score(out)

    assert_that(score.size()).is_equal_to(Size([]))
    assert_that(score).is_equal_to(1)


def test_gradcam_basics(mocker, gradcam_mock_setup):
    img, base_layers = gradcam_mock_setup

    mocker.patch(
        "midnite.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(3),
    )
    GradCAM(
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
        "midnite.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(3),
    )
    sal_map = GradCAM(
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
        "midnite.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones((4, 4)),
    )
    sal_map = GradCAM(
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
        "midnite.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones(()),
    )
    sal_map = GradCAM(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        NeuronSplit(),
    ).visualize(img)

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([3, 4, 4]))
