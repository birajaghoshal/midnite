"""Unit tests for the attribution methods."""
from unittest.mock import PropertyMock

import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import Sequential

from midnite.visualization.base import Backpropagation
from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import LayerSplit
from midnite.visualization.base import methods
from midnite.visualization.base import NeuronSelector
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import Occlusion
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector


@pytest.fixture
def tiny_net():
    layer0 = torch.nn.Linear(3, 3, bias=False)
    layer0.weight = Parameter(torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).float())
    layer1 = torch.nn.Linear(3, 2, bias=False)
    layer1.weight = Parameter(torch.tensor([[-1, 3, 0], [2, -1, 1]]).float())
    return Sequential(layer0, layer1)


@pytest.fixture
def id_img(mocker):
    """Setup an image where .clone/.detach etc return id."""
    img = mocker.Mock(spec=torch.Tensor)
    # We want to have the exact same img after these operations because that makes
    # testing for equality in assert_called_with easy
    img.clone = mocker.Mock(return_value=img)
    img.detach = mocker.Mock(return_value=img)
    img.to = mocker.Mock(return_value=img)
    img.requires_grad_ = mocker.Mock(return_value=img)

    return img


@pytest.fixture
def layer_mock_setup(mocker):
    out = (
        torch.zeros((1, 3, 4, 4)),
        torch.ones((1, 3, 4, 4)),
        torch.ones((1, 1, 4, 4)),
    )

    layers = [
        mocker.MagicMock(spec=Module, return_value=out[0]),
        mocker.MagicMock(spec=Module, return_value=out[1]),
    ]

    out[1].detach = mocker.Mock(return_value=out[1])
    out[1].squeeze = mocker.Mock(return_value=out[1])
    out[1].mul_ = mocker.Mock(return_value=out[2])

    for layer in layers:
        layer.to = mocker.Mock(return_value=layer)

    return layers, out


@pytest.fixture
def backprop_mock_setup(mocker):
    """Setup layers, selectors, and (intermediate) outputs."""
    # All intermediate outputs
    out = (torch.ones((1, 3, 4, 4)), torch.ones((1, 4, 4)))

    # Wire mocks/outputs together
    mocker.patch(
        "midnite.visualization.base.methods._calculate_single_mean", return_value=out[0]
    )
    mocker.patch("torch.autograd.backward")

    bottom_split = mocker.Mock(spec=LayerSplit)
    bottom_split.get_mean = mocker.Mock(return_value=out[1])

    return out, bottom_split


@pytest.fixture
def gradcam_mock_setup(mocker):
    """Patch splits and setup img and base layers"""
    out = (
        torch.ones((1, 3)),
        torch.ones((1, 3, 1, 1)),
        torch.ones((1, 1, 4, 4)),
        torch.ones((1, 1, 4, 4)),
    )
    out[0].detach = mocker.Mock(return_value=out[0])
    out[0].squeeze = mocker.Mock(return_value=out[0])

    # Wire mocks together
    mocker.patch(
        "midnite.visualization.base.methods.Backpropagation.visualize",
        return_value=out[0],
    )
    mocker.patch("torch.nn.functional.relu", return_value=out[3])

    bottom_split = mocker.Mock(spec=LayerSplit)
    bottom_split2 = mocker.Mock(spec=LayerSplit)

    bottom_split.get_mean = mocker.Mock(return_value=out[2])
    bottom_split.invert = mocker.Mock(return_value=bottom_split2)
    bottom_split2.fill_dimensions = mocker.Mock(return_value=out[1])

    return (bottom_split, bottom_split2), out


@pytest.mark.parametrize(
    "split,index,mean",
    [
        (ChannelSplit(), [1], 4.5),
        (NeuronSplit(), [0, 1, 1], 4),
        (SpatialSplit(), [1, 1], 5),
    ],
)
def test_calculate_single_mean(split, index, mean):
    """Checks mean calculations of the given tensor with different splits."""
    input_ = torch.tensor(
        [[[2, 3], [0, 4]], [[1, 1], [7, 9]], [[0, 1], [0, 2]]]
    ).float()
    select = SplitSelector(split, index)

    result = methods._calculate_single_mean(input_, select)
    assert_that(result.item()).is_close_to(mean, tolerance=1e-40)


def test_backpropagation_wiring(mocker, backprop_mock_setup, layer_mock_setup, id_img):
    """Check that layers are in correct mode and the gradient is calculated properly."""
    grad = torch.zeros((1, 3, 4, 4))
    grad.detach = mocker.Mock(return_value=grad)
    grad.squeeze = mocker.Mock(return_value=grad)
    id_img.grad = PropertyMock(return_value=grad)
    out, bottom_split = backprop_mock_setup
    layers, layer_out = layer_mock_setup
    top_layer_sel = mocker.Mock(spec=NeuronSelector)
    sut = Backpropagation(Sequential(*layers), top_layer_sel, bottom_split)

    result = sut.visualize(id_img)

    # Check preprocessing
    id_img.clone.assert_called_once()
    id_img.detach.assert_called_once()
    id_img.to.assert_called_once()
    id_img.requires_grad_.assert_called_once()

    layers[0].train.assert_called_once_with(False)
    layers[1].train.assert_called_once_with(False)

    # Check forward pass
    layers[0].assert_called_once()
    layers[1].assert_called_once()

    # Check gradient calculation
    # 1. Output mean
    methods._calculate_single_mean.assert_called_with(layer_out[1], top_layer_sel)
    # 2. Take gradient
    torch.autograd.backward.assert_called_once()
    # 4. Split mean
    bottom_split.get_mean.assert_called_once()
    # 5. Check result
    assert_that(result).is_same_as(out[1])


def test_gradcam_wiring(mocker, gradcam_mock_setup, layer_mock_setup, id_img):
    """Check gradam calculation."""
    layers, layer_out = layer_mock_setup
    split, out = gradcam_mock_setup

    sut = GradAM(
        mocker.Mock(spec=torch.nn.Module),
        mocker.Mock(spec=NeuronSelector),
        Sequential(*layers),
        split[0],
    )

    result = sut.visualize(id_img)

    # Check preprocessing
    id_img.clone.assert_called_once()
    id_img.detach.assert_called_once()
    id_img.to.assert_called_once()

    layers[0].train.assert_called_once_with(False)
    layers[1].train.assert_called_once_with(False)

    # Check Calculation
    # 1. forward pass
    layers[0].assert_called_once()
    layers[1].assert_called_once()
    # 2. backpropagation
    methods.Backpropagation.visualize.assert_called_with(layer_out[1])
    # 3. Split invert and fill dimensions
    split[0].invert.assert_called()
    split[1].fill_dimensions(out)  # -> out[1]
    # 4. Multiply together
    layer_out[1].mul_.assert_called_once_with(out[1])  # -> layer_out[2]
    # 5. Calculate mean
    split[0].get_mean.assert_called_with(layer_out[2])
    # 6. Relu
    torch.nn.functional.relu.assert_called_once_with(out[2])
    # Check result
    assert_that(result).is_same_as(out[3])


def test_remove_chunk(mocker):
    sut = Occlusion(
        mocker.Mock(spec=Module),
        mocker.Mock(spec=NeuronSelector),
        mocker.Mock(spec=LayerSplit),
        [1, 5, 5],
        stride_length=[1, 1, 1],
    )
    img = torch.ones((1, 10, 10))
    result = sut._remove_chunk(img, torch.tensor([[[0, 0], [1, 0]]]).float())

    assert_that(result.size()).is_equal_to((1, 10, 10))
    assert torch.all(result[0, :5] == 1.0)
    assert torch.all(result[0, 5:, :5] == 0.5)
    assert torch.all(result[0, 5:, 5:] == 1.0)


def test_backpropagation(tiny_net):
    """Checks backpropagation calculation with a tiny example."""
    input_ = torch.tensor([1, 3, 2]).float()
    sut = Backpropagation(tiny_net, SplitSelector(NeuronSplit(), [1]), NeuronSplit())
    assert_array_equal(sut.visualize(input_).numpy(), [4, -2, -2])


def test_guided_backpropagation(tiny_net):
    """Checks guided backpropagation calculation with a tiny example."""
    input_ = torch.tensor([1, 1, 2]).float()
    layers = list(tiny_net.children())
    sut = GuidedBackpropagation(
        layers, SplitSelector(NeuronSplit(), [1]), NeuronSplit()
    )
    assert_array_equal(sut.visualize(input_).numpy(), [4, 0, 0])


def test_gradam(tiny_net):
    """Checks gradam calculation with tiny example, using NeuronSplit."""
    input_ = torch.tensor([1, 1, 2]).float()
    layers = list(tiny_net.children())
    sut = GradAM(layers[1], SplitSelector(NeuronSplit(), [1]), layers[0], NeuronSplit())
    assert_array_almost_equal(
        sut.visualize(input_).numpy(), [1.333, 1.333, 0], decimal=3
    )
