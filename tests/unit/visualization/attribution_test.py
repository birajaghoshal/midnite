"""Unit tests for the attribution methods."""
import pytest
import torch
from assertpy import assert_that
from torch.nn import Module

from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import LayerSplit
from midnite.visualization.base import methods
from midnite.visualization.base import NeuronSelector
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector


@pytest.fixture
def id_img(mocker):
    """Setup an image where .clone/.detach etc return id."""
    img = torch.zeros((1, 3, 4, 4))

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
    out = (
        torch.ones((1, 3, 4, 4)),
        torch.ones((1, 3, 4, 4)),
        torch.ones((1, 3, 4, 4)),
        torch.ones((1, 4, 4)),
    )

    out[1].detach = mocker.Mock(return_value=out[1])
    out[2].squeeze = mocker.Mock(return_value=out[2])

    # Wire mocks/outputs together
    mocker.patch(
        "midnite.visualization.base.methods._calculate_single_mean", return_value=out[0]
    )
    mocker.patch("torch.autograd.grad", return_value=[out[1]])
    mocker.patch("torch.nn.functional.relu", return_value=out[2])

    bottom_split = mocker.Mock(spec=LayerSplit)
    bottom_split.get_mean = mocker.Mock(return_value=out[3])

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
        "midnite.visualization.base.methods.GuidedBackpropagation.visualize",
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
    input_ = torch.tensor(
        [[[2, 3], [0, 4]], [[1, 1], [7, 9]], [[0, 1], [0, 2]]]
    ).float()
    select = SplitSelector(split, index)

    result = methods._calculate_single_mean(input_, select)
    assert_that(result.item()).is_close_to(mean, tolerance=1e-40)


def test_guided_backpropagation(mocker, backprop_mock_setup, layer_mock_setup, id_img):
    """Check that layers are in correct mode and the gradient is calculated properly."""
    out, bottom_split = backprop_mock_setup
    layers, layer_out = layer_mock_setup
    top_layer_sel = mocker.Mock(spec=NeuronSelector)
    sut = GuidedBackpropagation(layers, top_layer_sel, bottom_split)

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
    torch.autograd.grad.assert_called_once_with(out[0], id_img)
    out[1].detach.assert_called_once()
    # 3. Relu
    torch.nn.functional.relu.assert_called_once_with(out[1])
    # 4. Split mean
    bottom_split.get_mean.assert_called_once_with(out[2])
    # 5. Check result
    assert_that(result).is_same_as(out[3])


def test_gradcam(mocker, gradcam_mock_setup, layer_mock_setup, id_img):
    """Check gradam calculation."""
    layers, layer_out = layer_mock_setup
    split, out = gradcam_mock_setup

    sut = GradAM(
        [mocker.Mock(spec=torch.nn.Module)],
        mocker.Mock(spec=NeuronSelector),
        layers,
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
    methods.GuidedBackpropagation.visualize.assert_called_with(layer_out[1])
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
