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
from midnite.visualization.base.methods import common


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
def layer_out(mocker):
    out = (torch.zeros((1, 3, 4, 4)), torch.ones((1, 3, 4, 4)))
    out[1].detach = mocker.Mock(return_value=out[1])
    out[1].squeeze = mocker.Mock(return_value=out[1])
    return out


@pytest.fixture
def layers(mocker, layer_out):
    layers = [
        mocker.MagicMock(spec=Module, return_value=layer_out[0]),
        mocker.MagicMock(spec=Module, return_value=layer_out[1]),
    ]

    for layer in layers:
        layer.to = mocker.Mock(return_value=layer)

    return layers


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

    result = common._calculate_single_mean(input_, select)
    assert_that(result.item()).is_close_to(mean, tolerance=1e-40)


def test_backpropagation_wiring(mocker, layers, layer_out, id_img):
    """Check that layers are in correct mode and the gradient is calculated properly."""
    out = (torch.ones((1, 3, 4, 4)), torch.ones((1, 4, 4)))

    # Wire mocks/outputs together
    mocker.patch(
        "midnite.visualization.base.methods.common._calculate_single_mean",
        return_value=out[0],
    )
    mocker.patch("torch.autograd.backward")

    bottom_split = mocker.Mock(spec=LayerSplit)
    bottom_split.get_mean = mocker.Mock(return_value=out[1])
    grad = torch.zeros((1, 3, 4, 4))
    grad.detach = mocker.Mock(return_value=grad)
    grad.squeeze = mocker.Mock(return_value=grad)
    id_img.grad = PropertyMock(return_value=grad)
    top_layer_sel = mocker.Mock(spec=NeuronSelector)
    sut = Backpropagation(Sequential(*layers), top_layer_sel, bottom_split)

    result = sut.visualize(id_img)

    # Check preprocessing
    id_img.clone.assert_called_once()
    id_img.detach.assert_called_once()
    id_img.to.assert_called_once()
    id_img.requires_grad_.assert_called_once()

    layers[0].train.assert_not_called()
    layers[1].train.assert_not_called()

    # Check forward pass
    layers[0].assert_called_once()
    layers[1].assert_called_once()

    # Check gradient calculation
    # 1. Output mean
    common._calculate_single_mean.assert_called_with(layer_out[1], top_layer_sel)
    # 2. Take gradient
    torch.autograd.backward.assert_called_once()
    # 4. Split mean
    bottom_split.get_mean.assert_called_once()
    # 5. Check result
    assert_that(result).is_same_as(out[1])


def test_gradcam_wiring(mocker, layers, layer_out, id_img):
    """Check gradam calculation."""
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

    split = (mocker.Mock(spec=LayerSplit), mocker.Mock(spec=LayerSplit))

    split[0].get_mean = mocker.Mock(return_value=out[2])
    split[0].invert = mocker.Mock(return_value=split[1])
    split[1].fill_dimensions = mocker.Mock(return_value=out[1])

    mul_out = torch.zeros((1, 1, 3, 4))
    layer_out[1].mul_ = mocker.Mock(return_value=mul_out)

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

    layers[0].train.assert_not_called()
    layers[1].train.assert_not_called()

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
    split[0].get_mean.assert_called_with(mul_out)
    # 6. Relu
    torch.nn.functional.relu.assert_called_once_with(out[2])
    # Check result
    assert_that(result).is_same_as(out[3])


@pytest.mark.parametrize(
    "size,smear,out",
    [
        (2, 1, [[1, 0], [0, 1]]),
        (2, 3, [[1, 0], [1, 1]]),
        (3, 2, [[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        (
            5,
            3,
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
            ],
        ),
    ],
)
def test_smear_matrix(size, smear, out):
    matrix = Occlusion._smear_matrix(size, smear)
    assert_array_equal(matrix.numpy(), out)


def test_chunk_matrixes(mocker):
    mocker.patch(
        "midnite.visualization.base.methods.occlusion.Occlusion._smear_matrix",
        return_value=mocker.Mock(spec=torch.Tensor),
    )
    sut = Occlusion(
        mocker.Mock(spec=Module),
        mocker.Mock(spec=NeuronSelector),
        mocker.Mock(spec=LayerSplit),
        [3, 2, 4],
        [1, 1, 1],
    )
    matrices = sut._chunk_matrixes([3, 5, 6])
    assert_that(matrices).is_length(3)
    assert_that(Occlusion._smear_matrix.call_count).is_equal_to(3)
    Occlusion._smear_matrix.assert_any_call(7, 2)
    Occlusion._smear_matrix.assert_any_call(6, 3)
    Occlusion._smear_matrix.assert_any_call(10, 4)


def test_chunk_mask(mocker):
    """Checks that selected pixels are correctly transformed into chunk masks."""
    sut = Occlusion(
        mocker.Mock(spec=Module),
        mocker.Mock(spec=NeuronSelector),
        mocker.Mock(spec=LayerSplit),
        [2, 1, 3],
        [1, 1, 1],
    )
    pixel_mask = torch.Tensor(
        [
            [[0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    mask = sut._chunk_mask(pixel_mask, sut._chunk_matrixes([3, 2, 4]))
    assert_array_equal(
        mask.numpy(),
        [
            [[1, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
        ],
    )


def test_remove_chunk(mocker):
    """Test that the correct chunk gets replaced by a grey square."""
    sut = Occlusion(
        mocker.Mock(spec=Module),
        mocker.Mock(spec=NeuronSelector),
        mocker.Mock(spec=LayerSplit),
        [1, 5, 5],
        stride=[1, 1, 1],
    )
    img = torch.ones((1, 10, 10))
    result = sut._remove_chunk(img, torch.tensor([[[0, 0], [1, 0]]]).float())

    assert_that(result.size()).is_equal_to((1, 10, 10))
    assert torch.all(result[0, :5] == 1.0)
    assert torch.all(result[0, 5:, :5] == 0.5)
    assert torch.all(result[0, 5:, 5:] == 1.0)


def test_occlusion_wiring(mocker, id_img, layers, layer_out):
    """Test the occlusion method."""
    # Mock
    id_img.squeeze = mocker.Mock(return_value=id_img)
    id_img.size = mocker.Mock(return_value=torch.Size([3, 4, 4]))

    mask = mocker.Mock(spec=torch.Tensor)
    mask.unsqueeze = mocker.Mock(return_value=mask)
    matrixes = [
        mocker.Mock(spec=torch.Tensor),
        mocker.Mock(spec=torch.Tensor),
        mocker.Mock(spec=torch.Tensor),
    ]

    mocker.patch("midnite.visualization.base.Occlusion._chunk_mask", return_value=mask)
    mocker.patch(
        "midnite.visualization.base.Occlusion._chunk_matrixes", return_value=matrixes
    )
    mocker.patch(
        "midnite.visualization.base.Occlusion._remove_chunk",
        return_value=mocker.Mock(spec=torch.Tensor),
    )
    return_value = mocker.Mock(spec=torch.Tensor)
    mocker.patch("torch.zeros", return_value=return_value)

    split = mocker.Mock(spec=NeuronSplit)
    splits = [mocker.Mock(spec=torch.Tensor), mocker.Mock(spec=torch.Tensor)]
    split.get_split = mocker.Mock(return_value=splits)
    split.get_mean = mocker.Mock(return_value=mocker.Mock(spec=torch.Tensor))
    sel = mocker.Mock(spec=NeuronSelector)
    pred_mask = torch.Tensor([[1, 1], [0, 0]])
    sel.get_mask = mocker.Mock(return_value=pred_mask)
    pred = mocker.Mock(spec=torch.Tensor)
    layer_out[1].mul_ = mocker.Mock(return_value=pred)

    # Execute methods
    sut = Occlusion(Sequential(*layers), sel, split, [4, 5, 6], [1, 2, 3])
    sut.visualize(id_img)

    # Verify
    assert_that(layers[0].call_count).is_equal_to(3)
    sel.get_mask.assert_called_with([1, 3, 4, 4])
    assert_array_equal(pred_mask, [[0.5, 0.5], [0, 0]])
    layer_out[1].mul_.assert_called_with(pred_mask)
    sut._chunk_matrixes.assert_called_with([3, 2, 1])
    split.get_split.assert_called_with([3, 2, 1])
    sut._chunk_mask.assert_any_call(splits[0], matrixes)
    sut._chunk_mask.assert_any_call(splits[1], matrixes)
    sut._remove_chunk.assert_called_with(id_img, mask)
    assert_that(sut._remove_chunk.call_count).is_equal_to(2)
    assert_that(return_value.add_.call_count).is_equal_to(2)
    split.get_mean.assert_called_with(return_value)


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
