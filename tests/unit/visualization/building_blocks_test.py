"""Test for the building blocks."""
import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal

from midnite.visualization.base import ChannelSplit
from midnite.visualization.base import Identity
from midnite.visualization.base import NeuronSplit
from midnite.visualization.base import SimpleSelector
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector
from midnite.visualization.base.splits import GroupSplit


@pytest.mark.parametrize(
    "split,index,size,mask",
    [
        (Identity(), [], (1, 2, 3), [[[1, 1, 1], [1, 1, 1]]]),
        (NeuronSplit(), [0, 1, 2], [1, 2, 3], [[[0, 0, 0], [0, 0, 1]]]),
        (SpatialSplit(), [0, 1], (2, 2, 2), [[[0, 1], [0, 0]], [[0, 1], [0, 0]]]),
        (
            ChannelSplit(),
            [1],
            (3, 2, 2),
            [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]],
        ),
        (
            GroupSplit(
                torch.tensor([1, 1]), torch.tensor([[0, 0, 1, 1, 0], [1, 1, 0, 0, 1]])
            ),
            [0],
            (5, 1),
            [0, 0, 1, 1, 0],
        ),
    ],
)
def test_mask(split, index, size, mask):
    """Test masking of all splits."""
    mask = split.get_mask(index, size).numpy()

    assert_array_equal(mask, mask)


def test_identity_split():
    """Check all identity masks."""
    masks = Identity().get_split([1, 2, 3])

    assert_that(masks).is_length(1)
    assert_array_equal(masks[0].numpy(), [[[1, 1, 1], [1, 1, 1]]])


def test_neuron_split():
    """Check all neuron masks."""
    masks = NeuronSplit().get_split([1, 1, 2])

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 1]]])


def test_spatial_split():
    """Check all spatial masks."""
    masks = SpatialSplit().get_split([2, 2, 2])

    assert_that(masks).is_length(4)
    assert_array_equal(masks[0].numpy(), [[[1, 0], [0, 0]], [[1, 0], [0, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
    assert_array_equal(masks[2].numpy(), [[[0, 0], [1, 0]], [[0, 0], [1, 0]]])
    assert_array_equal(masks[3].numpy(), [[[0, 0], [0, 1]], [[0, 0], [0, 1]]])


def test_spatial_split_dims():
    """Check all spatial masks for all different dimensions."""
    masks = SpatialSplit().get_split([3, 2, 1])

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1], [0]], [[1], [0]], [[1], [0]]])
    assert_array_equal(masks[1].numpy(), [[[0], [1]], [[0], [1]], [[0], [1]]])


def test_channel_split():
    """Check all channel masks."""
    masks = ChannelSplit().get_split([2, 2, 2])

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 0], [0, 0]], [[1, 1], [1, 1]]])


def test_channel_split_dims():
    """Check all channel masks for all different dimensions."""
    masks = ChannelSplit().get_split([3, 2, 1])

    assert_that(masks).is_length(3)
    assert_array_equal(masks[0].numpy(), [[[1], [1]], [[0], [0]], [[0], [0]]])
    assert_array_equal(masks[1].numpy(), [[[0], [0]], [[1], [1]], [[0], [0]]])
    assert_array_equal(masks[2].numpy(), [[[0], [0]], [[0], [0]], [[1], [1]]])


def test_split_selector():
    """Test if the neuron selector correctly selects its neurons."""
    selection = SplitSelector(SpatialSplit(), [0, 1]).get_mask([2, 2, 2]).numpy()

    assert_array_equal(selection, [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])


def test_simple_selector():
    """Simple test for simple selectors."""
    mask = torch.ones((10, 10))
    selection = SimpleSelector(mask).get_mask([10, 10])
    assert_that(mask).is_same_as(selection)


def test_neuron_mean():
    """Check the correct computation of neuron split mean."""
    input_ = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    mean = NeuronSplit().get_mean(input_)
    assert_array_equal(mean.numpy(), input_)
    assert_that(mean.size()).is_equal_to(input_.size())


def test_channel_mean():
    """Check the correct computation of the channel mean."""
    input_ = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 7.0], [2.0, 3.0]]]
    )
    mean = ChannelSplit().get_mean(input_)
    assert_array_equal(mean.numpy(), [2.5, 6.5, 5.25])
    # output dim = number of channels
    assert_that(mean.numpy().shape).is_equal_to((3,))


def test_spatial_mean():
    """Check the correct computation of spatial mean."""
    input_ = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 7.0], [2.0, 3.0]]]
    )
    mean = SpatialSplit().get_mean(input_)
    assert_array_equal(mean.numpy(), [[5.0, 5.0], [4.0, 5.0]])
    assert_that(mean.numpy().shape).is_equal_to(input_[0].numpy().shape)


def test_identity_mean():
    """Check the correct mean computation for the identity split."""
    input_ = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 7.0], [2.0, 3.0]]]
    )
    mean = Identity().get_mean(input_)
    assert_array_equal(mean.numpy(), input_.mean().numpy())


@pytest.mark.parametrize(
    "class_,inverse",
    [
        (NeuronSplit, Identity),
        (Identity, NeuronSplit),
        (SpatialSplit, ChannelSplit),
        (ChannelSplit, SpatialSplit),
    ],
)
def test_invert_neuron(class_, inverse):
    """Check that split inverses are properly configured."""
    split = class_().invert()

    assert_that(split).is_instance_of(inverse)


@pytest.mark.parametrize(
    "in_dim,split,out_dim",
    [
        ((4, 5), SpatialSplit(), (1, 4, 5)),
        (4, ChannelSplit(), (4, 1, 1)),
        ((3, 4, 5), NeuronSplit(), (3, 4, 5)),
        ((), Identity(), (1, 1, 1)),
        ((3, 4, 5), GroupSplit(torch.ones(3, 2), torch.ones(2, 4, 5)), (3, 4, 5)),
    ],
)
def test_fill_dimension(in_dim, split, out_dim):
    """Check that splits properly fill the cube dimensions."""
    input_ = torch.ones(in_dim)
    filled = split.fill_dimensions(input_)

    assert_that(tuple(filled.size())).is_equal_to(out_dim)
    # Checksum
    assert_that(filled.sum()).is_equal_to(input_.sum())
