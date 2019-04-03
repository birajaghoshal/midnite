"""Test for the building blocks"""
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal

from vinsight.visualization import ChannelSplit
from vinsight.visualization import NeuronSelector
from vinsight.visualization import NeuronSplit
from vinsight.visualization import SpatialSplit


def test_neuron_mask():
    """Test neuron masking"""
    mask = NeuronSplit().get_mask([0, 1, 2], [1, 2, 3]).numpy()

    assert_array_equal(mask, [[[0, 0, 0], [0, 0, 1]]])


def test_neuron_split():
    """Check all neuron masks"""
    masks = NeuronSplit().get_split([1, 1, 2])

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 1]]])


def test_spatial_mask():
    """Test spatial masking"""
    mask = SpatialSplit().get_mask([0, 1], (2, 2, 2)).numpy()

    expected_xy = [[0, 1], [0, 0]]
    assert_array_equal(mask[0], expected_xy)
    assert_array_equal(mask[1], expected_xy)


def test_spatial_split():
    """Check all spatial masks"""
    masks = SpatialSplit().get_split((2, 2, 2))

    assert_that(masks).is_length(4)
    assert_array_equal(masks[0].numpy(), [[[1, 0], [0, 0]], [[1, 0], [0, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
    assert_array_equal(masks[2].numpy(), [[[0, 0], [1, 0]], [[0, 0], [1, 0]]])
    assert_array_equal(masks[3].numpy(), [[[0, 0], [0, 1]], [[0, 0], [0, 1]]])


def test_spatial_split_dims():
    """Check all spatial masks for all different dimensions"""
    masks = SpatialSplit().get_split((3, 2, 1))

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1], [0]], [[1], [0]], [[1], [0]]])
    assert_array_equal(masks[1].numpy(), [[[0], [1]], [[0], [1]], [[0], [1]]])


def test_channel_mask():
    """Test channel masking"""
    mask = ChannelSplit().get_mask([1], (3, 2, 2)).numpy()

    assert_array_equal(mask, [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]])


def test_channel_split():
    """Check all channel masks"""
    masks = ChannelSplit().get_split((2, 2, 2))

    assert_that(masks).is_length(2)
    assert_array_equal(masks[0].numpy(), [[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 0], [0, 0]], [[1, 1], [1, 1]]])


def test_channel_split_dims():
    """Check all channel masks for all different dimensions"""
    masks = ChannelSplit().get_split((3, 2, 1))

    assert_that(masks).is_length(3)
    assert_array_equal(masks[0].numpy(), [[[1], [1]], [[0], [0]], [[0], [0]]])
    assert_array_equal(masks[1].numpy(), [[[0], [0]], [[1], [1]], [[0], [0]]])
    assert_array_equal(masks[2].numpy(), [[[0], [0]], [[0], [0]], [[1], [1]]])


def test_neuron_selector():
    """Test if the neuron selector correctly selects its neurons."""
    selection = NeuronSelector(SpatialSplit(), [0, 1]).get_mask((2, 2, 2)).numpy()

    assert_array_equal(selection, [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])


def test_neuron_mean():
    """tests if the neuron split mean is computed correctly and output has correct dimensions.
    neuron mean is the identity function."""
    input_ = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    mean = NeuronSplit().get_mean(input_)
    assert_array_equal(mean, input_)
    assert_that(mean.size()).is_equal_to(input_.size())


def test_channel_mean():
    """tests if the spatial split mean is computed correctly and output has correct dimensions.
    spatial mean output dimensions are the number of channels"""
    input_ = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 7.0], [2.0, 3.0]]]
    )
    mean = ChannelSplit().get_mean(input_)
    assert_array_equal(mean.numpy(), [2.5, 6.5, 5.25])
    # output dim = number of channels
    assert_that(mean.numpy().shape).is_equal_to((3,))


def test_spatial_mean():
    """tests if the channel split mean is computed correctly and output has correct dimensions.
    channel mean output dimension are the spatial dimensions."""
    input_ = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 7.0], [2.0, 3.0]]]
    )
    mean = SpatialSplit().get_mean(input_)
    assert_array_equal(mean.numpy(), [[5.0, 5.0], [4.0, 5.0]])
    assert_that(mean.numpy().shape).is_equal_to(input_[0].numpy().shape)


def test_invert_spatial():
    split = SpatialSplit().invert()
    assert_that(split).is_instance_of(ChannelSplit)


def test_invert_channel():
    split = ChannelSplit().invert()
    assert_that(split).is_instance_of(SpatialSplit)
