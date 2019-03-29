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


def test_neuron_value():
    """Test if the neuron selector correctly retrieves a neuron value from an input tensor"""
    # use test input with unique values and dimensions
    input_3d = torch.tensor([[[1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10]]])
    input_1d = torch.tensor([1, 2, 3, 4, 5])
    value = NeuronSelector(NeuronSplit(), [0, 0, 0]).get_value(input_3d)
    assert_array_equal(value.numpy(), 1)
    value = NeuronSelector(NeuronSplit(), [2]).get_value(input_1d)
    assert_array_equal(value.numpy(), 3)


def test_channel_value():
    """Test if the neuron selector correctly retrieves a channel from an input tensor"""
    # use test input with unique values and dimensions
    input_ = torch.tensor([[[1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10]]])
    value = NeuronSelector(ChannelSplit(), [0]).get_value(input_)
    assert_array_equal(value, [[1, 2, 3, 4, 5]])


def test_spatial_value():
    """Test if the neuron selector correctly retrieves a spatial from an input tensor"""
    # use test input with unique values and dimensions
    input_ = torch.tensor([[[1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10]]])
    value = NeuronSelector(SpatialSplit(), [0, 0]).get_value(input_)
    assert_array_equal(value, [1, 2, 3, 4, 5])
