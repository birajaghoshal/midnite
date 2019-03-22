"""Test for the building blocks"""
from assertpy import assert_that
from numpy.testing import assert_array_equal

from vinsight.visualization import ChannelSplit
from vinsight.visualization import NeuronSelector
from vinsight.visualization import NeuronSplit
from vinsight.visualization import SpatialSplit


def test_neuron_mask():
    """Test neuron masking"""
    mask = NeuronSplit().get_mask([0, 1, 2], (1, 2, 3)).numpy()

    assert_array_equal(mask, [[[0, 0, 0], [0, 0, 1]]])


def test_neuron_split():
    """Check all neuron masks"""
    masks = NeuronSplit().get_split((1, 1, 2))

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


def test_neuron_selector():
    """Test if the neuron selector correctly selects its neurons."""
    selection = NeuronSelector(SpatialSplit(), [0, 1]).get_mask((2, 2, 2)).numpy()

    assert_array_equal(selection, [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
