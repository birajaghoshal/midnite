"""Test for the building blocks"""
from numpy.testing import assert_array_equal
from torch import Size

from vinsight.visualization.building_blocks import NeuronSplit


def test_neuron_mask():
    mask = NeuronSplit().get_mask([0, 1, 2], Size([1, 2, 3])).numpy()
    assert_array_equal(mask, [[[0, 0, 0], [0, 0, 1]]])


def test_neuron_split():
    masks = NeuronSplit().get_split(Size([1, 1, 2]))
    assert_array_equal(masks[0].numpy(), [[[1, 0]]])
    assert_array_equal(masks[1].numpy(), [[[0, 1]]])
