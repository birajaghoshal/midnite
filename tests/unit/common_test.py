"""Tests for midnite commons."""
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal

from midnite.common import Flatten


def test_flatten_flat():
    """Test flatten on 2d layer."""
    sut = Flatten()

    input_ = torch.tensor([1, 2, 3])
    output = sut(input_)

    assert_that(output.size()).is_equal_to((3,))
    assert_array_almost_equal(output, [1, 2, 3])


def test_flatten_3d():
    """Test flatten on 3d layer."""
    sut = Flatten()

    input_ = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 0.0], [0.1, 0.2]]]]
    )
    out = sut(input_)

    assert_that(out.shape).is_equal_to((1, 12))
    assert_array_almost_equal(
        out, [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2]]
    )
