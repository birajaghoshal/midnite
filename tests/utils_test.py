"""Unit tests for utility methods."""
import torch
from assertpy import assert_that

from vinsight import utils


def test_defaults():
    """Test reading out defaults."""
    torch.set_default_tensor_type("torch.DoubleTensor")
    dev, dtype = utils.tensor_defaults()

    assert_that(dev).is_equal_to(torch.device("cpu"))
    assert_that(dtype).is_equal_to(torch.float64)
