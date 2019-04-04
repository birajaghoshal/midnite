import pytest
import torch
from assertpy import assert_that

from midnite import device
from midnite import get_device


def test_context_stack(mocker):
    """Check that the context stack is pushed and popped correctly."""
    # Default is cpu
    assert_that(get_device()).is_equal_to(torch.device("cpu"))
    # Patch out torch
    mock_device = mocker.Mock(spec=torch.device)
    mocker.patch("torch.device", return_value=mock_device)

    # Context
    with device("cuda:0"):

        torch.device.assert_called_once_with("cuda:0")
        torch.device.reset_mock()
        assert_that(get_device()).is_equal_to(mock_device)

        with device("cuda:1"):

            torch.device.assert_called_once_with("cuda:1")
            assert_that(get_device()).is_equal_to(mock_device)

    assert_that(get_device()).is_not_equal_to(mock_device)


def test_context_fail_early(mocker):
    """Checks that creating a context with invalid device immediately fails"""
    with pytest.raises(TypeError):
        with device("does not exist"):
            pass
