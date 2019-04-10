"""Unit test for the compound methods."""
import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal

from midnite.visualization.compound_methods import _prepare_input
from midnite.visualization.compound_methods import _top_k_selector


@pytest.fixture
def img(mocker):
    img = torch.zeros((3, 4, 4))
    img.detach = mocker.Mock(return_value=img)
    img.to = mocker.Mock(return_value=img)
    return img


@pytest.fixture
def img_clone(mocker, img):
    img_clone = torch.zeros((3, 4, 4))
    img_clone.clone = mocker.Mock(return_value=img)
    return img_clone


def test_prepare_input(img, img_clone):
    """Test image preparations."""
    out = _prepare_input(img_clone)

    img_clone.clone.assert_called_once()
    img.detach.assert_called_once()
    img.to.assert_called_once()
    assert_that(out).is_same_as(img)


def test_prepare_invalid_sizes():
    """Test errors for input that are not images."""
    with pytest.raises(ValueError):
        _prepare_input(torch.zeros((2, 4, 4)))
    with pytest.raises(ValueError):
        _prepare_input(torch.zeros(1, 3, 4, 4))
    with pytest.raises(ValueError):
        _prepare_input(torch.zeros(2, 3))


def test_top_k_selector():
    """Test that the selector selects the top k outputs."""
    out = torch.tensor([[0, 3, 0, 2, 1, 0]])
    selector = _top_k_selector(out, 2)
    assert_array_equal(selector.get_mask([6]), [0, 1, 0, 1, 0, 0])


def test_top_k_selector_invalid_inputs():
    """Check that the selector throws errors for invalid dimensions/ks."""
    with pytest.raises(ValueError):
        _top_k_selector(torch.zeros((3, 2, 2)))
    with pytest.raises(ValueError):
        _top_k_selector(torch.tensor([1, 2]), 0)
