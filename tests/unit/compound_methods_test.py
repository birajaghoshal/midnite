"""Unit test for the compound methods."""
import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal
from torch.nn import functional
from torch.nn import Module

from midnite.visualization import compound_methods
from midnite.visualization.compound_methods import _prepare_input
from midnite.visualization.compound_methods import _top_k_selector
from midnite.visualization.compound_methods import _upscale
from midnite.visualization.compound_methods import guided_gradcam


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


def test_upscale(mocker):
    """Check upscale wiring."""
    scaled = torch.zeros((4, 4))
    scaled.squeeze = mocker.Mock(return_value=scaled)
    mocker.patch("torch.nn.functional.interpolate", return_value=scaled)
    img = torch.zeros(2, 2)
    img.unsqueeze = mocker.Mock(return_value=img)

    res = _upscale(img, (4, 4))

    functional.interpolate.assert_called_with(
        img, size=(4, 4), mode="bilinear", align_corners=True
    )
    assert_that(res).is_same_as(scaled)
    assert_that(scaled.squeeze.call_count).is_equal_to(2)
    assert_that(img.unsqueeze.call_count).is_equal_to(2)


def test_guided_gradcam(mocker):
    """Check the guided gradcam wiring."""
    input_ = torch.zeros((3, 5, 5))
    gradcam_out = torch.zeros((2, 2))
    scaled_out = torch.ones((5, 5)).mul_(2)
    backprop_out = torch.ones((5, 5)).mul_(3)
    mocker.patch(
        "midnite.visualization.compound_methods.gradcam", return_value=gradcam_out
    )
    mocker.patch(
        "midnite.visualization.compound_methods.guided_backpropagation",
        return_value=backprop_out,
    )
    mocker.patch(
        "midnite.visualization.compound_methods._upscale", return_value=scaled_out
    )

    res = guided_gradcam([mocker.Mock(spec=Module)], [mocker.Mock(spec=Module)], input_)

    compound_methods.gradcam.assert_called_once()
    compound_methods.gradcam.assert_called_once()
    compound_methods._upscale.assert_called_once_with(gradcam_out, (5, 5))
    assert_that(res.size()).is_equal_to((5, 5))
    assert_that(res.sum()).is_equal_to(5 * 5 * 6)
