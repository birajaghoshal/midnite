"""Integration test of compound visualization methods against torch."""
from unittest.mock import PropertyMock

import pytest
import torch
from assertpy import assert_that

from midnite.common import Flatten
from midnite.visualization import gradcam
from midnite.visualization.base import PixelActivation
from midnite.visualization.compound_methods import class_visualization
from midnite.visualization.compound_methods import saliency_map


@pytest.fixture(scope="module")
def single_img(img):
    """Fixture for image without minibatch dimensions."""
    return img.squeeze(0)


def test_saliency_map(net, single_img):
    result = saliency_map(net, single_img)

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_gradcam(net, single_img):
    result = gradcam(
        [net.features, net.avgpool], [Flatten(), net.classifier], single_img
    )

    assert_that(result.size()).is_equal_to(single_img.size()[1:])
    assert torch.all(result >= 0)
    assert_that(result.sum().item()).is_greater_than(0)


def test_class_visualization(mocker, net):
    # Limit number of iterations
    mocker.patch.object(
        PixelActivation, "opt_n", create=True, new_callable=PropertyMock, return_value=2
    )
    mocker.patch.object(
        PixelActivation,
        "iter_n",
        create=True,
        new_callable=PropertyMock,
        return_value=2,
    )
    result = class_visualization(net, 283)
    assert_that(result.size()).is_length(3)
    assert_that(result.size(0)).is_equal_to(3)
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)
