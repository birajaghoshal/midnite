"""Numerical tests for the functional module"""
import numpy as np
import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from midnite.uncertainty import functional


def test_sample_mean():
    """Test the mean calculation."""
    input_ = torch.tensor([[[0.0, 1.0, 0.2], [1.0, 0.0, 0.8]]])
    output = functional.mean_prediction(input_).numpy()

    assert_array_almost_equal(output, [[0.4, 0.6]])


def test_sample_mean_batch():
    """Test the mean calculation on a mini batch of size > 1."""
    input_ = torch.tensor(
        [
            [[0.3, 0.3, 0.4, 0.4], [0.7, 0.7, 0.6, 0.6]],
            [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]],
        ]
    )
    output = functional.mean_prediction(input_).numpy()

    assert_array_almost_equal(output, [[0.35, 0.65], [0.25, 0.75]])


def test_sample_mean_image_batch():
    """Test the mean calculation on a mini batch (size > 1) of images"""
    # Two samples with minibatch size 3, 4 output classes, 5x5 images
    sample_1 = torch.ones(3, 4, 5, 5)
    sample_2 = torch.zeros(3, 4, 5, 5)

    output = functional.mean_prediction(torch.stack((sample_1, sample_2), dim=4))

    assert_array_equal(output.size(), [3, 4, 5, 5])
    assert_that(torch.all(output == 0.5)).is_true()


@pytest.mark.parametrize("inplace", [True, False])
def test_pred_entropy(inplace):
    """Test the predictive entropy function."""
    input_ = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]]])
    output = functional.predictive_entropy(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [[0.3218, 0.1785]], decimal=4)


@pytest.mark.parametrize("inplace", [True, False])
def test_pred_entropy_batch(inplace):
    """Test the entropy calculation on a mini batch of size > 1."""
    input_ = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            [[0.2, 0.3, 0.1, 0.6], [0.8, 0.7, 0.9, 0.4]],
        ]
    )
    output = functional.predictive_entropy(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [[0.3466, 0.2158], [0.3612, 0.2497]], decimal=4)


@pytest.mark.parametrize("inplace", [True, False])
def test_pred_entropy_image_batch(inplace):
    """Test predictive entropy on a mini batch (size > 1) of images."""
    sample_1 = torch.ones(3, 4, 5, 5)
    sample_2 = torch.zeros(3, 4, 5, 5)

    output = functional.predictive_entropy(
        torch.stack((sample_1, sample_2), dim=4), inplace=inplace
    )

    assert_array_equal(output.size(), [3, 4, 5, 5])
    assert_array_almost_equal(output.numpy(), np.ones((3, 4, 5, 5)) * 0.3457, decimal=2)


@pytest.mark.parametrize("inplace", [True, False])
def test_mutual_information(inplace):
    """Test the mutual information function."""
    input_ = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]]])
    output = functional.mutual_information(input_.clone(), inplace=inplace).numpy()

    assert_array_almost_equal(output, [[0.3219, 0.1785]], decimal=4)

    # More confident prediction
    input_ = torch.tensor([[[0.2, 0.3, 0.1], [0.8, 0.7, 0.9]]])
    output = functional.mutual_information(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [[0.0174, 0.0042]], decimal=4)


@pytest.mark.parametrize("inplace", [True, False])
def test_mutual_information_batch(inplace):
    """Test the mutual information on a mini batch of size > 1."""
    input_ = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            [[0.2, 0.3, 0.1, 0.2], [0.8, 0.7, 0.9, 0.8]],
        ]
    )
    output = functional.mutual_information(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [[0.3466, 0.2158], [0.0131, 0.0031]], decimal=4)


@pytest.mark.parametrize("inplace", [True, False])
def test_mutual_information_image_batch(inplace):
    """Test the mutual information calculation on a mini-batch (size > 1) of images"""
    sample_1 = torch.ones(3, 4, 5, 5)
    sample_2 = torch.zeros(3, 4, 5, 5)

    output = functional.mutual_information(
        torch.stack((sample_1, sample_2), dim=4), inplace=inplace
    )

    assert_array_equal(output.size(), [3, 4, 5, 5])
    assert_array_almost_equal(output.numpy(), np.ones((3, 4, 5, 5)) * 0.3457, decimal=2)


@pytest.mark.parametrize("inplace", [True, False])
def test_variation_ratio(inplace):
    """Test the variation ratio function."""
    input_ = torch.tensor([[[0.0, 1.0, 0.2], [1.0, 0.0, 0.8]]])
    output = functional.variation_ratio(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [0.4])


@pytest.mark.parametrize(
    "acquisition_function",
    [
        functional.variation_ratio,
        functional.predictive_entropy,
        functional.mutual_information,
    ],
)
def test_variation_non_probabilistic(acquisition_function):
    """Test the variation ratio function with non-probabilistic inputs"""
    input_ = torch.tensor([[[0.0, 2.0, 0.4], [2.0, 0.0, 1.6]]])
    with pytest.raises(ValueError):
        acquisition_function(input_)


@pytest.mark.parametrize("inplace", [True, False])
def test_variation_ratio_batch(inplace):
    """Test the variation ratio on a mini batch of size > 1."""
    input_ = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            [[0.2, 0.3, 0.1, 0.2], [0.8, 0.7, 0.9, 0.8]],
        ]
    )
    output = functional.variation_ratio(input_, inplace=inplace).numpy()

    assert_array_almost_equal(output, [0.25, 0.2])


@pytest.mark.parametrize("inplace", [True, False])
def test_variation_ratio_image_batch(inplace):
    """Test the variation ratio on a mini-batch (size > 1) of images"""
    # 3 in batch with 4 classes of 5x5 images, using 2 samples
    sample_1 = torch.div(torch.ones(3, 4, 5, 5), 2)
    sample_2 = torch.zeros(3, 4, 5, 5)

    input_ = torch.stack((sample_1, sample_2), dim=4)
    output = functional.variation_ratio(input_, inplace=inplace)

    assert_array_equal(output.size(), [3, 5, 5])
    assert_array_almost_equal(output.numpy(), np.ones((3, 5, 5)) * 0.75)


@pytest.mark.parametrize("inplace", [True, False])
def test_mutual_information_zero(inplace):
    """Mutual information should be zero on equal predictions"""
    input_ = torch.ones((1, 1000, 200)) * 0.001
    res = functional.mutual_information(input_, inplace=inplace)
    assert torch.all(res == 0)
