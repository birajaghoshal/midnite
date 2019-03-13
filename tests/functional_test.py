"""Numerical tests for the functional module"""
import torch
from numpy.testing import assert_array_almost_equal

from interpretability_framework import functional


def test_sample_mean():
    """Test the mean calculation.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = functional.sample_mean(input_).numpy()

    assert_array_almost_equal(output, [0.4, 0.6])


def test_sample_mean_batch():
    """Test the mean calculation on a mini batch.

    """
    input_ = torch.tensor(
        [
            [[0.3, 0.7], [0.1, 0.9]],
            [[0.3, 0.7], [0.2, 0.8]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.4, 0.6], [0.4, 0.6]],
        ]
    )
    output = functional.sample_mean(input_).numpy()

    assert_array_almost_equal(output, [[0.35, 0.65], [0.25, 0.75]])


def test_pred_entropy():
    """Test the predictive entropy function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = functional.predictive_entropy(input_).numpy()

    assert_array_almost_equal(output, [0.3218, 0.1785], decimal=4)


def test_pred_entropy_batch():
    """Test the entropy calculation on a mini batch.

    """
    input_ = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.0, 1.0], [0.3, 0.7]],
            [[0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.6, 0.4]],
        ]
    )
    output = functional.predictive_entropy(input_).numpy()

    assert_array_almost_equal(output, [[0.3466, 0.2158], [0.3612, 0.2497]], decimal=4)


def test_mutual_information():
    """Test the mutual information function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = functional.mutual_information(input_)

    assert output.allclose(torch.tensor([0.3219, 0.1785]), atol=1e-4)

    # More confident prediction
    input_ = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    output = functional.mutual_information(input_).numpy()

    assert_array_almost_equal(output, [0.0174, 0.0042], decimal=4)


def test_mutual_information_batch():
    """Test the mutual information on a mini batch.

    """

    input_ = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.0, 1.0], [0.3, 0.7]],
            [[0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    output = functional.mutual_information(input_).numpy()

    assert_array_almost_equal(output, [[0.3466, 0.2158], [0.0131, 0.0031]], decimal=4)


def test_variation_ratio():
    """Test the variation ratio function.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = functional.variation_ratio(input_).numpy()

    assert_array_almost_equal(output, [0.4])


def test_variation_ratio_batch():
    """Test the variation ratio on a mini batch.

    """
    input_ = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.0, 1.0], [0.3, 0.7]],
            [[0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    output = functional.variation_ratio(input_).numpy()

    assert_array_almost_equal(output, [0.25, 0.2])
