import torch

from interpretability_framework import functional as F


def test_sample_mean():
    """Test the mean calculation.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = F.sample_mean(input_)

    assert output.allclose(torch.tensor([0.4, 0.6]))


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
    output = F.sample_mean(input_)

    assert output.allclose(torch.tensor([[0.35, 0.65], [0.25, 0.75]]), atol=1e-4)


def test_pred_entropy():
    """Test the predictive entropy function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = F.predictive_entropy(input_)

    assert output.allclose(torch.tensor([0.3218, 0.1785]), atol=1e-4)


def test_pred_entropy_batch():
    """Test teh entropy calculation on a mini batch.

    """
    input_ = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.0, 1.0], [0.3, 0.7]],
            [[0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.6, 0.4]],
        ]
    )
    output = F.predictive_entropy(input_)

    assert output.allclose(
        torch.tensor([[0.3466, 0.2158], [0.3612, 0.2497]]), atol=1e-4
    )


def test_mutual_information():
    """Test the mutual information function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = F.mutual_information(input_)

    assert output.allclose(torch.tensor([0.3219, 0.1785]), atol=1e-4)

    # More confident prediction
    input_ = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    output = F.mutual_information(input_)

    assert output.allclose(torch.tensor([0.0174, 0.0042]), atol=1e-4)


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
    output = F.mutual_information(input_)

    assert output.allclose(
        torch.tensor([[0.3466, 0.2158], [0.0131, 0.0031]]), atol=1e-4
    )


def test_variation_ratio():
    """Test the variation ratio function.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = F.variation_ratio(input_)

    assert output.allclose(torch.tensor([0.4]))


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
    output = F.variation_ratio(input_)

    assert output.allclose(torch.tensor([0.25, 0.2]))
