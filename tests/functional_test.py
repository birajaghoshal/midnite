import torch

from interpretability_framework import functional as F


def test_sample_mean():
    """Test the mean calculation.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = F.sample_mean(input_)

    assert output.allclose(torch.tensor([0.4, 0.6]))


def test_pred_entropy():
    """Test the predictive entropy function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = F.predictive_entropy(input_)

    assert output.allclose(torch.tensor([0.5004]))


def test_mutual_information():
    """Test the mutual information function.

    """
    input_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    output = F.mutual_information(input_)

    assert output.allclose(torch.tensor([0.5004]))

    # More confident prediction
    input_ = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    output = F.mutual_information(input_)

    assert output.allclose(torch.tensor([0.02162]), rtol=1e-4)


def test_variation_ratio():
    """Test the variation ratio function.

    """
    input_ = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]])
    output = F.variation_ratio(input_)

    assert output.allclose(torch.tensor([0.4]))
