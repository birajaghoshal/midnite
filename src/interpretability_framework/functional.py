import torch
from torch import Tensor


"""Acquisition functions and helpers

Definitions from https://arxiv.org/pdf/1703.02910.pdf.
See Also https://arxiv.org/pdf/1802.10501.pdf, Appendix C for derivations of MC approximations.

"""


def sample_mean(input: Tensor) -> Tensor:
    """Gives the mean over sampled predictions.

    Approximation: p(y|x,D) = 1/T sum_t p(y|x,w_t)

    Args:
        input: concatenated predictions for N classes and T samples, of shape (T, N)

    Returns:
        the mean prediction for all samples, i.e. p(y|x,D)

    """
    return torch.mean(input, dim=(0,))


def predictive_entropy(input: Tensor, mean_min_clamp=1e-40) -> Tensor:
    """Calculates the predictive entropy over the samples in the input.

    Approximation: H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D)

    Args:
        mean_min_clamp: min value for prediction mean, to avoid NaN errors. Default value: 1e-40
        input: concatenated predictions for N classes and T samples, of shape (T, N)

    Returns: the predictive entropy per class, i.e. H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D)

    """
    _ensemble_mean = sample_mean(input).clamp(mean_min_clamp)

    # Min clamp necessary in case of log(0)
    return -_ensemble_mean * torch.log(_ensemble_mean)


def mutual_information(input_: Tensor, input_mean_clamp=1e-40) -> Tensor:
    """Calculates the mutual information over the samples in the input.

    Approximation: I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]]
        = H[y|x,D] + sum_t,y p(y|x,w_t) * log p(y|x,w_t)

    Args:
        input_mean_clamp: min value for input mean, to avoid NaN errors. Default value: 1e-40
        input_: concatenated predictions for N classes and T samples, of shape (T, N)

    Returns: the mutual information, i.e. I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]]

    """
    pred_entropy = predictive_entropy(input_)

    num_samples = input_.size(dim=0)
    input_ = input_.clamp(min=input_mean_clamp)

    # Min clamp necessary in case of log(0)
    expected_entropy = torch.div(
        torch.sum(input_ * torch.log(input_), dim=(0,)), num_samples
    )

    return pred_entropy + expected_entropy


def variation_ratio(input_: Tensor) -> Tensor:
    """Calculates the variation ratio over the samples in the input.

    Args:
        input_: concatenated predictions for N classes and T samples, of shape (T, N)

    Returns: the variation ratio, i.e. 1 - max_y p(y|x,D)

    """
    mean = sample_mean(input_)

    # Last dimensions should be classes
    max_mean = torch.max(mean, dim=(mean.dim() - 1))[0]

    return torch.add(-max_mean, 1.0)
