"""Acquisition functions and helpers

Definitions from https://arxiv.org/pdf/1703.02910.pdf.
See Also https://arxiv.org/pdf/1802.10501.pdf, Appendix C for derivations
of MC approximations.

"""
import logging

import torch
from torch import Tensor
from torch.nn import functional

log = logging.getLogger(__name__)


def mean_prediction(input_: Tensor) -> Tensor:
    """Gives the mean over sampled predictions.

    Approximation: p(y|x,D) = 1/T sum_t p(y|x,w_t)

    Args:
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, ..., T)

    Returns:
        the mean prediction for all samples, i.e. p(y|x,D), of shape (N, K, ...)

    """
    return torch.mean(input_, dim=(input_.dim() - 1,))


def predictive_entropy(input_: Tensor, mean_min_clamp=1e-40) -> Tensor:
    """Calculates the predictive entropy over the samples in the input.

    Approximation: H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D)

    Args:
        mean_min_clamp: min value for prediction mean, to avoid NaN errors.
         Default value: 1e-40
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, T, ...)

    Returns: the predictive entropy per class,
     i.e. H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D), of shape (N, K, ...)

    """
    _ensemble_mean = mean_prediction(input_).clamp(mean_min_clamp)

    # Min clamp necessary in case of log(0)
    return -_ensemble_mean * torch.log(_ensemble_mean)


def mutual_information(input_: Tensor, input_mean_clamp=1e-40) -> Tensor:
    """Calculates the mutual information over the samples in the input.

    Approximation: I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]]
        = H[y|x,D] + sum_t,y p(y|x,w_t) * log p(y|x,w_t)

    Args:
        input_mean_clamp: min value for input mean, to avoid NaN errors.
         Default value: 1e-40
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, T, ...)

    Returns: the mutual information, i.e. I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]],
     of shape (N, K, ...)

    """
    pred_entropy = predictive_entropy(input_)

    num_samples = input_.size(dim=input_.dim() - 1)
    input_ = input_.clamp(min=input_mean_clamp)

    # Min clamp necessary in case of log(0)
    expected_entropy = torch.div(
        torch.sum(input_ * torch.log(input_), dim=(input_.dim() - 1,)), num_samples
    )

    return pred_entropy + expected_entropy


def variation_ratio(input_: Tensor) -> Tensor:
    """Calculates the variation ratio over the samples in the input.

    Args:
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, ..., T)

    Returns: the variation ratio, i.e. 1 - max_y p(y|x,D), of shape (N, K, ...)

    """
    # shape (N, K, ...)
    mean = mean_prediction(input_)
    # extract (N, ...)
    prob_sum_sizes = list(mean.size())
    del prob_sum_sizes[1]

    if not mean.sum(dim=(1,)).allclose(torch.ones(tuple(prob_sum_sizes))):
        log.warning("variation ratio input not in probabilistic form, using softmax")
        mean = functional.softmax(mean, dim=1)

    # shape (N, ...)
    max_mean = torch.max(mean, dim=1)[0]

    return torch.add(-max_mean, 1.0)
