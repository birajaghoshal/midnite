"""Acquisition functions and helpers.

Definitions from https://arxiv.org/pdf/1703.02910.pdf.
See Also https://arxiv.org/pdf/1802.10501.pdf, Appendix C for derivations
of MC approximations.

"""
import logging

import torch
from torch import Tensor
from torch.nn import functional

import midnite

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
    input_.to(midnite.get_device())
    return torch.mean(input_, dim=(-1,))


def predictive_entropy(input_: Tensor, log_clamp=1e-40, inplace=False) -> Tensor:
    """Calculates the predictive entropy over the samples in the input.

    Approximation: H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D)

    Args:
        log_clamp: min value for prediction mean, to avoid NaN errors.
         Default value: 1e-40
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, T, ...)
        inplace: whether to perform the operations in-place

    Returns: the predictive entropy per class,
     i.e. H[y|x,D] = - sum_y p(y|x,D) * log p(y|x,D), of shape (N, K, ...)

    """
    input_.to(midnite.get_device())
    # Min clamp necessary in case of log(0)
    _ensemble_mean = mean_prediction(input_)
    if inplace:
        _ensemble_mean.clamp_(log_clamp)
        return _ensemble_mean.mul_(_ensemble_mean.log()).mul_(-1.0)
    else:
        _ensemble_mean = _ensemble_mean.clamp(log_clamp)
        log = _ensemble_mean.log()
        res = _ensemble_mean.mul(log).mul(-1.0)
        return res


def mutual_information(input_: Tensor, log_clamp=1e-40, inplace=False) -> Tensor:
    """Calculates the mutual information over the samples in the input.

    Approximation: I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]]
        = H[y|x,D] + sum_t,y p(y|x,w_t) * log p(y|x,w_t)

    Args:
        log_clamp: min value for input mean, to avoid NaN errors.
         Default value: 1e-40
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, T, ...)
        inplace: whether to perform the operations in-place

    Returns: the mutual information, i.e. I[y,w|x,D] = H[y|x,D] - E[sum_y H[y|x,w]],
     of shape (N, K, ...)

    """
    input_.to(midnite.get_device())
    # Min clamp necessary in case of log(0)
    pred_entropy = predictive_entropy(input_, log_clamp, inplace)

    num_samples = input_.size(dim=input_.dim() - 1)
    if inplace:
        clamped_input = input_.clamp_(min=log_clamp)
        expected_entropy = (
            clamped_input.mul_(clamped_input.log()).sum(dim=-1).div_(num_samples)
        )

        return expected_entropy.add_(pred_entropy)
    else:
        clamped_input = input_.clamp(min=log_clamp)
        expected_entropy = (
            clamped_input.mul(clamped_input.log()).sum(dim=-1).div(num_samples)
        )

        return expected_entropy.add(pred_entropy)


def variation_ratio(input_: Tensor, inplace=False) -> Tensor:
    """Calculates the variation ratio over the samples in the input.

    Args:
        input_: concatenated predictions for K classes and T samples (minibatch size N),
         of shape (N, K, ..., T)
        inplace: whether to perform the operations in-place

    Returns: the variation ratio, i.e. 1 - max_y p(y|x,D), of shape (N, K, ...)

    """
    # shape (N, K, ...)
    input_.to(midnite.get_device())
    mean = mean_prediction(input_)

    prob_sum = mean.sum(dim=(1,))

    if not torch.allclose(prob_sum, torch.ones_like(prob_sum)):
        log.warning("variation ratio input not in probabilistic form, using softmax")
        mean = functional.softmax(mean, dim=1)

    # shape (N, ...)
    max_mean = torch.max(mean, dim=1)[0]
    if inplace:
        return max_mean.mul_(-1.0).add_(1.0)
    else:
        return max_mean.mul(-1.0).add(1.0)
