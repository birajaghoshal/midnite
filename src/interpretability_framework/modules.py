"""Custom modules for MC dropout ensembles and uncertainty."""
import logging

import torch
from torch import Tensor
from torch.nn import Dropout
from torch.nn import functional
from torch.nn import Module

import interpretability_framework.functional as func

log = logging.getLogger(__name__)


class PredDropout(Dropout):
    """Layer for dropout at prediction time."""

    def forward(self, input_: Tensor) -> Tensor:
        """Performs dropout at training and at test time.

        Args:
            input_: the input layer

        Returns:
            the dropout forward pass

        """
        return functional.dropout(input_, p=self.p, inplace=self.inplace)


class _Ensemble(Module):
    """Module for ensemble models."""

    def __init__(self, inner: Module, sample_size: int = 20):
        """Create a network with probabilistic predictions from a variable inner model.

        Args:
            inner: the inner network block
            sample_size: the number of samples of the inner prediction to use
             in the forward pass at eval time

        """
        super(_Ensemble, self).__init__()

        if sample_size < 2:
            raise ValueError("At least two samples are necessary")

        if sample_size < 20:
            log.warning("Using low number of samples may give greatly skewed results")

        self.sample_size = sample_size
        self.inner = inner


class MeanEnsemble(_Ensemble):
    """Module for models that want the mean of their ensemble predictions."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for ensemble, calculating mean.

        Args:
            input_: the input data

        Returns:
            the mean of the inner network's foward pass for sample_size samples

        """
        pred = self.inner.forward(input_)

        if self.training:
            return pred
        else:
            for i in range(self.sample_size - 1):
                pred += self.inner.forward(input_)
            return pred.div(self.sample_size)


class PredictionEnsemble(_Ensemble):
    """Generic module for ensemble models."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for ensemble, returning all samples.

        Args:
            input_: the input data

        Returns:
            all prediction samples stacked in dim 0

        """
        # Calculate first prediction
        pred = self.inner.forward(input_)

        # In eval mode, stack ensemble predictions
        if not self.training:
            pred = torch.unsqueeze(pred, dim=0)
            for i in range(self.sample_size - 1):
                new_pred = torch.unsqueeze(self.inner.forward(input_), dim=0)
                pred = torch.cat((pred, new_pred))

        return pred


class PredictiveEntropy(Module):
    """Module to calculate the predictive entropy from an generic ensemble model."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for predictive entropy (also called max entropy),
         to measure total uncertainty.

        Args:
            input_: concatenated predictions for N classes and T samples,
             of shape (T, N)

        Returns:
            pred_entropy: the predictive entropy per class

        """

        return func.predictive_entropy(input_)


class MutualInformation(Module):
    """Module to calculate the mutual information from an generic ensemble model."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for mutual information, to measure epistemic uncertainty.

        Also called Bayesian Active Learning by Disagreement (BALD)

        Args:
            input_: concatenated predictions for N classes and T samples,
             of shape (T, N)

        Returns:
            the mutual information per class

        """
        return func.mutual_information(input_)


class VariationRatio(Module):
    """Module to calculate the variation ratio from an generic ensemble model."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for the variation ratio, to give a percentage
         of total predictive uncertainty.

        Args:
            input_: concatenated probabilistic predictions for N classes and T samples,
             of shape (T, N)

        Returns:
            the total variation ratio

        """
        return func.variation_ratio(input_)


class ConfidenceMeanPrediction(Module):
    """Module to conveniently calculate sampled mean and uncertainties."""

    def forward(self, input_: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """Forward pass to calculate sampled mean and different uncertainties.

        Args:
            input_: concatenated probabilistic predictions for N classes and T samples,
             of shape (T, N)

        Returns:
            mean prediction, predictive entropy, mutual information, variation ratio

        """
        return (
            input_.mean(dim=(0,)),
            func.predictive_entropy(input_),
            func.mutual_information(input_),
            func.variation_ratio(input_),
        )
