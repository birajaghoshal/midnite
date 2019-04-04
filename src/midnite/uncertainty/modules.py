"""Custom modules for MC dropout ensembles and uncertainty."""
import logging
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import AlphaDropout
from torch.nn import Dropout
from torch.nn import Dropout2d
from torch.nn import Dropout3d
from torch.nn import FeatureAlphaDropout
from torch.nn import functional
from torch.nn import Module
from tqdm import tqdm

import midnite.uncertainty.functional as func
from midnite import get_device

log = logging.getLogger(__name__)


class PredictionDropout(Dropout):
    """Layer for dropout at prediction time."""

    def forward(self, input_: Tensor) -> Tensor:
        """Performs dropout at training and at test time.

        Args:
            input_: the input data

        Returns:
            the dropout forward pass

        """
        return functional.dropout(input_, p=self.p, inplace=self.inplace)


def _is_dropout(layer: Module) -> bool:
    return (
        isinstance(layer, Dropout)
        or isinstance(layer, Dropout2d)
        or isinstance(layer, Dropout3d)
        or isinstance(layer, AlphaDropout)
        or isinstance(layer, FeatureAlphaDropout)
    )


class _Ensemble(Module):
    """Module for ensemble models."""

    def __init__(
        self, inner: Module, sample_size: int = 20, dropout_train: bool = True
    ):
        """Create a network with probabilistic predictions from a variable inner model.

        Args:
            inner: the inner network block
            sample_size: the number of samples of the inner prediction to use
             in the forward pass at eval time
            dropout_train: flag whether to use dropout layers in eval mode

        """
        super(_Ensemble, self).__init__()
        self.to(get_device())
        inner.to(get_device())

        if sample_size < 2:
            raise ValueError("At least two samples are necessary")

        if sample_size < 20:
            log.warning("Using low number of samples may give greatly skewed results")

        self.sample_size = sample_size
        self.inner = inner

        # Set dropout to train mode
        self.dropout_eval = dropout_train
        if dropout_train:
            for module in inner.modules():
                if _is_dropout(module):
                    module.train()

    def train(self, mode=True):
        """Keep dropout modules in train mode, if necessary."""
        self.training = mode

        # Set everything to proper mode
        for module in self.children():
            module.train(mode)

        # Set dropout modules to train.
        if not mode and self.dropout_eval:
            for module in filter(_is_dropout, self.inner.modules()):
                module.train()

        return self


class MeanEnsemble(_Ensemble):
    """Module for models that want the mean of their ensemble predictions."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for ensemble, calculating mean.

        Args:
            input_: the input data

        Returns:
            the mean of the inner network's foward pass for sample_size samples

        """
        input_.to(get_device())

        if self.training:
            return self.inner.forward(input_)
        else:
            with torch.no_grad():
                with tqdm(
                    range(self.sample_size - 1), total=self.sample_size
                ) as sample_range:
                    sample_range.update()
                    pred = self.inner.forward(input_)

                    # Calc remaining forward pass
                    for _ in sample_range:
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
        input_.to(get_device())

        # In eval mode, stack ensemble predictions
        if self.training:
            return self.inner.forward(input_)
        else:
            with torch.no_grad():
                # Disable all gradients
                for param in self.parameters():
                    param.requires_grad = False
                input_.requires_grad = False

                # Track progress
                with tqdm(
                    range(self.sample_size - 1), total=self.sample_size
                ) as sample_range:
                    sample_range.update()

                    pred = self.inner.forward(input_)
                    pred_shape = pred.size()

                    # Allocate result tensor
                    result = torch.zeros(
                        (*pred_shape, self.sample_size), device=get_device()
                    )
                    # Store results in their slice
                    result.select(len(pred_shape), 0).copy_(pred)

                    for i in sample_range:
                        pred = self.inner.forward(input_)
                        result.select(len(pred_shape), i).copy_(pred)

                    return result


class PredictiveEntropy(Module):
    """Module to calculate the predictive entropy from an generic ensemble model."""

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass for predictive entropy (also called max entropy),
         to measure total uncertainty.

        Args:
            input_: concatenated predictions for K classes and T samples (minibatch size N),
             of shape (N, K, ..., T)

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
            input_: concatenated predictions for K classes and T samples (minibatch size N),
             of shape (N, K, ..., T)

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
            input_: concatenated predictions for K classes and T samples (minibatch size N),
             of shape (N, K, ..., T)

        Returns:
            the total variation ratio

        """
        return func.variation_ratio(input_)


class PredictionAndUncertainties(Module):
    """Module to conveniently calculate sampled mean and uncertainties."""

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass to calculate sampled mean and different uncertainties.

        Args:
            input_: concatenated predictions for K classes and T samples (minibatch size N),
             of shape (N, K, ..., T)

        Returns:
            mean prediction, predictive entropy, mutual information

        """
        input_.to(get_device())
        return (
            input_.mean(dim=(input_.dim() - 1,)),
            func.predictive_entropy(input_),
            func.mutual_information(input_),
        )
