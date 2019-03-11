import logging

import torch
from torch.nn import Dropout
from torch.nn import functional as F
from torch.nn import Module

import interpretability_framework.functional as Self_F

log = logging.getLogger(__name__)


class PredDropout(Dropout):
    def forward(self, input):
        """Performs dropout at training and at test time.

        Args:
            input: the input layer

        Returns:
            the dropout forward pass

        """
        return F.dropout(input, self.p, True, self.inplace)


class _Ensemble(Module):
    def __init__(self, inner: Module, sample_size):
        """Create a network with probabilistic predictions from a variable inner model.

        Args:
            inner: the inner network block
            sample_size: the number of samples of the inner prediction to use in the forward pass
        """
        super(_Ensemble, self).__init__()

        if sample_size < 2:
            raise ValueError("At least two samples are necessary")

        if sample_size < 20:
            log.warning("Using low number of samples may give greatly skewed results")

        self.sample_size = sample_size
        self.inner = inner


class MeanEnsemble(_Ensemble):
    def forward(self, input):
        """Forward pass for ensemble, calculating mean.

        Args:
            input: the input data

        Returns:
            the mean of the inner network's foward pass for sample_size samples
        """
        pred = self.inner.forward(input)

        if self.training:
            return pred
        else:
            for i in range(self.sample_size - 1):
                pred += self.inner.forward(input)
            return pred.div(self.sample_size)


class PredictionEnsemble(_Ensemble):
    def forward(self, input):
        """Forward pass for ensemble, returning all samples.

        Args:
            input: the input data

        Returns:
            all prediction samples stacked
        """
        # Calculate first prediction
        pred = self.inner.forward(input)

        # In eval mode, stack ensemble predictions
        if not self.training:
            # Retain dim of one prediction
            pred_dim = pred.dim()

            pred = torch.unsqueeze(pred, dim=pred_dim)
            for i in range(self.sample_size - 1):
                new_pred = torch.unsqueeze(self.inner.forward(input), dim=pred_dim)
                pred = torch.cat((pred, new_pred), dim=pred_dim)

        return pred


class PredictiveEntropy(Module):
    def forward(self, input):
        """Forward pass for predictive entropy (also called max entropy), to measure total uncertainty.

        Args:
            input: concatenated predictions for N classes and T samples, of shape (N, T)

        Returns:
            pred_entropy: the predictive entropy for each class
        """

        return Self_F.predictive_entropy(input)


class MutualInformation(Module):
    def forward(self, input):
        """Forward pass for mutual information (also called BALD), to measure epistemic uncertainty.

        Args:
            input: concatenated predictions for N classes and T samples, of shape (N, T)

        Returns:
            the mutual information for each class
        """
        return Self_F.mutual_information(input)
