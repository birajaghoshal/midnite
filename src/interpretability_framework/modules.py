import logging

import torch
from torch.nn import Dropout
from torch.nn import functional as F
from torch.nn import Module

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
    def forward(self, ensemble_mean):
        """

        Args:
            ensemble_mean: mean of the MC ensemble of previous dropout layer

        Returns:
            pred_entropy: the entropy of the predicted distribution, the uncertainty of the entire distribution
        """

        log_pred = ensemble_mean * torch.log(ensemble_mean)
        pred_entropy = - log_pred

        return pred_entropy


class MutualInformation(PredictiveEntropy):
    def forward(self, input):
        """Forward pass for mutual information (also called BALD), to measure epistemic uncertainty.

        Args:
            input: predictions for n classes for T samples, of dimension (n, T).

        Returns:
            the mutual information for each class
        """
        pred_entropy = super(MutualInformation, self).forward(input)
        # TODO
        return pred_entropy

