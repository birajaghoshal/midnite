import logging

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


class EnsembleMean(Module):
    def __init__(self, inner: Module, sample_size):
        """Create a network with probabilistic predictions from a variable inner model.

        Args:
            inner: the inner network block
            sample_size: the number of samples of the inner prediction to use in the forward pass
        """
        super(EnsembleMean, self).__init__()

        if sample_size < 2:
            raise ValueError("At least two samples are necessary")

        if sample_size < 20:
            log.warning("Using low number of samples my give greatly skewed results")

        self.sample_size = sample_size
        self.inner = inner

    def forward(self, input):
        """Forward pass for ensemble, calculating mean.

        Args:
            input: the input data

        Returns:
            the mean of the inner network's foward pass for sample_size samples
        """
        if self.training:
            return self.inner.forward(input)
        else:
            pred = self.inner.forward(input)
            for i in range(self.sample_size - 1):
                pred += self.inner.forward(input)
            return pred.div(self.sample_size)
