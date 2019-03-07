from torch.nn import functional as F
from torch.nn import Module


class PredDropout(Module):
    """Performs dropout at training and at test time.

    Args:
        input: Input layer
    """

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class EnsembleMean(Module):
    def __init__(self, inner: Module, sample_size):
        """Create a network with probabilistic predictions from a variable inner model.

        Args:
            inner: the inner network block
            sample_size: the number of samples of the inner prediction to use in the forward pass
        """
        super(EnsembleMean, self).__init__()
        self.sample_size = sample_size
        self.inner = inner

    def forward(self, input):
        """Forward pass for ensemble, calculating mean.

        Args:
            input: the input data

        Returns:

        """
        if self.training:
            return self.inner.forward(input)
        else:
            pred = self.inner.forward(input)
            for i in range(self.sample_size - 1):
                pred += self.inner.forward(input)
            return pred / self.sample_size
