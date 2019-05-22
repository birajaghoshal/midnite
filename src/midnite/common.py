"""Common midnite modules."""
from torch.nn import Module


class Flatten(Module):
    """One layer module that flattens its input, except for minibatch dimension."""

    def forward(self, x):
        if len(x.size()) == 1:
            return x
        else:
            return x.view(x.size(0), -1)
