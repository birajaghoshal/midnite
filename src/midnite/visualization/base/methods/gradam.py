"""Concrete implementations of gradient attribution mapping methods."""
from torch import Tensor
from torch.nn import functional
from torch.nn import Module

import midnite
from ...base.interface import Attribution
from ...base.interface import LayerSplit
from ...base.interface import NeuronSelector
from .backpropagation import Backpropagation


class GradAM(Attribution):
    """Gradient attribution mapping.

    Scales the propagated input with the gradient of its inverse split.
    Interpretation: How much does neuron x of layer X (bottom layer selection)
    contribute to neuron y of layer Y (top layer selection)?

    """

    def __init__(
        self,
        inspection_net: Module,
        top_layer_selector: NeuronSelector,
        base_net: Module,
        bottom_layer_split: LayerSplit,
    ):
        """

        Args:
            inspection_net: part of the network from selected until output layer
            top_layer_selector: specifies selected neurons in the output layer
            base_net: part of the network from first layer up to selected
            bottom_layer_split: specifies the split of interest in the selected layer

        """
        super().__init__([], inspection_net, top_layer_selector, bottom_layer_split)
        self.base_net = base_net
        self.backpropagator = Backpropagation(
            inspection_net, self.top_layer_selector, self.bottom_layer_split.invert()
        )

    def visualize(self, input_: Tensor) -> Tensor:
        # Prepare layers/input
        self.base_net.to(midnite.get_device()).eval()
        input_ = input_.clone().detach().to(midnite.get_device())

        # forward pass through base layers
        intermediate = self.base_net(input_)

        # retrieve the alpha weight
        intermediate_back = self.backpropagator.visualize(intermediate)
        # get rid of batch dimension
        alpha = intermediate_back.detach().squeeze(dim=0)
        intermediate = intermediate.detach().squeeze(dim=0)
        # fill dimensions s.t. each split has dim (c, h, w)
        alpha = self.bottom_layer_split.invert().fill_dimensions(
            alpha, len(intermediate.size())
        )

        result = self.bottom_layer_split.get_mean(intermediate.mul_(alpha))

        return functional.relu(result)
