"""Concrete implementations of backpropagation methods."""
from contextlib import contextmanager
from typing import List
from typing import Optional

from torch import Tensor
from torch.nn import functional
from torch.nn import Module
from torch.nn import Sequential

import midnite
from . import common
from ...base.interface import Attribution
from ...base.interface import LayerSplit
from ...base.interface import NeuronSelector


class Backpropagation(Attribution):
    """Propagates gradients back for a specific split."""

    def __init__(
        self,
        net: Module,
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
    ):
        """
        Args:
            net: the (part of the) network to backpropagate over
            top_layer_selector: relevant output neurons in the last layer
            bottom_layer_split: neuron split of the bottom layer

        """
        super().__init__([], net, top_layer_selector, bottom_layer_split)

    def visualize(self, input_: Tensor) -> Tensor:
        # Prepare layers/input
        input_ = input_.clone().detach().to(midnite.get_device()).requires_grad_()
        self.black_box_net.to(midnite.get_device())

        # forward pass
        out = self.black_box_net(input_)

        # retrieve mean score of top layer selector
        score = common._calculate_single_mean(out, self.top_layer_selector)

        # backward pass
        score.backward()

        # mean over bottom layer split dimension
        return self.bottom_layer_split.get_mean(input_.grad.detach().squeeze(dim=0))


class GuidedBackpropagation(Attribution):
    """Calculates guided gradients for a specific split."""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
    ):
        """
        Args:
            layers: the adjacent layers to perform guided gradcam on
            top_layer_selector: relevant output neurons in the last layer
            bottom_layer_split: neuron split of the bottom layer
        """
        super().__init__(layers, None, top_layer_selector, bottom_layer_split)
        self.backpropagator = Backpropagation(
            Sequential(*layers), top_layer_selector, bottom_layer_split
        )

    @staticmethod
    def _relu_grad(grad_iput: Optional[Tensor]):
        return None if grad_iput is None else functional.relu(grad_iput)

    @staticmethod
    def _gradient_relu_hook(module, grad_input, grad_output: Tensor):
        return tuple(map(GuidedBackpropagation._relu_grad, grad_input))

    @contextmanager
    def _gradient_hook_context(self):
        hooks = list(
            map(
                lambda layer: layer.register_backward_hook(self._gradient_relu_hook),
                self.white_box_layers,
            )
        )
        yield
        for hook in hooks:
            hook.remove()

    def visualize(self, input_: Tensor) -> Tensor:
        with self._gradient_hook_context():
            grad = self.backpropagator.visualize(input_)
            # Since only grad_input can be rectified via hook, apply relu for last layer
            return functional.relu(grad)
