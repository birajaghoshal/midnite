"""Concrete implementations of visualization methods."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from numpy import random
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential

from vinsight import get_device
from vinsight.visualization import Activation
from vinsight.visualization import Attribution
from vinsight.visualization import ChannelSplit
from vinsight.visualization import LayerSplit
from vinsight.visualization import NeuronSelector
from vinsight.visualization import NeuronSplit
from vinsight.visualization import SpatialSplit
from vinsight.visualization.transforms import TransformStep


class OutputRegularization(ABC):
    """Base class for regularizations on an output term."""

    def __init__(self, coefficient: float = 0.1):
        """

        Args:
            coefficient: how much regularization to apply

        """
        self.coefficient = coefficient

    @abstractmethod
    def loss(self, out: Tensor) -> Tensor:
        """Calculates the loss for an output.

         Args:
             out: the tensor to calculate the loss for, of shape
              (channels, height,width)

         Returns:
             a tensor representing the loss

         """
        raise NotImplementedError()


class TVRegularization(OutputRegularization):
    """Total Variation norm.

    For more information, read https://arxiv.org/pdf/1412.0035v1.pdf.
    Loss is divided by number of elements to have consistent coefficients.

    """

    def __init__(self, coefficient: float = 0.1, beta: float = 2.0):
        """

        Args:
            coefficient: how much regularization to apply
            beta: beta-parameter of total variation

        """
        super().__init__(coefficient)
        self.beta = beta

    def loss(self, out: Tensor) -> Tensor:
        x_dist = (out[:, :-1, 1:] - out[:, :-1, :-1]).pow_(2)
        y_dist = (out[:, 1:, :-1] - out[:, :-1, :-1]).pow_(2)
        loss_sum = x_dist.add_(y_dist).pow_(self.beta / 2.0).sum()

        return loss_sum.div_(out.size(0) * out.size(1) * out.size(2)).mul_(
            self.coefficient
        )


class WeightDecay:
    """Weight decay regularization."""

    def __init__(self, decay_factor):
        """

        Args:
            decay_factor: decay factor for optimizer

        """
        self.decay_factor = decay_factor


Regularization = Union[OutputRegularization, WeightDecay]


class PixelActivation(Activation):
    """Activation optimization (in Pixel space)."""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        lr: float = 1e-2,
        iter_n: int = 12,
        opt_n: int = 20,
        init_size: int = 250,
        clamp: bool = True,
        transform: Optional[TransformStep] = None,
        regularization: List[Regularization] = None,
    ):
        """
        Args:
            layers: the list of adjacent layers to account for
            top_layer_selector: selector for the relevant activations
            lr: learning rate of ADAM optimizer
            iter_n: number of iterations
            opt_n: number of optimization steps per iteration
            init_size: initial width/height of visualization
            clamp: clamp image to [0, 1] space (natural image)
            transform: optional transformation after each step
            regularization: optional list of regularization terms

        """
        super().__init__(layers, top_layer_selector)
        self.lr = lr
        if iter_n < 1:
            raise ValueError("Must have at least one iteration")
        self.iter_n = iter_n
        if opt_n < 1:
            raise ValueError("Must have at least one optimization step per iteration")
        self.opt_n = opt_n
        if init_size < 1:
            raise ValueError("Initial size has to be at least one")
        self.init_size = init_size
        self.clamp = clamp
        self.transforms = transform

        if regularization is None:
            regularization = []

        # Filter out weight decay
        weight_decay = list(
            reg for reg in regularization if isinstance(reg, WeightDecay)
        )
        if len(weight_decay) > 1:
            raise ValueError("Can at most have one weight decay regularizer")
        elif len(weight_decay) == 1:
            self.weight_decay = weight_decay[0].decay_factor
        else:
            self.weight_decay = 0.0
        # Add all output regularizers
        self.output_regularizers = list(
            reg for reg in regularization if isinstance(reg, OutputRegularization)
        )

    def _opt_step(self, opt_img: Tensor) -> Tensor:
        """Performs a single optimization step.

        Args:
            opt_img: the tensor to optimize, of shape (c, h, w)

        Returns:
            the optimized image as tensor of shape (c, h, w)

        """
        # Add minibatch dim and clean up gradient
        opt_img = opt_img.unsqueeze(dim=0).detach().requires_grad_(True)

        # Optimizer for image
        optimizer = torch.optim.Adam(
            [opt_img], lr=self.lr, weight_decay=self.weight_decay
        )

        for _ in range(self.opt_n):
            # Reset gradient
            optimizer.zero_grad()

            # Do a forward pass
            out = opt_img
            for layer in self.layers:
                out = layer(out)

            out = out.squeeze(dim=0)

            # Calculate loss (mean of output of the last layer w.r.t to our mask)
            loss = -((self.top_layer_selector.get_mask(out.size()) * out).mean())

            loss = loss.unsqueeze(dim=0)

            for reg in self.output_regularizers:
                loss += reg.loss(opt_img.squeeze(dim=0))

            # Backward pass
            loss.backward()
            # Update image
            optimizer.step()

        if self.clamp:
            opt_img = opt_img.clamp(min=0, max=1)
        return opt_img.squeeze(dim=0).detach()

    def visualize(self, input_: Optional[Tensor] = None) -> Tensor:
        # Move all layers to current device and set to eval mode
        for layer in self.layers:
            layer.to(get_device()).eval()

        if input_ is None:
            # Create uniform random starting image
            input_ = torch.from_numpy(
                np.uint8(random.uniform(150, 180, (self.init_size, self.init_size, 3)))
                / float(255)
            ).float()
        opt_img = input_.permute((2, 0, 1)).to(get_device())

        for n in range(self.iter_n):
            # Optimization step
            opt_img = self._opt_step(opt_img)

            # Transformation step, if necessary
            if n + 1 < self.iter_n and self.transforms is not None:
                img = opt_img.detach().permute((1, 2, 0)).cpu().numpy()
                img = self.transforms.transform(img)
                opt_img = torch.from_numpy(img).to(get_device()).permute(2, 0, 1)

        return opt_img.detach().permute((1, 2, 0))


class SaliencyMap(Attribution):
    """computes a saliency map of a bottom layer selection wrt the top layer selection.
        How much does neuron x of layer X (bottom layer selection)
        contribute to neuron y of layer Y (top layer selection)?
    """

    def __init__(
        self,
        inspection_layers: List[Module],
        top_layer_selector: NeuronSelector,
        base_layers: List[Module],
        bottom_layer_split: LayerSplit,
    ):
        """
        Args:
            inspection_layers: the list of layers from selected layer until output layer (model split)
            top_layer_selector: specifies the split of interest in the output layer
            base_layers: list of layers from first layer of the model up to the selected layer
            bottom_layer_split: specifies the split of interest in the selected layer

        """
        super().__init__(
            inspection_layers, top_layer_selector, base_layers, bottom_layer_split
        )

    def _forward_pass(self, input_tensor) -> Tuple[Tensor, Tensor]:
        """ Computes a forward pass through the network.
        Args:
            input_tensor: input image
        Returns:
             features: intermediate output of the network. pass through the base_layers
             out: output of the pass through the whole network. pass through base_layers and inspection_layers
        """
        # features of the selected layer
        features = Sequential(*self.base_layers)(input_tensor)
        # features of the output layer
        out = Sequential(*self.inspection_layers)(features).squeeze()

        return features, out

    def _generate_gradients(self, out: Tensor, backprop_variable: Tensor) -> Tensor:
        """ Computes the gradients of the base layer output w.r.t. the inspection layer output.

        Args:
            out: output of the forward pass
            backprop_variable: the Tensor for which the gradients should be computed
        Returns:
             features: result of the forward pass through base_layers
             gradients: gradients of features with respect to the model output
        """

        # select the output element, for which gradient should be computed
        out_masked = self.top_layer_selector.get_mask(out.size()) * out

        # compute a single value from the mask, reduce dimensions with mean
        score = out_masked.mean(-1).mean(-1).mean(-1)

        # computes and returns the sum of gradients of the output layer score
        # w.r.t. the features of the selected layer
        gradients = torch.autograd.grad(score, backprop_variable)[0][0]

        return gradients

    def guided_backprop(self, input_tensor, from_input=True) -> Tensor:
        """ Computes the positive guided gradients of an input tensor.

        Args:
            input_tensor: input image
            from_input: specifies if gradients should be computed from either the input image
                or intermediate features of the selected layer.
        Returns:
             a tensor of positive gradients (activations)
        """

        input_tensor.to(get_device())
        input_tensor.requires_grad_(True)

        features, out = self._forward_pass(input_tensor)

        if from_input:
            gradients = self._generate_gradients(out, input_tensor)
        else:
            gradients = self._generate_gradients(out, features)

        positive_grads = (
            torch.max(gradients, torch.zeros_like(gradients).to(get_device()))
            .cpu()
            .detach()
        )

        return positive_grads

    def visualize(self, input_tensor: Tensor) -> Tensor:
        """generates a saliency tensor for an input image at the selected intermediate layer
        Args:
            input_tensor: input image as torch tensor

        """
        # Move all layers to current device
        for layer in self.layers + self.base_layers:
            layer.to(get_device())
        input_tensor.to(get_device())

        features, out = self._forward_pass(input_tensor)
        grads = self._generate_gradients(out, features)

        _, N, H, W = features.size()

        avg_pooling_weight = self.bottom_layer_split.get_mean(grads)

        if isinstance(self.bottom_layer_split, SpatialSplit):
            saliency = torch.matmul(avg_pooling_weight, features.view(N, H * W))
            saliency = saliency.view(H, W)
        elif isinstance(self.bottom_layer_split, ChannelSplit):
            saliency = torch.matmul(
                features.view(N, H * W), avg_pooling_weight.view(H * W)
            )
        elif isinstance(self.bottom_layer_split, NeuronSplit):
            saliency = features.squeeze()
        else:
            raise ValueError("not a valid split class for bottom_layer_split")

        sal_map = torch.max(
            saliency, torch.zeros_like(saliency).to(get_device())
        ).detach()

        return sal_map
