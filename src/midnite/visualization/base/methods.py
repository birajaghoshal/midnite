"""Concrete implementations of visualization methods."""
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import tqdm
from numpy import random
from torch import Tensor
from torch.nn import functional
from torch.nn import Module
from torch.nn import Sequential
from torch.optim import Adam

from midnite import get_device
from midnite.visualization.base.interface import Activation
from midnite.visualization.base.interface import Attribution
from midnite.visualization.base.interface import LayerSplit
from midnite.visualization.base.interface import NeuronSelector
from midnite.visualization.base.interface import OutputRegularization
from midnite.visualization.base.transforms import TransformStep


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
        opt_img = opt_img.unsqueeze(dim=0).detach()

        # Optimizer for image
        optimizer = Adam([opt_img], lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.opt_n):
            opt_img.requires_grad_(True)
            # Reset gradient
            optimizer.zero_grad()

            # Do a forward pass
            out = self.net(opt_img).squeeze(dim=0)

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
        # Prepare layers/input
        self.net.to(get_device()).eval()

        if input_ is None:
            # Create uniform random starting image
            input_ = torch.from_numpy(
                np.uint8(random.uniform(150, 180, (self.init_size, self.init_size, 3)))
                / float(255)
            ).float()
        else:
            input_ = input_.clone().detach()
        opt_img = input_.permute((2, 0, 1)).to(get_device())

        for n in tqdm.trange(self.iter_n):
            # Optimization step
            opt_img = self._opt_step(opt_img).detach()

            # Transformation step, if necessary
            if n + 1 < self.iter_n and self.transforms is not None:
                img = opt_img.permute((1, 2, 0)).cpu().numpy()
                img = self.transforms.transform(img)
                opt_img = torch.from_numpy(img).to(get_device()).permute(2, 0, 1)

        return opt_img.permute((1, 2, 0))


class GuidedBackpropagation(Attribution):
    """Calculates (guided) gradients for a specific split."""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
    ):
        super().__init__(layers, top_layer_selector, bottom_layer_split)

    def select_top_layer_score(self, out: Tensor) -> Tensor:
        """Computes the mean of the top layer selection.

        Args:
            out: output of the forward pass
        Returns:
            score: target value (single value tensor) for gradient computation

        """
        # ignore batch dimension for masking
        out = out.squeeze(dim=0)

        # select the output element, for which gradient should be computed
        out_masked = self.top_layer_selector.get_mask(out.size()) * out
        # take mean over all dimensions
        score = out_masked.mean()

        return score

    def visualize(self, input_tensor: Tensor) -> Tensor:
        # Prepare layers/input
        input_tensor.detach().to(get_device()).retain_grad()
        self.net.to(get_device()).eval()

        # forward pass
        out = self.net(input_tensor)

        # retrieve mean score of top layer selector
        score = self.select_top_layer_score(out)

        # gradients of input w.r.t the top level score yields partial derivatives
        gradients = torch.autograd.grad(score, input_tensor)[0]

        # get only positive gradients
        positive_grads = functional.relu(gradients.detach())

        # mean over bottom layer split dimension
        return self.bottom_layer_split.get_mean(positive_grads.squeeze(dim=0))


class GradAM(Attribution):
    """Gradient attribution mapping.

    Scales the propagated input with the gradient of its inverse split.
    Interpretation: How much does neuron x of layer X (bottom layer selection)
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
            inspection_layers: the list of layers from selected layer until output layer
            top_layer_selector: specifies the split of interest in the output layer
            base_layers: list of layers from first layer of the model up to the selected
            bottom_layer_split: specifies the split of interest in the selected layer

        """
        super().__init__(inspection_layers, top_layer_selector, bottom_layer_split)
        self.base_net = Sequential(*base_layers)
        self.backpropagator = GuidedBackpropagation(
            inspection_layers, self.top_layer_selector, self.bottom_layer_split.invert()
        )

    def visualize(self, input_tensor: Tensor) -> Tensor:
        # Prepare layers/input
        self.net.to(get_device()).eval()
        self.base_net.to(get_device()).eval()
        input_tensor.detach().to(get_device())

        # forward pass through base layers
        intermediate = self.base_net(input_tensor)

        # retrieve the alpha weight
        intermediate_back = self.backpropagator.visualize(intermediate)
        # get rid of batch dimension
        alpha = intermediate_back.detach().squeeze(dim=0)
        # fill dimensions s.t. each split has dim (c, h, w)
        alpha = self.bottom_layer_split.invert().fill_dimensions(alpha)

        result = self.bottom_layer_split.get_mean(
            intermediate.detach().squeeze(dim=0).mul_(alpha)
        )

        return functional.relu(result)
