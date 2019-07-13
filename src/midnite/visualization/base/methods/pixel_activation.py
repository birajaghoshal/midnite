"""Concrete implementations of pixel activation methods."""
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import tqdm
from numpy import random
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam

import midnite
from . import common
from ...base.interface import Activation
from ...base.interface import NeuronSelector
from ...base.interface import OutputRegularization
from ...base.transforms import TransformStep


class TVRegularization(OutputRegularization):
    """Total Variation norm.

    For more information, read https://arxiv.org/pdf/1412.0035v1.pdf.
    Loss is divided by number of elements to have consistent coefficients.

    """

    def __init__(self, coefficient: float = 1e2, beta: float = 2.0):
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

    def __init__(self, decay_factor=1e-7):
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
        inspection_net: Module,
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
            inspection_net: the list of adjacent layers to account for
            top_layer_selector: selector for the relevant activations
            lr: learning rate of ADAM optimizer
            iter_n: number of iterations
            opt_n: number of optimization steps per iteration
            init_size: initial width/height of visualization
            clamp: clamp image to [0, 1] space (natural image)
            transform: optional transformation after each step
            regularization: optional list of regularization terms

        """
        super().__init__([], inspection_net, top_layer_selector)
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
        opt_img = opt_img.unsqueeze(0).detach()

        # Optimizer for image
        optimizer = Adam([opt_img], lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.opt_n):
            opt_img.requires_grad_(True)
            # Reset gradient
            optimizer.zero_grad()

            # Do a forward pass
            out = self.black_box_net(opt_img).squeeze(0)

            # Calculate loss (mean of output of the last layer w.r.t to our mask)
            loss = -common.calculate_single_mean(
                out, self.top_layer_selector
            ).unsqueeze(0)

            for reg in self.output_regularizers:
                loss += reg.loss(opt_img.squeeze(0))

            # Backward pass
            loss.backward()
            # Update image
            optimizer.step()

        if self.clamp:
            opt_img = opt_img.clamp(min=0, max=1)
        return opt_img.squeeze(0).detach()

    def visualize(self, input_: Optional[Tensor] = None) -> Tensor:
        # Prepare layers/input
        self.black_box_net.to(midnite.get_device())

        if input_ is None:
            # Create uniform random starting image
            input_ = torch.from_numpy(
                np.uint8(random.uniform(150, 180, (3, self.init_size, self.init_size)))
                / float(255)
            ).float()
        else:
            input_ = input_.clone().detach()
        opt_img = input_.to(midnite.get_device())

        for n in tqdm.trange(self.iter_n):
            # Optimization step
            opt_img = self._opt_step(opt_img).detach()

            # Transformation step, if necessary
            if n + 1 < self.iter_n and self.transforms is not None:
                img = opt_img.permute((1, 2, 0)).cpu().numpy()
                img = self.transforms.transform(img)
                opt_img = (
                    torch.from_numpy(img).permute(2, 0, 1).to(midnite.get_device())
                )

        return opt_img
