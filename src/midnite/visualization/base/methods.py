"""Concrete implementations of visualization methods."""
from contextlib import contextmanager
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

import midnite
from ..base.interface import Activation
from ..base.interface import Attribution
from ..base.interface import LayerSplit
from ..base.interface import NeuronSelector
from ..base.interface import OutputRegularization
from ..base.transforms import TransformStep


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


def _calculate_single_mean(out, select: NeuronSelector) -> Tensor:
    """Calculates the selected mean for a minibatch with one element, retaining grads.

    Args:
        out: network output of shape (1, c, h, w)
        select: selector

    Returns:
        the calculated mean as scalar tensor

    """
    out = out.squeeze(dim=0)
    mask = select.get_mask(list(out.size()))
    return (out * mask).sum() / mask.sum()


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
        self.black_box_net.to(midnite.get_device()).eval()

        # forward pass
        out = self.black_box_net(input_)

        # retrieve mean score of top layer selector
        score = _calculate_single_mean(out, self.top_layer_selector)

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


class Occlusion(Attribution):
    """Calculates attribution by occluding parts of the image and comparing outputs"""

    def __init__(
        self,
        net: Module,
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
        chunk_size: List[int],
        pixel_stride: List[int],
        norm=2,
    ):
        """
        Args:
            net: part of the network to analyze
            top_layer_selector: specifies elected neurons in the output layer
            bottom_layer_split: neuron split of the bottom layer (i.e., input image)
            chunk_size: chunk size in number of strides
            norm: the norm to be computed
        """
        super().__init__([], net, top_layer_selector, bottom_layer_split)
        self.chunk_size = chunk_size
        self.pixel_stride = pixel_stride
        if norm < 1:
            raise ValueError("Must be valid distance norm")
        self.norm = norm

    @classmethod
    def _smear_matrix(cls, size: int, smear: int) -> Tensor:
        # Identity
        matrix = torch.eye(size, device=midnite.get_device())
        # Add smear diagonals
        for i in range(1, smear):
            matrix[i:, :-i] += torch.eye(size - i)
        return matrix.clamp_(max=1)

    def _chunk_matrixes(self, input_size: List[int]) -> List[Tensor]:
        """Build the chunk matrixes"""
        if not len(input_size) == len(self.chunk_size):
            raise ValueError("Chunks size must be given for every input dimension")
        return list(
            map(
                lambda sizes: self._smear_matrix(sizes[0] + sizes[1], sizes[1]),
                zip(input_size, self.chunk_size),
            )
        )

    def _chunk_mask(self, pixel_mask: Tensor, chunk_matrixes: List[Tensor]) -> Tensor:
        sizes = list(zip(pixel_mask.size(), self.chunk_size))
        mask_size = tuple(map(lambda s: s[0] + s[1], sizes))
        # Create mask with initial pixel mask
        mask = torch.zeros(mask_size, device=midnite.get_device())
        mask[tuple(map(lambda s: slice(None, s), pixel_mask.size()))] = pixel_mask
        # build mask
        for i, matrix in enumerate(chunk_matrixes):
            mask = (matrix @ mask.transpose(1, i)).transpose(1, i)
        # Get relevant mask area
        mask_area = tuple(map(lambda s: slice(s[1] // 2, s[0] + (s[1] // 2)), sizes))
        # Normalize indexes that were selected multiple times
        return mask[mask_area].clamp_(max=1)

    def _remove_chunk(self, input_: Tensor, chunk_mask: Tensor) -> Tensor:
        # Upsample to input image dimensions (use volumetric to get 3d upsample)
        mask = functional.interpolate(
            chunk_mask.unsqueeze(1), tuple(input_.squeeze(0).size())
        ).squeeze(1)
        # Fill chunk with grey
        return input_.mul_(1 - mask).add_(mask * 0.5)

    def visualize(self, input_: Tensor) -> Tensor:
        # Preparation
        input_ = input_.clone().detach().to(midnite.get_device())
        self.black_box_net.to(midnite.get_device()).eval()

        # Make reference prediction
        pred = self.black_box_net(input_).squeeze(0)
        # Store top layer mask for the future
        pred_mask = self.top_layer_selector.get_mask(list(pred.size()))
        pred_mask.div_(pred_mask.sum())
        pred.mul_(pred_mask)

        # Dimensions of subsampled image
        subsample_dims = list(
            map(
                lambda t: t[0] // t[1], zip(input_.squeeze(0).size(), self.pixel_stride)
            )
        )

        chunk_matrixes = self._chunk_matrixes(subsample_dims)

        # For every pixel in subsampled image, measure difference to true prediction
        result_img = torch.zeros(*subsample_dims, device=midnite.get_device())
        for pixel in tqdm.tqdm(self.bottom_layer_split.get_split(subsample_dims)):
            mask = self._chunk_mask(pixel, chunk_matrixes)
            img = self._remove_chunk(input_.clone(), mask.unsqueeze(0))
            out = self.black_box_net(img).squeeze(0).mul_(pred_mask)
            result_img.add_(mask.squeeze(0).mul_(out.dist(pred, self.norm).item()))

        return self.bottom_layer_split.get_mean(result_img)


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
            loss = -_calculate_single_mean(out, self.top_layer_selector).unsqueeze(0)

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
        self.black_box_net.to(midnite.get_device()).eval()

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
