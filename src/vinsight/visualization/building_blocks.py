"""POC for the visualization interface"""
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Size
from torch import Tensor
from torch.nn import Module
from torchvision import models


class LayerSplit(ABC):
    @abstractmethod
    def get_split(self, size: Size) -> List[Tensor]:
        """Returns a split, i.e. all masks"""
        pass

    @abstractmethod
    def get_mask(self, index: List[int], size: Size) -> Tensor:
        """Returns a single mask for the indexed split"""
        pass


class NeuronSplit(LayerSplit):
    """Split a layer neuron-wise"""

    def get_split(self, size: Size) -> List[Tensor]:
        if not len(size) == 3:
            raise ValueError("layer has to have 3 dimensions")
        indexes = product(*map(range, size))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Size) -> Tensor:
        if not len(index) == 3 or not len(size) == 3:
            raise ValueError("Index and layer have to have 3 dimensions")
        mask = torch.zeros(*size)
        mask[tuple(index)] = 1
        return mask


class SpatialSplit(LayerSplit):
    """Split a layer by spatial position"""

    def get_split(self, size: Size) -> List[Tensor]:
        raise NotImplementedError()

    def get_mask(self, index: List[int], size: Size) -> Tensor:
        if not len(index) == 2:
            raise ValueError("Spatial index need two dimensions. Got: ", index)
        if not len(size) == 3:
            raise ValueError("Layer has to have 3 dimensions. Got: ", size)
        mask = torch.zeros(*size)
        mask[:, index] = 1
        return mask


class ChannelSplit(LayerSplit):
    def get_split(self, size: Size) -> List[Tensor]:
        raise NotImplementedError()

    def get_mask(self, index: List[int], size: Size) -> Tensor:
        if not len(index) == 1:
            raise ValueError("Channel index needs one dimension. Got: ", index)
        if not len(size) == 3:
            raise ValueError("Layer has to have 3 dimensions. Got: ", size)
        mask = torch.zeros(*size)
        mask[index] = 1
        return mask


# class GroupSplit(LayerSplit):


class NeuronSelector:
    def __init__(self, layer_split: LayerSplit, element: List[int]):
        self.layer_split = layer_split
        self.element = element

    def get_mask(self, size: Size) -> Tensor:
        return self.layer_split.get_mask(self.element, size)


class Attribution(ABC):
    def __init__(self, layers: List[Module], bottom_layer_split: LayerSplit):
        self.layers = layers
        self.bottom_layer_split = bottom_layer_split

    @abstractmethod
    def visualize(self, input_: Tensor) -> Tensor:
        pass


class Activation(ABC):
    def __init__(self, layers: List[Module], top_layer_selector: NeuronSelector):
        self.layers = layers
        self.top_layer_selector = top_layer_selector


class PixelActivationOpt(Activation):
    """Simple Activation Optimization, without regularization"""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        lr=1e-2,
        opt_steps=200,
        length=200,
    ):
        super().__init__(layers, top_layer_selector)
        self.lr = lr
        self.opt_steps = opt_steps
        self.length = length

    def visualize(self) -> Tensor:
        # Starting image - tune uniform distribution weights if necessary
        rand_img = np.uint8(
            np.random.uniform(150, 180, (1, 3, self.length, self.length))
        ) / float(255)
        opt_res = torch.from_numpy(rand_img).float()
        opt_res.requires_grad_(True)

        # Optimizer for image
        optimizer = torch.optim.Adam([opt_res], lr=self.lr, weight_decay=1e-6)

        for _ in range(self.opt_steps):
            # Reset gradient
            optimizer.zero_grad()

            # Do a forward pass
            out = opt_res
            for layer in self.layers:
                out = layer(out)

            # Retain grad for loss computation
            # out.retain_grad()

            # Calculate loss (mean of output of the last layer w.r.t to our mask)
            # TODO cache mask
            loss = -(self.top_layer_selector.get_mask(out.size()[1:]) * out[0]).mean()
            # Backward pass
            loss.backward()
            # Update image
            optimizer.step()

        return opt_res.data


if __name__ == "__main__":
    alexnet = models.alexnet(pretrained=True)
    layers = alexnet.features[:10]
    selector = NeuronSelector(ChannelSplit(), [0])

    res = PixelActivationOpt(layers, selector).visualize()
    res_img = torch.sub(res, res.min())
    res_img = torch.div(res_img, res_img.max())
    plt.imshow(res_img.squeeze(dim=0).permute(1, 2, 0).detach().numpy())
    plt.show()
