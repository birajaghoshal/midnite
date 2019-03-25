"""Concrete implementations of visualization methods."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential

from vinsight.visualization import Activation
from vinsight.visualization import Attribution
from vinsight.visualization import LayerSplit
from vinsight.visualization import NeuronSelector


class TransformStep(ABC):
    @abstractmethod
    def transform(self, img: ndarray) -> ndarray:
        pass


class BlurTransformStep(TransformStep):
    def __init__(self, blur_size: int = 5):
        super().__init__()
        self.blur_size = blur_size

    def transform(self, img: ndarray) -> ndarray:
        return cv2.blur(img, (self.blur_size, self.blur_size))


class BlurResizeStep(BlurTransformStep):
    def __init__(self, blur_size: int = 5, scale_fac: float = 1.2):
        super().__init__(blur_size)
        self.scale_fac = scale_fac

    def transform(self, img: ndarray) -> ndarray:
        return cv2.resize(
            super().transform(img),
            (0, 0),
            fx=self.scale_fac,
            fy=self.scale_fac,
            interpolation=cv2.INTER_CUBIC,
        )


class PixelActivationOpt(Activation):
    """Simple Activation Optimization, using only blur regularization"""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        iter_transform: Optional[TransformStep] = None,
        lr: float = 1e-2,
        num_iters: int = 12,
        steps_per_iter: int = 20,
        init_size: int = 250,
        weight_decay: float = 1e-6,
    ):
        """
        Args:
            layers: the list of adjacent layers to account for
            top_layer_selector: selector for the relevant activations
            lr: learning rate
            steps: number of optimization steps per iteration
            iters: number of iteraions
            size: width/height of visualization
            weight_decay: weight decay of optimization
            blur: blur factor for regularization

        """
        super().__init__(layers, top_layer_selector)
        self.iter_transform = iter_transform
        self.lr = lr
        if num_iters < 1 or steps_per_iter < 1:
            raise ValueError("Must have at least one iteration")
        self.num_iters = num_iters
        self.steps_per_iter = steps_per_iter
        self.init_size = init_size
        self.weight_decay = weight_decay

    def visualize(self) -> Tensor:
        """Perform visualization.

        Returns: a tensor of dim (h, w, c) of the feature visualization

        """
        for layer in self.layers:
            layer.eval()

        # Starting image - tune uniform distribution weights if necessary
        img = np.uint8(
            np.random.uniform(150, 180, (self.init_size, self.init_size, 3))
        ) / float(255)

        device = torch.device("cuda" if torch.tensor([]).is_cuda else "cpu")

        for i in range(self.num_iters):
            opt_res = (
                torch.from_numpy(img)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(dim=0)
                .to(device)
            )
            opt_res.requires_grad_(True)

            # Optimizer for image
            optimizer = torch.optim.Adam(
                [opt_res], lr=self.lr, weight_decay=self.weight_decay
            )

            for _ in range(self.steps_per_iter):
                # Reset gradient
                optimizer.zero_grad()

                # Do a forward pass
                out = opt_res
                for layer in self.layers:
                    out = layer(out)

                # Calculate loss (mean of output of the last layer w.r.t to our mask)
                loss = -(
                    self.top_layer_selector.get_mask(out.size()[1:]) * out[0]
                ).mean()
                # Backward pass
                loss.backward()
                # Update image
                optimizer.step()

            if i + 1 < self.num_iters:
                img = opt_res.squeeze(dim=0).permute(1, 2, 0).detach().cpu().numpy()
                if self.iter_transform is not None:
                    img = self.iter_transform.transform(img)

        return opt_res.squeeze(dim=0).permute(1, 2, 0).detach().cpu()


class SaliencyMap(Attribution):
    """computes a saliency map for a selected class"""

    def __init__(
        self,
        layers: List[Module],
        bottom_layers: List[Module],
        bottom_layer_split: LayerSplit,
    ):
        """
        Args:
            layers: the list of adjacent layers from ModelSplit
            bottom_layers: list of lower level layers from ModelSplit
            bottom_layer_split: the split starting from which information are propagated
             through the network

        """
        super().__init__(layers, bottom_layer_split)
        self.bottom_layers = bottom_layers

    def visualize(self, selected_class, input_tensor: Tensor) -> Tensor:
        """generates a saliency tensor for selected_class
        Args:
            selected_class: class for which the saliency is computed
            input_tensor: input image as torch tensor
        """
        device = torch.device("cuda" if torch.tensor([]).is_cuda else "cpu")

        features = Sequential(*self.bottom_layers)(input_tensor.to(device))
        output = Sequential(*self.layers)(features)

        class_score = output[0, int(selected_class)]
        _, N, H, W = features.size()

        # computes and returns the sum of gradients of the class score
        # w.r.t. the features of the selected layer
        grads = torch.autograd.grad(class_score, features)
        w = grads[0][0].mean(-1).mean(-1)
        sal_map = torch.matmul(w, features.view(N, H * W))
        sal_map = sal_map.view(H, W).detach()
        sal_map = torch.max(sal_map, torch.zeros_like(sal_map).to(device)).cpu()

        return sal_map
