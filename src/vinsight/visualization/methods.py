"""Concrete implementations of visualization methods."""
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from vinsight.visualization import Activation
from vinsight.visualization import NeuronSelector


class PixelActivationOpt(Activation):
    """Simple Activation Optimization, using only blur regularization"""

    def __init__(
        self,
        layers: List[Module],
        top_layer_selector: NeuronSelector,
        lr=1e-2,
        steps=20,
        iters=10,
        size=250,
        weight_decay=1e-6,
        blur=5,
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
        self.lr = lr
        self.steps = steps
        self.iters = iters
        self.size = size
        self.weight_decay = weight_decay
        self.blur = blur

    def visualize(self) -> Tensor:
        """Perform visualization.

        Returns: a tensor of dim (h, w, c) of the feature visualization

        """
        for layer in self.layers:
            layer.eval()

        # Starting image - tune uniform distribution weights if necessary
        img = np.uint8(np.random.uniform(150, 180, (self.size, self.size, 3))) / float(
            255
        )

        for _ in range(self.iters):
            # Blur image
            img = cv2.blur(img, (self.blur, self.blur))
            opt_res = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(dim=0)
            opt_res.requires_grad_(True)

            # Optimizer for image
            optimizer = torch.optim.Adam(
                [opt_res], lr=self.lr, weight_decay=self.weight_decay
            )

            for _ in range(self.steps):
                # Reset gradient
                optimizer.zero_grad()

                # Do a forward pass
                out = opt_res
                for layer in self.layers:
                    out = layer(out)

                # Calculate loss (mean of output of the last layer w.r.t to our mask)
                # TODO cache mask
                loss = -(
                    self.top_layer_selector.get_mask(out.size()[1:]) * out[0]
                ).mean()
                # Backward pass
                loss.backward()
                # Update image
                optimizer.step()

            img = opt_res.data.squeeze(dim=0).permute(1, 2, 0).detach().numpy()

        return torch.from_numpy(img).float()
