"""Different image transformations, used for regularization in visualization."""
import random as rand

import cv2
import numpy as np
from numpy import ndarray

from ..base.interface import TransformStep


class BlurTransform(TransformStep):
    """Blur transformation step."""

    def __init__(self, blur_size: int = 5):
        """
        Args:
            blur_size: how many pixels to blur

        """
        self.blur_size = blur_size

    def transform(self, img: ndarray) -> ndarray:
        return cv2.blur(img, (self.blur_size, self.blur_size))


class ResizeTransform(TransformStep):
    """Resize transformation step.

    If you use this in an iterative optimization, make sure to start with a smaller
    init_size in order not to run out of GPU memory!

    """

    def __init__(self, scale_fac: float = 1.2):
        """
        Args:
            scale_fac: resizing factor of each step. Keep in mind that large factors
             and/or many iterations will create huge memory requirements!

        """
        self.scale_fac = scale_fac

    def transform(self, img: ndarray) -> ndarray:
        return cv2.resize(
            img,
            (0, 0),
            fx=self.scale_fac,
            fy=self.scale_fac,
            interpolation=cv2.INTER_CUBIC,
        )


class RandomTransform(TransformStep):
    """Applies translation, rotation, and scale in a random direction"""

    def __init__(
        self, pixel_shift: int = 1, rot_deg: float = 0.3, scale_fac: float = 0.1
    ):
        """
        Args:
            pixel_shift: maximum amount of pixels to shift
            rot_deg: maximum degree of rotation
            scale_fac: maximum relative scale factor to resize

        """
        self.px = pixel_shift
        self.deg = rot_deg
        self.fac = scale_fac

    def transform(self, img: ndarray) -> ndarray:
        shift = np.float32(
            [
                [1, 0, rand.randint(-self.px, self.px)],
                [0, 1, rand.randint(-self.px, self.px)],
            ]
        )
        rot = cv2.getRotationMatrix2D(
            (img.shape[0] / 2, img.shape[1] / 2), rand.uniform(-self.deg, self.deg), 1
        )
        scale = 1 + rand.uniform(-self.fac, self.fac)

        img = cv2.warpAffine(img, shift, (0, 0))
        img = cv2.warpAffine(img, rot, (0, 0))
        return cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )


class BilateralTransform(TransformStep):
    """Bilateral filter step"""

    def __init__(self, diameter=9, color_tolerance=75, dist_tolerance=75):
        """
        Args:
            diameter: diameter of pixel neighbourhood for the filter
            color_tolerance: larger means that more distant colors are mixed together
            dist_tolerance: larger means that pixels further away are mixed together

        """
        self.diam = diameter
        self.color_sigma = color_tolerance
        self.space_sigma = dist_tolerance

    def transform(self, img: ndarray):
        return cv2.bilateralFilter(img, self.diam, self.color_sigma, self.space_sigma)
