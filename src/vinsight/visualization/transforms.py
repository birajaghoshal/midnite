"""Different image transformations, used for regularization in visualization."""
import random as rand
from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np
from numpy import ndarray


class TransformStep(ABC):
    """Abstract base class for an image transformation step"""

    @abstractmethod
    def transform(self, img: ndarray) -> ndarray:
        """Transforms the image.

        Args:
            img: the image to transform

        Returns:
            the transformed image

        """
        raise NotImplementedError()

    def __add__(self, other):
        """Method to easily concatenate two transformations.

        Args:
            other: the second transformation to apply

        Returns:
            a TransformationSequence containing self and other

        """
        return TransformationSequence(self, other)


class TransformationSequence(TransformStep):
    """Concatenates a number of transform steps as a sequence."""

    def __init__(self, *steps: TransformStep):
        self.steps = steps

    def transform(self, img: ndarray) -> ndarray:
        for step in self.steps:
            img = step.transform(img)
        return img

    def __add__(self, other: TransformStep) -> TransformStep:
        self.steps += (other,)
        return self


class BlurTrans(TransformStep):
    """Blur transformation step."""

    def __init__(self, blur_size: int = 5):
        self.blur_size = blur_size

    def transform(self, img: ndarray) -> ndarray:
        return cv2.blur(img, (self.blur_size, self.blur_size))


class ResizeTrans(TransformStep):
    """Resize transformation step.

    If you use this in an iterative optimization, make sure to start with a smaller
    init_size in order not to run out of GPU memory!

    """

    def __init__(self, scale_fac: float = 1.2):
        self.scale_fac = scale_fac

    def transform(self, img: ndarray) -> ndarray:
        return cv2.resize(img, (0, 0), fx=self.scale_fac, fy=self.scale_fac)


class RandomTrans(TransformStep):
    """Applies translation, rotation, and scale in a random direction"""

    def __init__(
        self, pixel_shift: int = 1, rot_deg: float = 0.3, scale_fac: float = 1.1
    ):
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
        scale = rand.uniform(1.0 / self.fac, self.fac)

        img = cv2.warpAffine(img, shift, (0, 0))
        img = cv2.warpAffine(img, rot, (0, 0))
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)


class BilateralTrans(TransformStep):
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
