"""Tests for the image transforms."""
import cv2
import numpy as np
from assertpy import assert_that
from numpy.testing import assert_array_equal

from midnite.visualization.transforms import BilateralTransform
from midnite.visualization.transforms import BlurTransform
from midnite.visualization.transforms import RandomTransform
from midnite.visualization.transforms import ResizeTransform
from midnite.visualization.transforms import TransformSequence
from midnite.visualization.transforms import TransformStep


def test_transform_step_add():
    """Test adding of transform steps"""
    step = BlurTransform() + BlurTransform()

    assert_that(step).is_instance_of(TransformSequence)
    assert_that(step.steps).is_length(2)


def test_sequence(mocker):
    """Tests the transformation sequence."""
    img0 = np.array(0)
    img1 = np.array(1)
    step1 = mocker.Mock(spec=TransformStep)
    step1.transform = mocker.Mock(return_value=img1)
    step2 = mocker.Mock(spec=TransformStep)

    seq = TransformSequence(step1, step2)
    seq.transform(img0)

    step1.transform.assert_called_once_with(img0)
    step2.transform.assert_called_once_with(img1)


def test_sequence_add(mocker):
    """Test addition of a sequence (should keep it flat)."""
    step1 = mocker.Mock(spec=TransformStep)
    step2 = mocker.Mock(spec=TransformStep)
    step3 = mocker.Mock(spec=TransformStep)

    seq = TransformSequence(step1, step2) + step3
    seq.transform(np.array(0))

    step1.transform.assert_called_once()
    step2.transform.assert_called_once()
    step3.transform.assert_called_once()


def test_blur_trans(mocker):
    """Test that blur step correctly calls cv2."""
    img0 = np.array(0)
    img1 = np.array(1)
    mocker.patch("cv2.blur", return_value=img1)

    step = BlurTransform(blur_size=10)
    res = step.transform(img0)

    cv2.blur.assert_called_once_with(img0, (10, 10))
    assert_that(res).is_equal_to(img1)


def test_resize_trans(mocker):
    """Test that the resize step correctly calls cv2."""
    img0 = np.array(0)
    img1 = np.array(1)
    mocker.patch("cv2.resize", return_value=img1)

    step = ResizeTransform(scale_fac=2.0)
    res = step.transform(img0)

    cv2.resize.assert_called_once_with(img0, (0, 0), fx=2.0, fy=2.0)
    assert_that(res).is_equal_to(img1)


def test_bilateral_trans(mocker):
    """Test cv2 call of the bilateral filter step."""
    img0 = np.array(0)
    img1 = np.array(1)
    mocker.patch("cv2.bilateralFilter", return_value=img1)

    step = BilateralTransform(10, 100, 110)
    res = step.transform(img0)

    cv2.bilateralFilter.assert_called_once_with(img0, 10, 100, 110)
    assert_that(res).is_equal_to(img1)


def test_random_trans(mocker):
    """Test cv2 calls of the random transformation step."""
    img0 = np.zeros((3, 3, 3))
    img1 = np.ones((3, 3, 3))
    mocker.patch("cv2.warpAffine", return_value=img0)
    mocker.patch("cv2.resize", return_value=img1)

    step = RandomTransform()
    res = step.transform(img0)

    assert_that(cv2.warpAffine.call_count).is_equal_to(2)
    cv2.resize.assert_called_once()
    assert_array_equal(res, img1)
