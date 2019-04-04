"""Unit test for the visualization methods"""
import numpy as np
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal
from torch import Size
from torch.nn import Dropout2d
from torch.nn import Module

from midnite.visualization import NeuronSelector
from midnite.visualization import PixelActivation
from midnite.visualization import TransformStep
from midnite.visualization import TVRegularization
from midnite.visualization import WeightDecay


def test_tv_reg_all_same():
    """Test that tv loss is zero if all pixels are eqal."""
    reg = TVRegularization()
    loss = reg.loss(torch.ones((2, 2, 3)))

    assert_that(loss.item()).is_equal_to(0.0)


def test_tv_reg():
    """Test tv loss for an arbitrary example."""
    reg = TVRegularization(0.5, 4.0)
    loss = reg.loss(
        torch.tensor(
            [
                [[1.0, 0.9, 0.1], [0.8, 0.4, 0.4]],
                [[1.0, 0.9, 0.1], [0.8, 0.4, 0.4]],
                [[0, 0, 0], [0, 0, 0]],
            ]
        )
    )

    assert_that(loss.item()).is_close_to(0.0441, 1e-4)


def test_activation_init(mocker):
    """Test the more complex init logic in the pixel activation"""
    sut = PixelActivation(
        [mocker.Mock(spec=torch.nn.Module)], mocker.Mock(spec=NeuronSelector)
    )

    assert_that(sut.weight_decay).is_equal_to(0.0)
    assert_that(sut.output_regularizers).is_not_none()
    assert_that(sut.output_regularizers).is_empty()

    sut = PixelActivation(
        [mocker.Mock(spec=torch.nn.Module)],
        mocker.Mock(spec=NeuronSelector),
        regularization=[WeightDecay(0.5)],
    )
    assert_that(sut.weight_decay).is_equal_to(0.5)
    assert_that(sut.output_regularizers).is_not_none()
    assert_that(sut.output_regularizers).is_empty()

    tv_reg = TVRegularization()
    sut = PixelActivation(
        [mocker.Mock(spec=torch.nn.Module)],
        mocker.Mock(spec=NeuronSelector),
        regularization=[tv_reg, WeightDecay(0.5)],
    )
    assert_that(sut.weight_decay).is_equal_to(0.5)
    assert_that(sut.output_regularizers).contains(tv_reg)


def test_pixel_activation_visualize(mocker):
    """Test the iterative optimization of the activation visualization."""
    # Setup data
    img = torch.zeros((3, 4, 4))
    np_img = np.zeros((4, 4, 3))

    # Setup mocks
    mocker.patch(
        "midnite.visualization.methods.PixelActivation._opt_step", return_value=img
    )
    tran_step = mocker.Mock(spec=TransformStep)
    tran_step.transform = mocker.Mock(return_value=np_img)

    visualizer = PixelActivation(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        iter_n=3,
        init_size=4,
        transform=tran_step,
    )
    out = visualizer.visualize().numpy()

    # Check transform step calls
    assert_that(tran_step.transform.call_count).is_equal_to(2)
    tran_step = tran_step.transform.call_args_list
    assert_array_equal(tran_step[0][0][0], np_img)
    assert_array_equal(tran_step[1][0][0], np_img)

    # Check opt step calls
    assert_array_equal(out, np_img)
    assert_that(PixelActivation._opt_step.call_count).is_equal_to(3)
    opt_step = PixelActivation._opt_step.call_args_list
    assert_that(opt_step[0][0][0].size()).contains_sequence(3, 4, 4)
    assert_array_equal(opt_step[1][0][0].detach(), img.numpy())
    assert_array_equal(opt_step[2][0][0].detach(), img.numpy())


def test_pixel_act_optimize(mocker):
    """Test the optimization step of the activation visualization."""
    img = torch.zeros((3, 4, 4))
    layers = [Dropout2d(), Dropout2d()]

    mocker.spy(layers[0], "forward")
    mocker.spy(layers[1], "forward")
    neuron_sel = mocker.Mock(spec=NeuronSelector)
    neuron_sel.get_mask = mocker.Mock(return_value=torch.ones((3, 4, 4)))
    reg = mocker.Mock(spec=TVRegularization)
    reg.loss = mocker.Mock(return_value=0.05)

    visualizer = PixelActivation(
        layers, neuron_sel, lr=1e-4, opt_n=2, regularization=[WeightDecay(1e-5), reg]
    )

    visualizer._opt_step(img)

    # Check layers
    layers[0].forward.assert_called()
    layers[1].forward.assert_called()
    assert_that(layers[0].forward.call_count).is_equal_to(2)
    assert_that(layers[1].forward.call_count).is_equal_to(2)
    layer_call_args = layers[0].forward.call_args_list
    assert_that(layer_call_args[0][0][0].size()).is_equal_to(Size([1, 3, 4, 4]))
    assert_that(layer_call_args[1][0][0].size()).is_equal_to(Size([1, 3, 4, 4]))

    # Check neuron selector
    neuron_sel.get_mask.assert_called_with(Size([3, 4, 4]))
    assert_that(neuron_sel.get_mask.call_count).is_equal_to(2)

    # Check regularizer
    reg.loss.assert_called()
    assert_that(reg.loss.call_count).is_equal_to(2)
    loss_call_args = reg.loss.call_args_list
    assert_that(loss_call_args[0][0][0].size()).is_equal_to(Size([3, 4, 4]))
    assert_that(loss_call_args[1][0][0].size()).is_equal_to(Size([3, 4, 4]))
