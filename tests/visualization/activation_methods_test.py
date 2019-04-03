"""Unit test for the visualization methods"""
import numpy as np
import torch
from assertpy import assert_that
from numpy.testing import assert_array_equal
from torch import Size
from torch.nn import Dropout2d
from torch.nn import Module

from vinsight.visualization import ChannelSplit
from vinsight.visualization import GuidedBackpropagation
from vinsight.visualization import LayerSplit
from vinsight.visualization import NeuronSelector
from vinsight.visualization import NeuronSplit
from vinsight.visualization import PixelActivation
from vinsight.visualization import SaliencyMap
from vinsight.visualization import SpatialSplit
from vinsight.visualization import TransformStep
from vinsight.visualization import TVRegularization
from vinsight.visualization import WeightDecay


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
        "vinsight.visualization.methods.PixelActivation._opt_step", return_value=img
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


def test_select_top_layer_score_3d(mocker):
    """tests if the top-layer output score is correctly computed
    for 3d top layer dimension."""

    # 3 dimensional selection, example neuron split
    top_layer_selector_3d = mocker.Mock(spec=NeuronSelector)
    top_layer_selector_3d.get_mask = mocker.Mock(return_value=torch.ones((4, 3, 3)))

    out = torch.ones((1, 4, 3, 3))

    backprop = GuidedBackpropagation(
        [mocker.Mock(spec=Module)], top_layer_selector_3d, mocker.Mock(spec=LayerSplit)
    )

    score = backprop.select_top_layer_score(out)

    assert_that(score.size()).is_equal_to(Size([]))
    assert_that(score).is_equal_to(1)


def test_select_top_layer_score_1d(mocker):
    """tests if the top-layer output score is correctly computed
    for 1d top layer dimension."""

    # 1 dimensional selection, example neuron split
    top_layer_selector_1d = mocker.Mock(spec=NeuronSelector)
    top_layer_selector_1d.get_mask = mocker.Mock(return_value=torch.ones((50,)))

    out = torch.ones((50,))

    backprop = GuidedBackpropagation(
        [mocker.Mock(spec=Module)], top_layer_selector_1d, mocker.Mock(spec=LayerSplit)
    )

    score = backprop.select_top_layer_score(out)

    assert_that(score.size()).is_equal_to(Size([]))
    assert_that(score).is_equal_to(1)


def test_guided_backprop_visualize(mocker):
    img = torch.zeros((1, 3, 4, 4))
    layers = [Dropout2d(), Dropout2d()]

    mocker.spy(layers[0], "forward")
    mocker.spy(layers[1], "forward")
    mocker.patch("torch.autograd.grad", return_value=torch.randn(1, 3, 4, 4))

    top_layer_sel = mocker.Mock(spec=NeuronSelector)
    top_layer_sel.get_mask = mocker.Mock(return_value=torch.ones((3, 4, 4)))

    backprop_spatial = GuidedBackpropagation(layers, top_layer_sel, SpatialSplit())
    pos_grad_spatial = backprop_spatial.visualize(img)

    # Check forward pass
    layers[0].forward.assert_called()
    layers[1].forward.assert_called()
    assert_that(layers[0].training).is_false()
    assert_that(layers[1].training).is_false()

    assert_that(np.all(pos_grad_spatial.numpy() >= 0)).is_true()
    assert_that(pos_grad_spatial.size()).is_equal_to(Size([4, 4]))

    backprop_channel = GuidedBackpropagation(layers, top_layer_sel, ChannelSplit())
    pos_grad_channel = backprop_channel.visualize(img)

    assert_that(np.all(pos_grad_channel.numpy() >= 0)).is_true()
    # check output wrt split dimension
    assert_that(pos_grad_channel.size()).is_equal_to(Size([3]))

    backprop_neuron = GuidedBackpropagation(layers, top_layer_sel, NeuronSplit())
    pos_grad_neuron = backprop_neuron.visualize(img)

    assert_that(np.all(pos_grad_neuron.numpy() >= 0)).is_true()
    # check output wrt split dimension
    assert_that(pos_grad_neuron.size()).is_equal_to(Size([3, 4, 4]))


def test_saliency_visualize_spatial(mocker):
    img = torch.zeros((1, 3, 4, 4))
    base_layers = [Dropout2d(), Dropout2d()]
    mocker.spy(base_layers[0], "forward")
    mocker.spy(base_layers[1], "forward")

    # spatial case
    mocker.patch(
        "vinsight.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones((3)),
    )
    mocker.patch(
        "vinsight.visualization.interface.SpatialSplit.invert",
        return_value=ChannelSplit(),
    )
    mocker.patch(
        "vinsight.visualization.interface.ChannelSplit.fill_dimensions",
        return_value=torch.ones((3, 1, 1)),
    )

    sal_map = SaliencyMap(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        SpatialSplit(),
    ).visualize(img)

    base_layers[0].forward.assert_called()
    base_layers[1].forward.assert_called()
    assert_that(base_layers[0].training).is_false()
    assert_that(base_layers[1].training).is_false()

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([4, 4]))


def test_saliency_visualize_channel(mocker):
    img = torch.zeros((1, 3, 4, 4))
    base_layers = [Dropout2d(), Dropout2d()]
    mocker.spy(base_layers[0], "forward")
    mocker.spy(base_layers[1], "forward")

    # spatial case
    mocker.patch(
        "vinsight.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones((4, 4)),
    )
    mocker.patch(
        "vinsight.visualization.interface.ChannelSplit.invert",
        return_value=SpatialSplit(),
    )
    mocker.patch(
        "vinsight.visualization.interface.SpatialSplit.fill_dimensions",
        return_value=torch.ones((1, 4, 4)),
    )

    sal_map = SaliencyMap(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        ChannelSplit(),
    ).visualize(img)

    base_layers[0].forward.assert_called()
    base_layers[1].forward.assert_called()
    assert_that(base_layers[0].training).is_false()
    assert_that(base_layers[1].training).is_false()

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([3]))


def test_saliency_visualize_neuron(mocker):
    img = torch.zeros((1, 3, 4, 4))
    base_layers = [Dropout2d(), Dropout2d()]
    mocker.spy(base_layers[0], "forward")
    mocker.spy(base_layers[1], "forward")

    # spatial case
    mocker.patch(
        "vinsight.visualization.methods.GuidedBackpropagation.visualize",
        return_value=torch.ones((3, 4, 4)),
    )
    mocker.patch(
        "vinsight.visualization.interface.NeuronSplit.invert",
        return_value=NeuronSplit(),
    )
    mocker.patch(
        "vinsight.visualization.interface.NeuronSplit.fill_dimensions",
        return_value=torch.ones((3, 4, 4)),
    )

    sal_map = SaliencyMap(
        [mocker.Mock(spec=Module)],
        mocker.Mock(spec=NeuronSelector),
        base_layers,
        NeuronSplit(),
    ).visualize(img)

    base_layers[0].forward.assert_called()
    base_layers[1].forward.assert_called()
    assert_that(base_layers[0].training).is_false()
    assert_that(base_layers[1].training).is_false()

    assert_that(np.all(sal_map.numpy() >= 0)).is_true()
    assert_that(sal_map.size()).is_equal_to(Size([3, 4, 4]))
