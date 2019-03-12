"""Main script. Run your experiments from here"""
import torch
import torchvision.models as models

from interpretability_framework import data_utils
from interpretability_framework import functional as func
from interpretability_framework import modules


def main():
    """Tutorial for using uncertainty measures.

    """
    # Parameters
    dropout_prob = 0.8
    sample_size = 20

    # Get input image from dataloader, torch tensor of dim: (1 x 3 x 227 x 227)
    input_img = data_utils.get_imagenet_example()

    # Get pretrained model from torchvision
    alexnet = models.alexnet(pretrained=True)

    layers = list(alexnet.classifier)

    # Exchange dropout layers
    for idx, layer in enumerate(layers):
        if isinstance(layer, torch.nn.Dropout):
            layers.remove(layer)
            layers.insert(idx, modules.PredDropout(dropout_prob))

    # Get MC Dropout predictions
    # resnet18.forward(cifar10_data)
    new_classifier = torch.nn.Sequential(*layers)
    new_model = torch.nn.Sequential(alexnet.features, new_classifier)

    ensemble_layer = modules.PredictionEnsemble(new_model, sample_size)
    ensemble_layer.eval()

    # Compute predictions
    pred_ensemble = ensemble_layer(input_img)

    # Measure uncertainties for MC ensemble
    pred_entropy = func.predictive_entropy(pred_ensemble)
    mutual_info = func.mutual_information(pred_ensemble)
    var_ratio = func.variation_ratio(pred_ensemble)
    print("pred_entropy: ", pred_entropy)
    print("mutual_information: ", mutual_info)
    print("variational_ratio: ", var_ratio)

    # TODO Plot uncertainties?
    pass


if __name__ == "__main__":
    main()
