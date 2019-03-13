"""Main script. Run your experiments from here"""
import torchvision.models as models
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import Softmax

import data_utils
from interpretability_framework import modules


def main():
    """Example on how to use uncertainty measures for existing model.

    """
    # Get pre-trained model from torchvision
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier.add_module("softmax", Softmax(dim=1))

    # MC Ensemble that calculates the mean, predictive entropy, mutual information,
    # and variation ratio.
    ensemble = Sequential(
        modules.PredictionEnsemble(inner=alexnet), modules.ConfidenceMeanPrediction()
    )

    # Prepare ensemble
    ensemble.eval()

    # Re-activate dropout layers
    for layer in list(alexnet.modules()):
        if isinstance(layer, Dropout):
            layer.train()

    # Get input images from dataloader, torch tensor of dim: (1 x 3 x 227 x 227)
    for img in [
        # data_utils.get_imagenet_example(),
        # data_utils.get_ood_example(),
        data_utils.get_random_example()
    ]:
        # Do prediction
        pred, pred_entropy, mutual_info, var_ratio = ensemble.forward(img)

        print(f"mean prediction: {pred.argmax()}, class probability: {pred.max()}")
        print(f"total predictive entropy: {pred_entropy.sum()}")
        print(f"total mutual information: {mutual_info.sum()}")
        print(f"variational ratio: {var_ratio.item()}")


if __name__ == "__main__":
    main()
