"""Main script. Run your experiments from here"""
import torchvision.models as models
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import Softmax

from interpretability_framework import data_utils
from interpretability_framework import modules


def main():
    """Example on how to use uncertainty measures for existing model.

    """
    # Get pre-trained model from torchvision
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier.add_module("softmax", Softmax(dim=1))

    # MC Ensemble that calculates the mean, predictive entropy, mutual information, and variation ratio.
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
    # TODO [data_utils.get_imagenet_example(), data_utils.get_ood_example(), data_utils.get_random_example()]:
    for img in [data_utils.get_random_example()]:
        # Do prediction
        prediction, predictive_entropy, mutual_information, variation_ratio = ensemble(
            img
        )

        print(
            f"prediction: {prediction.argmax()}, point estimate class probability: {prediction.max()}"
        )
        print(f"predictive entropy: {predictive_entropy.sum()}")
        print(f"mutual information: {mutual_information.sum()}")
        print(f"variational ratio: {variation_ratio}")

        # TODO Plot uncertainties?


if __name__ == "__main__":
    main()
