import torch


def ensemble_mean(input):
    """Gives the mean over sampled predictions.

    Args:
        input: concatenated predictions for N classes and T samples, of shape (N, T)

    Returns:
        the mean prediction for all samples

    """
    return torch.mean(input, dim=1)


def predictive_entropy(input):
    """Calculates the predictive entropy over the samples in the input.

    Args:
        input: concatenated predictions for N classes and T samples, of shape (N, T)

    Returns: the predictive entropy, i.e. -sum_y p(y|x)*log p(y|x)

    """
    _ensemble_mean = ensemble_mean(input)

    return -(_ensemble_mean * torch.log(_ensemble_mean))


def mutual_information(input):
    """Calculates the mutual information over the samples in the input.

    Args:
        input: concatenated predictions for N classes and T samples, of shape (N, T)

    Returns: the mutual information, i.e. H[p(y|x)] + 1/T sum_t p(y|x,w_t)*log p(y|x,w_t)

    """
    num_samples = input.dim()[1]
    expected_entropy = torch.div(
        torch.sum(input * torch.log(input), dim=2), num_samples
    )

    return predictive_entropy(input) + expected_entropy
