import numpy as np
from .utils import predict_proba, l0_norm, num_subsequences


def adversarial_loss(sample, y_nun, model, device="cpu"):
    """
    Classifier's probability for the desired class y_nun given sample'.

    :param `sample`: New generated sample
    :param `y_nun`: Label of the desired class
    :return `prob`: Probability prediction of the model
    """
    probabilities, pred_class = predict_proba(model, sample, device)
    prob = probabilities[0, y_nun].item()
    return prob, pred_class


def sparsity_loss(mask: np.ndarray) -> float:
    """
    Calculates the sparsity loss of a mask.

    :param `mask`: Numpy array representing the mask
    :return: Sparsity loss
    :raises `ValueError`: If the mask is empty
    """
    if len(mask) == 0:
        raise ValueError("The mask cannot be empty.")
    return -l0_norm(mask) / mask.size


def contiguity_loss(mask: np.ndarray, gamma: float = 0.25) -> float:
    """
    Calculates the contiguity loss of a mask.

    :param `mask`: Numpy array representing the mask
    :param `gamma`: Adjustment parameter, default is 0.25
    :return: Contiguity loss
    :raises `ValueError`: If the mask is empty
    """
    if len(mask) == 0:
        raise ValueError("The mask cannot be empty.")
    return -((num_subsequences(mask) / (mask.size / 2)) ** gamma)
