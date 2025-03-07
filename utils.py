import numpy as np

# Adversarial Loss / Sparsity / Contiguity / Plausibility (AE)
def adversarial_loss(x, y_nun, model):
    """
    Classifier's probability for the desired class y_nun given x'.

    :param x: New generated sample
    :param y_nun: Label of the desired class
    :return: Probability prediction of the model 
    """
    raise NotImplementedError

def l0_norm(mask):
    return np.count_nonzero(mask)

def sparsity_loss(mask):
    return -l0_norm(mask)/len(mask)

def num_subsequences(mask):
    return np.count_nonzero(np.diff(mask))

def contiguity_loss(mask, gamma=0.25):
    return -(num_subsequences(mask)/(len(mask)/2))**gamma

def l1_norm(x, cfe):
    return np.linalg.norm(x.flatten()-cfe.flatten(), ord=1)

def l2_norm(x, cfe):
    return np.linalg.norm(x.flatten()-cfe.flatten(), ord=2)

