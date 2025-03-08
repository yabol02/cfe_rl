import os
import typing as tp
import numpy as np

def load_dataset(dataset: str) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the dataset from the specified path.

    :param `dataset`: Name of the dataset. The files must follow the path `./data/<dataset>`
    :return: A tuple containing (X_train, y_train, X_test, y_test)
    :raises FileNotFoundError: If any of the required files are missing
    :raises ValueError: If the dataset name is invalid
    """
    if not dataset:
        raise ValueError("Dataset name cannot be empty.")

    base_path = f"./data/{dataset}"
    files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]

    data = {}
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        data[file.split('.')[0]] = np.load(file_path, allow_pickle=True)

    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

# Adversarial Loss / Sparsity / Contiguity / Plausibility (AE)
def adversarial_loss(x, y_nun, model):
    """
    Classifier's probability for the desired class y_nun given x'

    :param `x`: New generated sample
    :param `y_nun`: Label of the desired class
    :return: Probability prediction of the model 
    """
    raise NotImplementedError

def l0_norm(mask: np.ndarray) -> int:
    """
    Calculates the L0 norm of a mask

    :param `mask`: Numpy array representing the mask
    :return: Number of non-zero elements in the mask
    """
    return np.count_nonzero(mask)

def sparsity_loss(mask: np.ndarray) -> float:
    """
    Calculates the sparsity loss of a mask

    :param `mask`: Numpy array representing the mask
    :return: Sparsity loss
    :raises ValueError: If the mask is empty
    """
    if len(mask) == 0:
        raise ValueError("The mask cannot be empty.")
    return -l0_norm(mask) / len(mask)

def num_subsequences(mask: np.ndarray) -> int:
    """
    Calculates the number of subsequences in a mask

    :param `mask`: Numpy array representing the mask
    :return: Number of subsequences
    """
    return np.count_nonzero(np.diff(mask))

def contiguity_loss(mask: np.ndarray, gamma: float = 0.25) -> float:
    """
    Calculates the contiguity loss of a mask

    :param `mask`: Numpy array representing the mask
    :param `gamma`: Adjustment parameter, default is 0.25
    :return: Contiguity loss
    :raises ValueError: If the mask is empty
    """
    if len(mask) == 0:
        raise ValueError("The mask cannot be empty.")
    return -(num_subsequences(mask) / (len(mask) / 2)) ** gamma

def l1_norm(x: tp.Union[np.ndarray, list], cfe: tp.Union[np.ndarray, list]) -> float:
    """
    Calculates the L1 norm between two arrays

    :param `x`: First array
    :param `cfe`: Second array
    :return: L1 norm between the two arrays
    """
    x = np.asarray(x).flatten()
    cfe = np.asarray(cfe).flatten()
    return np.linalg.norm(x - cfe, ord=1)

def l2_norm(x: tp.Union[np.ndarray, list], cfe: tp.Union[np.ndarray, list]) -> float:
    """
    Calculates the L2 norm between two arrays

    :param `x`: First array
    :param `cfe`: Second array
    :return: L2 norm between the two arrays
    """
    x = np.asarray(x).flatten()
    cfe = np.asarray(cfe).flatten()
    return np.linalg.norm(x - cfe, ord=2)

