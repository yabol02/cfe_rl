import os
import torch
import tensorflow as tf
import typing as tp
import numpy as np
from torch.nn import functional as F

def min_max_scale_data(X_train, X_test):
    """
    Scales the training and test data using Min-Max normalization.

    :param `X_train`: The training data
    :param `X_test`: The test data
    :return: Scaled training and test data
    """
    max = 1
    min = 0
    data_max = X_train.max()
    data_min = X_train.min()

    X_train_scaled = (X_train - data_min) / (data_max - data_min)
    X_train_scaled = X_train_scaled * (max - min) + min

    X_test_scaled = (X_test - data_min) / (data_max - data_min)
    X_test_scaled = X_test_scaled * (max - min) + min

    return X_train_scaled, X_test_scaled

def standard_scale_data(X_train, X_test):
    """
    Scales the training and test data using standardization (Z-score).

    :param `X_train`: The training data
    :param `X_test`: The test data
    :return: Standardized training and test data
    """
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()

    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return X_train, X_test

def load_dataset(dataset: str, scaling: str = "none", backend: str = "torch") -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the dataset from the specified path and applies optional scaling and backend handling.

    :param `dataset`: Name of the dataset. The files must follow the path `./data/<dataset>`
    :param `scaling`: Scaling method ('min_max', 'standard', 'none')
    :param `backend`: Backend to format the data ('torch' or 'tf')
    :return: A tuple containing (X_train, y_train, X_test, y_test)
    :raises `FileNotFoundError`: If any of the required files are missing
    :raises `ValueError`: If the dataset name is invalid or scaling method is unsupported
    """
    if not dataset:
        raise ValueError("Dataset name cannot be empty.")

    base_path = f"./data/{dataset}"
    files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]

    data = dict()
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        data[file.split('.')[0]] = np.load(file_path, allow_pickle=True)

    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if scaling == "min_max":
        X_train, X_test = min_max_scale_data(X_train, X_test)
    elif scaling == "standard":
        X_train, X_test = standard_scale_data(X_train, X_test)
    elif scaling != "none":
        raise ValueError("Invalid scaling method. Choose 'min_max', 'standard', or 'none'.")

    if backend == "torch":
        X_train = torch.tensor(X_train).permute(0, 2, 1)
        X_test = torch.tensor(X_test).permute(0, 2, 1)
    elif backend == "tf":
        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)
    else:
        raise ValueError("Invalid backend. Choose 'torch' or 'tf'.")

    return X_train, y_train, X_test, y_test

def predict_proba(model, data, device='cpu'):
    """
    Predicts the class probabilities and the predicted class for the given data using the model.

    The function performs preprocessing on the input data, passes it through the model, and returns the probabilities for each class as well as the predicted class.

    :param `model`: The trained model used for making predictions.
    :param `data`: The input data (either as a tensor or a numpy array) for which predictions are made.
    :param `device`: The device ('cpu' or 'cuda') on which the model and data should be processed.
    :return `probabilities`: A tensor containing class probabilities for the input data
    _return `predicted_class`: The index of the class with the highest probability
    """
    model.eval()

    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        data_tensor = data
    if len(data_tensor.shape) == 2:
        seq_length, features = data_tensor.shape
        data_tensor = data_tensor.reshape(1, features, seq_length)
    elif len(data_tensor.shape) == 1:
        seq_length = data_tensor.shape[0]
        data_tensor = data_tensor.reshape(1, 1, seq_length)

    data_tensor = data_tensor.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = F.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, 1)
    
    return probabilities, predicted_class.item()

def adversarial_loss(x, y_nun, model, device='cpu'):
    """
    Classifier's probability for the desired class y_nun given x'.

    :param `x`: New generated sample
    :param `y_nun`: Label of the desired class
    :return `prob`: Probability prediction of the model 
    """
    probabilities, pred_class = predict_proba(model, x, device)
    prob = probabilities[0, y_nun].item()
    return prob, pred_class

def l0_norm(mask: np.ndarray) -> int:
    """
    Calculates the L0 norm of a mask.

    :param `mask`: Numpy array representing the mask
    :return: Number of non-zero elements in the mask
    """
    return np.count_nonzero(mask)

def sparsity_loss(mask: np.ndarray) -> float:
    """
    Calculates the sparsity loss of a mask.

    :param `mask`: Numpy array representing the mask
    :return: Sparsity loss
    :raises `ValueError`: If the mask is empty
    """
    if len(mask) == 0:
        raise ValueError("The mask cannot be empty.")
    return -l0_norm(mask) / len(mask)

def num_subsequences(mask: np.ndarray) -> int:
    """
    Calculates the number of subsequences in a mask.

    :param `mask`: Numpy array representing the mask
    :return: Number of subsequences
    """
    return np.count_nonzero(np.diff(mask))

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
    return -(num_subsequences(mask) / (len(mask) / 2)) ** gamma

def l1_norm(x: tp.Union[np.ndarray, list], cfe: tp.Union[np.ndarray, list]) -> float:
    """
    Calculates the L1 norm between two arrays.

    :param `x`: First array
    :param `cfe`: Second array
    :return: L1 norm between the two arrays
    """
    x = np.asarray(x).flatten()
    cfe = np.asarray(cfe).flatten()
    return np.linalg.norm(x - cfe, ord=1)

def l2_norm(x: tp.Union[np.ndarray, list], cfe: tp.Union[np.ndarray, list]) -> float:
    """
    Calculates the L2 norm between two arrays.

    :param `x`: First array
    :param `cfe`: Second array
    :return: L2 norm between the two arrays
    """
    x = np.asarray(x).flatten()
    cfe = np.asarray(cfe).flatten()
    return np.linalg.norm(x - cfe, ord=2)

def load_model(dataset, experiment, t='best'):
    """
    Loads a trained model from the specified experiment based on the chosen criteria.

    This function loads a model from the specified dataset and experiment folder. 
    The experiment can be selected based on various criteria: 'best', 'random', 'worst', 'median', or an integer index to choose a specific experiment.

    :param `dataset`: The name of the dataset
    :param `experiment`: The name of the experiment folder within the dataset
    :param `t`: The criteria for selecting the experiment. Can be 'best', 'random', 'worst', 'median' or an integer index. Default is 'best'
    :return `model`: The trained model loaded from the specified experiment
    :raises `ValueError`: If 't' is not one of the valid selection methods or if an invalid index is provided
    """
    from pandas import read_excel
    exp_path = os.path.join('models', dataset, experiment)
    df = read_excel(f'{os.path.join(exp_path, "all_results.xlsx")}')
    if t == 'best':
        exp_hash = df.experiment_hash.iloc[0]
    elif t == 'random':
        exp_hash = df.experiment_hash.sample(n=1).iloc[0]
    elif t == 'worst':
        exp_hash = df.experiment_hash.iloc[-1]
    elif t == 'median':
        exp_hash = df.experiment_hash.iloc[len(df)//2]
    elif t.isdigit():
        idx = int(t)
        if idx < 0 or idx >= len(df):
            raise ValueError(f"Index {idx} out of range. Valid range: 0-{len(df)-1}")
        exp_hash = df.experiment_hash.iloc[idx]
    else:
        raise ValueError('The way to choose an experiment (t) should be one of these: best, random, worst, median or a number.')
    model = torch.load(os.path.join(exp_path, exp_hash, 'model.pth'), weights_only=False)
    return model
