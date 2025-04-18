import os
import torch as th
import typing as tp
import numpy as np
from torch.nn import functional
from json import JSONEncoder


def predict_proba(model, data, device="cpu"):
    """
    Predicts the class probabilities and the predicted class for the given data using the model.

    The function performs preprocessing on the input data, passes it through the model, and returns the probabilities for each class as well as the predicted class.

    :param `model`: The trained model used for making predictions.
    :param `data`: The input data (either as a tensor or a numpy array) for which predictions are made.
    :param `device`: The device ('cpu' or 'cuda') on which the model and data should be processed.
    :return `probabilities`: A tensor containing class probabilities for the input data
    :return `predicted_class`: The index of the class with the highest probability
    """
    model.eval()

    if isinstance(data, np.ndarray):
        data_tensor = th.tensor(data, dtype=th.float32)
    else:
        data_tensor = data

    if data_tensor.ndim == 2:
        features, seq_length = data_tensor.shape
        data_tensor = data_tensor.reshape(1, features, seq_length)
    elif data_tensor.ndim == 1:
        seq_length = data_tensor.shape[0]
        data_tensor = data_tensor.reshape(1, 1, seq_length)

    data_tensor = data_tensor.to(device)
    model = model.to(device)

    with th.no_grad():
        logits = model(data_tensor)
        probabilities = functional.softmax(logits, dim=1)
        _, predicted_class = th.max(logits, 1)

    pred_class = (
        predicted_class.item() if predicted_class.shape[0] == 1 else predicted_class
    )

    return probabilities, pred_class


def num_subsequences(mask: np.ndarray) -> int:
    """
    Calculates the number of subsequences in a mask.

    :param `mask`: Numpy array representing the mask
    :return: Number of subsequences
    """
    return np.count_nonzero(np.diff(mask, prepend=0, axis=1) == 1, axis=(0, 1))


def l0_norm(mask: np.ndarray) -> int:
    """
    Calculates the L0 pseudo-norm of a mask.

    :param `mask`: Numpy array representing the mask
    :return: Number of non-zero elements in the mask
    """
    return np.count_nonzero(mask)


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


def plot_signal(X, X2, mask, ax, dataset=None):
    """
    Plots three versions of a time series signal:
    - The original signal (in blue)
    - A modified signal (in red), where masked values are replaced by X2
    - The X2 signal (in black)

    It also highlights masked regions with semi-transparent red spans.

    :param `X`: Original time series signal
    :param `X2`: Alternative signal used to replace masked values
    :param `mask`: Mask indicating which values in X are replaced by X2
    :param `ax`: Axis on which to plot
    :param `dataset`: Name to show in the title, default is None
    :raises ValueError: If X and X2 do not have the same size
    :raises ValueError: If X and mask do not have the same size
    :raises ValueError: If mask is not a one-dimensional array
    :raises ValueError: If the variables cannot be properly flattened
    """
    X, X2, mask = map(np.asarray, (X, X2, mask))

    X_flat, X2_flat, mask_flat = map(
        lambda arr: arr.flatten() if np.prod(arr.shape) == max(arr.shape) else arr,
        (X, X2, mask),
    )

    if X_flat.shape != X2_flat.shape or X_flat.shape != mask_flat.shape:
        raise ValueError(
            f"Variables cannot be properly flattened: {X.shape}, {X2.shape}, {mask.shape}"
        )

    X, X2, mask = X_flat, X2_flat, mask_flat

    if X.shape != X2.shape:
        raise ValueError(
            f"`X` and `X2` must have the same size, but got {X.shape} and {X2.shape}"
        )
    if X.shape != mask.shape:
        raise ValueError(
            f"`X` and `mask` must have the same size, but got {X.shape} and {mask.shape}"
        )
    if mask.ndim != 1:
        raise ValueError("`mask` must be a one-dimensional array")
    try:
        mask = mask.astype(np.bool_)
    except ValueError:
        raise TypeError("`mask` must be a list of binary values (0/1 or True/False)")

    mod = X.copy()
    mod[mask] = X2[mask]
    ax.clear()
    ax.plot(X2, c="k", label="NUN")
    ax.plot(X, c="b", label="Original")
    ax.plot(mod, c="r", label="CFE")

    submasks = extract_submasks(mask)
    for subm in submasks:
        x, y = subm
        ax.axvspan(
            x - (1 if x > 0 else 0),
            y - (1 if y >= len(X) else 0),
            color="red",
            alpha=0.1,
        )

    ax.set_title(
        f"CFE{f' - {dataset}' if dataset else ''}", fontsize=14, fontweight="bold"
    )


def extract_submasks(mask):
    """
    Identifies contiguous regions where the mask is True and returns their dimension, start and end indices.

    :param `mask`: Boolean mask indicating the regions of interest
    :return: List of tuples (start_idx, end_idx) representing the masked regions
    :raises TypeError: If mask does not contain binary values (0/1 or True/False)
    """
    if not all(value in [True, False] for dim in mask for value in dim):
        raise TypeError("The mask must contain only binary values (True/False or 1/0).")

    if not any([any(dimension) for dimension in mask]):
        return []

    submasks = []

    for dim in range(len(mask)):
        start_idx = None
        for i, value in enumerate(mask[dim]):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                submasks.append([dim, start_idx, i])
                start_idx = None

        if start_idx is not None:
            submasks.append([dim, start_idx, mask.shape[-1]])

    return submasks


def load_model(dataset, experiment, t="best"):
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

    exp_path = os.path.join("models", dataset, experiment)
    df = read_excel(f'{os.path.join(exp_path, "all_results.xlsx")}')
    if t == "best":
        exp_hash = df.experiment_hash.iloc[0]
    elif t == "random":
        exp_hash = df.experiment_hash.sample(n=1).iloc[0]
    elif t == "worst":
        exp_hash = df.experiment_hash.iloc[-1]
    elif t == "median":
        exp_hash = df.experiment_hash.iloc[len(df) // 2]
    elif t.isdigit():
        idx = int(t)
        if idx < 0 or idx >= len(df):
            raise ValueError(f"Index {idx} out of range. Valid range: 0-{len(df)-1}")
        exp_hash = df.experiment_hash.iloc[idx]
    else:
        raise ValueError(
            "The way to choose an experiment (t) should be one of these: best, random, worst, median or a number."
        )
    model = th.load(os.path.join(exp_path, exp_hash, "model.pth"), weights_only=False)
    return model


class ArrayTensorEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, th.Tensor)):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
