import os
import json
import re
import itertools
import torch as th
import typing as tp
import numpy as np
from torch.nn import functional


def compute_cfe(mask: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """
    Calculates the Counterfactual Explanation (CFE) of a signal based on the mask and another signal.

    :param `mask`: Boolean array used to select between elements of x1 and x2
    :param `x1`: Array of values to select when the mask is False
    :param `x2`: Array of values to select when the mask is True
    :return: Array with elements selected from x1 or x2 based on the mask
    """
    return np.where(mask, x2, x1)


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
    Calculates the number of subsequences in a mask (how many times the mask changes from 0 to 1).

    :param `mask`: Numpy array representing the mask
    :return: Number of subsequences
    """
    return np.count_nonzero(np.diff(mask, prepend=0, axis=1) == 1, axis=(0, 1))


def find_diferences(mask, x1, x2):
    """
    Find the positions where the mask changes from 0 to 1 or from 1 to 0 and calculate the distances between the CFE and the other signal.

    :param `mask`: Boolean array used to select between elements of x1 and x2.
    :param `x1`: Array of values to select when the mask is False.
    :param `x2`: Array of values to select when the mask is True.
    :return `diferences`: List of differences between the CFE and the corresponding values in x1 or x2 at positions where changes occur in the mask.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    cfe = compute_cfe(mask, x1, x2)

    changes = np.diff(mask, prepend=mask[:, [0]], axis=1)
    ids_changes = np.where(changes != 0)
    ids_changes = [tuple(ids) for ids in zip(*ids_changes)]

    diferences = list()
    for pos in ids_changes:
        value = x1[pos] if mask[pos] == 1 else x2[pos]
        dif = np.linalg.norm(cfe[pos] - value)
        diferences.append(dif)

    return diferences


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


def l2_norm(x1: tp.Union[np.ndarray, list], x2: tp.Union[np.ndarray, list]) -> float:
    """
    Calculates the L2 norm between two arrays.

    :param `x1`: First array
    :param `x2`: Second array
    :return: L2 norm between the two arrays
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    return np.linalg.norm(x1 - x2, ord=2)


def plot_signal(X, X2, mask, ax, title=None):
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

    if X.shape != X2.shape:
        raise ValueError(
            f"`X` and `X2` must have the same size, but got {X.shape} and {X2.shape}"
        )
    if X.shape != mask.shape:
        raise ValueError(
            f"`X` and `mask` must have the same size, but got {X.shape} and {mask.shape}"
        )
    try:
        mask = mask.astype(np.bool_)
    except ValueError:
        raise TypeError("`mask` must be a list of binary values (0/1 or True/False)")

    mod = X.copy()
    mod[mask] = X2[mask]
    ax.clear()
    ax.plot(X2_flat, c="k", label="NUN")
    ax.plot(X_flat, c="b", label="Original")
    ax.plot(mod.flatten(), c="r", label="CFE")

    submasks = extract_submasks(mask)
    for subm in submasks:
        dim, x, y = subm  # In the future we will have more dimensions
        ax.axvspan(
            x - (1 if x > 0 else 0),
            y - (1 if y >= len(X) else 0),
            color="red",
            alpha=0.1,
        )

    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_title(f"CFE{f' - {title}' if title else ''}", fontsize=14, fontweight="bold")


def extract_submasks(mask):
    """
    Identifies contiguous regions where the mask is True and returns their dimension, start and end indices.

    :param `mask`: Boolean mask indicating the regions of interest
    :return: List of tuples (dimension_idx, start_idx, end_idx) representing the masked regions
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


def load_model(dataset, experiment, mode="best"):
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
    if mode == "best":
        exp_hash = df.experiment_hash.iloc[0]
    elif mode == "random":
        exp_hash = df.experiment_hash.sample(n=1).iloc[0]
    elif mode == "worst":
        exp_hash = df.experiment_hash.iloc[-1]
    elif mode == "median":
        exp_hash = df.experiment_hash.iloc[len(df) // 2]
    elif mode.isdigit():
        idx = int(mode)
        if idx < 0 or idx >= len(df):
            raise ValueError(f"Index {idx} out of range. Valid range: 0-{len(df)-1}")
        exp_hash = df.experiment_hash.iloc[idx]
    else:
        raise ValueError(
            "The way to choose an experiment (t) should be one of these: best, random, worst, median or a number."
        )
    model = th.load(os.path.join(exp_path, exp_hash, "model.pth"), weights_only=False)
    return model


def load_json_params(file_path: str) -> dict:
    """
    Loads a JSON file into a dictionary, ignoring lines with '//' comments.

    :param `file_path`: Path to the JSON file
    :return: Dictionary with parameters
    """
    with open(file_path, "r") as f:
        content = f.readlines()

    # Remove lines with comments
    cleaned = []
    for line in content:
        if "//" in line:
            line = re.sub(r"(?<!http:)//.*", "", line)
        if line.strip():
            cleaned.append(line)

    json_str = "".join(cleaned)
    return json.loads(json_str)


def generate_param_combinations(
    config: tp.Dict[str, tp.Any],
) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Generates all combinations of parameter configurations to run experiments.

    :param `config`: Dictionary with 'static' and 'grid' fields
    :return `all_configs`: List of parameter dictionaries
    """
    static_params = config.get("static", {})
    grid_params = config.get("grid", {})

    if not grid_params:
        return [static_params]

    keys, values = zip(*grid_params.items())
    combinations = list(itertools.product(*values))

    all_configs = []
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        full_config = {**static_params, **combo_dict}
        all_configs.append(full_config)

    return all_configs


class ArrayTensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, th.Tensor)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
