import os
import random
import json
import pickle
import itertools
import hashlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.datasets import UCR_UEA_datasets

from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
)

import torch
import tensorflow as tf
from torch import nn, load
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..models.FCN import FCN, ElasticFCN, DilatedFCN, DynamicFCN
from ..models.learners import BasicLearner
from torchsummary import summary


def get_subsample(X_test, y_test, n_instances, seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    subset_idx = np.random.choice(len(X_test), n_instances, replace=False)
    subset_idx = np.sort(subset_idx)
    X_test = X_test[subset_idx]
    y_test = y_test[subset_idx]
    return X_test, y_test, subset_idx


def get_hash_from_params(params):
    params_str = "".join(f"{key}={value}," for key, value in sorted(params.items()))
    params_hash = hashlib.sha1(params_str.encode()).hexdigest()
    return params_hash


def generate_settings_combinations(original_dict):
    # Create a list of keys with lists as values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    # Generate all possible combinations
    combinations = list(itertools.product(*[original_dict[key] for key in list_keys]))
    # Create a set of experiments dictionaries with unique combinations
    result = {}
    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        experiment_hash = get_hash_from_params(new_dict)
        result[experiment_hash] = new_dict
    return result


def load_parameters_from_json(json_filename):
    with open(json_filename, "r") as json_file:
        params = json.load(json_file)
    return params


def min_max_scale_data(X_train, X_test):
    max = 1
    min = 0
    """maximums = X_train.max(axis=(0, 1))
    minimums = X_train.min(axis=(0, 1))"""
    data_max = X_train.max()
    data_min = X_train.min()

    # Min Max scale data between 0 and 1
    X_train_scaled = (X_train - data_min) / (data_max - data_min)
    X_train_scaled = X_train_scaled * (max - min) + min

    X_test_scaled = (X_test - data_min) / (data_max - data_min)
    X_test_scaled = X_test_scaled * (max - min) + min

    return X_train_scaled, X_test_scaled


def standard_scale_data(X_train, X_test):
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test


def ucr_data_loader(dataset, scaling, backend="torch", store_path="../../data/UCR"):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    if X_train is not None:
        np.save(f"{store_path}/{dataset}/X_train.npy", X_train)
        np.save(f"{store_path}/{dataset}/X_test.npy", X_test)
        np.save(f"{store_path}/{dataset}/y_train.npy", y_train)
        np.save(f"{store_path}/{dataset}/y_test.npy", y_test)

        # Scaling
        if scaling == "min_max":
            X_train, X_test = min_max_scale_data(X_train, X_test)
        elif scaling == "standard":
            X_train, X_test = standard_scale_data(X_train, X_test)
        elif scaling == "none":
            pass
        else:
            raise ValueError("Not valid scaling value")

        # Backend
        if backend == "torch":
            X_train = X_train.transpose(0, 2, 1)
            X_test = X_test.transpose(0, 2, 1)
        elif backend == "tf":
            pass
        else:
            raise ValueError("backend not valid. Choose torch or tf")
    return X_train, y_train, X_test, y_test


def local_data_loader(dataset, scaling, backend="torch", data_path="../../data"):
    X_train = np.load(f"{data_path}/UCR/{dataset}/X_train.npy", allow_pickle=True)
    X_test = np.load(f"{data_path}/UCR/{dataset}/X_test.npy", allow_pickle=True)
    y_train = np.load(f"{data_path}/UCR/{dataset}/y_train.npy", allow_pickle=True)
    y_test = np.load(f"{data_path}/UCR/{dataset}/y_test.npy", allow_pickle=True)

    # Scaling
    if scaling == "min_max":
        X_train, X_test = min_max_scale_data(X_train, X_test)
    elif scaling == "standard":
        X_train, X_test = standard_scale_data(X_train, X_test)
    elif scaling == "none":
        pass
    else:
        raise ValueError("Not valid scaling value")

    # Backend
    if backend == "torch":
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
    elif backend == "tf":
        pass
    else:
        raise ValueError("backend not valid. Choose torch or tf")

    return X_train, y_train, X_test, y_test


def label_encoder(training_labels, testing_labels):
    # If label represent integers, try to cast it. If it is not possible, then resort to Label Encoding
    try:
        y_train = []
        for label in training_labels:
            y_train.append(int(float(label)))
        y_test = []
        for label in testing_labels:
            y_test.append(int(float(label)))

        # Check if labels are consecutive
        if sorted(y_train) == list(range(min(y_train), max(y_train) + 1)):
            # Add class 0 in case it does not exist
            y_train, y_test = np.array(y_train).reshape(-1, 1), np.array(
                y_test
            ).reshape(-1, 1)
            classes = np.unique(y_train)
            if 0 not in classes:
                y_train = y_train - 1
                y_test = y_test - 1
        else:
            # Raise exception so each class is treated as a category
            raise ValueError(
                "The classes can be casted to integers but they are non consecutive numbers. Treating them as categories"
            )

    except Exception:
        le = LabelEncoder()
        le.fit(np.concatenate((training_labels, testing_labels), axis=0))
        y_train = le.transform(training_labels)
        y_test = le.transform(testing_labels)
    return y_train, y_test


def one_hot_encoder(training_labels, testing_labels):
    # If label represent integers, try to cast it. If it is not possible, then resort to Label Encoding
    try:
        y_train = []
        for label in training_labels:
            y_train.append(int(float(label)))
        y_test = []
        for label in testing_labels:
            y_test.append(int(float(label)))

        # Check if labels are consecutive
        if sorted(y_train) == list(range(min(y_train), max(y_train) + 1)):
            # Add class 0 in case it does not exist
            y_train, y_test = np.array(y_train).reshape(-1, 1), np.array(
                y_test
            ).reshape(-1, 1)
            classes = np.unique(y_train)
            if 0 not in classes:
                y_train = y_train - 1
                y_test = y_test - 1
        else:
            # Raise exception so each class is treated as a category
            raise ValueError(
                "The classes can be casted to integers but they are non consecutive numbers. Treating them as categories"
            )

    except Exception:
        oh = OneHotEncoder(categories="auto", sparse_output=False)
        oh.fit(np.concatenate((training_labels, testing_labels), axis=0))
        y_train = oh.transform(training_labels)
        y_test = oh.transform(testing_labels)
    return y_train, y_test


def select_best_model(dataset, exp_name):
    experiment_folder = f"./models/{dataset}/{exp_name}"
    # Locate all experiment hashes for the given dataset by inspecting the folders
    experiment_sub_dirs = [
        f
        for f in os.listdir(experiment_folder)
        if os.path.isdir(os.path.join(experiment_folder, f))
    ]
    # Iterate through the combinations and retrieve the results file
    experiment_info_list = []
    train_params_set = set()
    for experiment_sub_dir in experiment_sub_dirs:
        results_path = f"{experiment_folder}/{experiment_sub_dir}"
        try:
            # Read the params file
            with open(f"{results_path}/train_params.json") as f:
                train_params = json.load(f)
            # Read the metrics file
            with open(f"{results_path}/metrics.json") as f:
                metrics = json.load(f)
            # Merge all info
            experiment_info = {**train_params, **metrics}
            experiment_info_list.append(experiment_info)
            train_params_set.update(list(train_params.keys()))
        except FileNotFoundError:
            print(f"Experiment {experiment_sub_dir} not saved.")

    # Create the a dataframe containing all info and store it
    all_results_df = pd.DataFrame.from_records(experiment_info_list)
    param_list = list(train_params_set)
    param_list.remove("seed")
    param_list.remove("experiment_hash")
    param_list.remove("total_params")
    for column in all_results_df.columns:
        if all_results_df[column].dtype == object:
            all_results_df[column] = all_results_df[column].astype(str)

    # Create averaged results across seeds
    results_mean = (
        all_results_df.drop("experiment_hash", axis=1).groupby(param_list).mean()
    )
    results_std = (
        all_results_df.drop("experiment_hash", axis=1).groupby(param_list).std()
    )
    results_counts = (
        all_results_df.drop("experiment_hash", axis=1).groupby(param_list).size()
    )
    average_results_df = pd.DataFrame(index=results_mean.index)
    average_results_df["total_params"] = results_mean["total_params"]
    for metric in metrics.keys():
        average_results_df[f"{metric}_mean"] = results_mean[metric]
        average_results_df[f"{metric}_std"] = results_std[metric]
    average_results_df["n_seeds"] = results_counts
    aux_seed_0 = (
        all_results_df[all_results_df["seed"] == 0][param_list + ["experiment_hash"]]
        .rename(columns={"experiment_hash": "seed_0_experiment_hash"})
        .set_index(param_list)
    )
    average_results_df = average_results_df.merge(
        aux_seed_0, how="left", left_index=True, right_index=True
    )

    experiment_results_df = all_results_df.sort_values("test_f1", ascending=False)
    experiment_results_df.to_excel(f"{experiment_folder}/all_results.xlsx")
    experiment_average_results_df = average_results_df.sort_values(
        "test_f1_mean", ascending=False
    )
    experiment_average_results_df.to_excel(
        f"{experiment_folder}/average_seed_results.xlsx"
    )


def model_selector(dataset, in_channels, ts_len, n_classes, params):
    # Model trainer
    if "weight_decay" in params:
        weight_decay = params["weight_decay"]
    else:
        weight_decay = 0

    # Select criterion
    try:
        if params["criterion"] == "NLL":
            criterion = nn.NLLLoss(reduction='mean')
        elif params["criterion"] == "CE":
            criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError("Not valid criterion")
    except KeyError:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    # Create model and learner
    model_type = params.get("model_type", "FCN")
    if model_type == "FCN":
        model = FCN(
            in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
            num_classes=n_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
        trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    else:
        raise ValueError("Not valid model type")

    return model, optimizer, scheduler, trainer

def load_model(
    dataset,
    experiment,
    in_channels,
    ts_len,
    n_classes,
    mode="best",
    device="cpu",
):
    """
    Loads a trained model from the specified experiment based on the chosen criteria.

    This function loads a model from the specified dataset and experiment folder.
    The experiment can be selected based on various criteria: 'best', 'random', 'worst', 'median', or an
    integer index to choose a specific experiment.

    :param `dataset`: The name of the dataset
    :param `experiment`: The name of the experiment folder within the dataset
    :param `mode`: The criteria for selecting the experiment. Can be 'best', 'random', 'worst', 'median'
                   or an integer index. Default is 'best'
    :param `device`: Where to load the model ("cpu" or "cuda"). Default is "cpu"
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

    model_folder = os.path.join(exp_path, exp_hash)
    with open(f"{model_folder}/train_params.json") as f:
        train_params = json.load(f)
    model, _, _, _ = model_selector(
        dataset, in_channels, ts_len, n_classes, train_params
    )
    model_weights = load(os.path.join(model_folder, "model.pth"), weights_only=True)
    model.load_state_dict(model_weights)
    # model_wrapper = ModelWrapper(model, "torch")
    # model_wrapper.to(device)
    return model


def load_back_bone(back_bone_type, params, dataset):
    model_params = {
        k.replace("back_bone_model_", ""): v
        for k, v in params.items()
        if "back_bone_model" in k
    }
    pretrained_model = infer_model_folder(
        dataset,
        scaling=params["scaling"],
        seed=params["seed"],
        model_type=back_bone_type,
        model_params=model_params,
    )
    back_bone = pretrained_model.back_bone
    return back_bone


def infer_model_folder(dataset, scaling, seed, model_type, model_params):
    results_path = f"./models/{dataset}/{model_type}"
    all_results_df = pd.read_excel(f"{results_path}/all_results.xlsx")

    # Track experiment
    filtered_results = all_results_df[
        (all_results_df["scaling"] == scaling) & (all_results_df["seed"] == seed)
    ]
    for param, value in model_params.items():
        filtered_results = filtered_results[filtered_results[param] == str(value)]
    experiment_hash = filtered_results.iloc[0]["experiment_hash"]
    # Load model
    # model_params = {**{"in_channels": in_channels}, **model_params}
    # model.load_state_dict(torch.load(f"{results_path}/{experiment_hash}/model.pth"))
    model = torch.load(f"{results_path}/{experiment_hash}/model.pth")
    return model


def train_experiment(dataset, exp_name, exp_hash, params, model_type):
    # Set seed
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])

    # Load data
    scaling = params["scaling"]
    if os.path.isdir(f"./data/UCR/{dataset}"):
        X_train, y_train, X_test, y_test = local_data_loader(
            str(dataset), scaling, backend="torch", data_path="./data"
        )
    else:
        os.makedirs(f"./data/UCR/{dataset}")
        X_train, y_train, X_test, y_test = ucr_data_loader(
            dataset, scaling, backend="torch", store_path="./data/UCR"
        )
        if X_train is None:
            raise ValueError(f"Dataset {dataset} could not be downloaded")
    y_train, y_test = label_encoder(y_train, y_test)
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_channels = X_train.shape[1]
    ts_length = X_train.shape[2]

    # y_train, y_test = one_hot_encoder(y_train, y_test)
    # n_classes = y_train.shape[1]

    # Create model folder if it does not exist
    results_path = f"./models/{dataset}/{exp_name}/{exp_hash}"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Define model architecture
    model, optimizer, scheduler, trainer = model_selector(
        dataset, n_channels, ts_length, n_classes, params, model_type, results_path
    )
    # summary(model.cuda(), X_train.shape[1:])

    # Model Fit
    epoch_metrics = trainer.fit(
        X_train,
        y_train,
        optimizer,
        scheduler,
        batch_size=params["batch_size"],
        val_size=0.1,
    )
    training_history = pd.DataFrame(epoch_metrics)

    # Evaluation
    y_train_probs = trainer.predict(X_train)
    predicted_train_labels = np.argmax(y_train_probs, axis=1)
    # print(classification_report(y_train, predicted_train_labels))
    y_test_probs = trainer.predict(X_test)
    predicted_test_labels = np.argmax(y_test_probs, axis=1)
    print(classification_report(y_test, predicted_test_labels))

    # Export model
    torch.save(model.state_dict(), f"{results_path}/model.pth")

    # Export training params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    params = {
        **{"experiment_hash": exp_hash, "total_params": pytorch_total_params / 1e6},
        **params,
    }
    with open(f"{results_path}/train_params.json", "w") as outfile:
        json.dump(params, outfile)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Export result metrics
    if n_classes > 2:
        train_roc_auc = roc_auc_score(
            y_train, y_train_probs, multi_class="ovr", average="macro"
        )
        test_roc_auc = roc_auc_score(
            y_test, y_test_probs, multi_class="ovr", average="macro"
        )
    else:
        train_roc_auc = roc_auc_score(y_train, y_train_probs[:, 1], average="macro")
        test_roc_auc = roc_auc_score(y_test, y_test_probs[:, 1], average="macro")
    train_f1 = f1_score(y_train, predicted_train_labels, average="weighted")
    test_f1 = f1_score(y_test, predicted_test_labels, average="weighted")
    train_acc = accuracy_score(y_train, predicted_train_labels)
    test_acc = accuracy_score(y_test, predicted_test_labels)
    result_metrics = {
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }
    with open(f"{results_path}/metrics.json", "w") as outfile:
        json.dump(result_metrics, outfile)

    # Export confusion matrix
    cm = confusion_matrix(y_test, predicted_test_labels)
    cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(12, 12))
    cmp.plot(ax=ax).figure_.savefig(f"{results_path}/confusion_matrix.png")

    # Export loss training loss evolution
    fig, (ax1, ax2) = plt.subplots(1, 2)
    training_history[["train_loss", "val_loss"]].plot(ax=ax1)
    # pd.DataFrame(training_history.history)[['acc', 'val_acc']].plot(ax=ax2)
    plt.savefig(f"{results_path}/loss_curve.png")

class ModelWrapper:
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend.lower()

        # Prepare for backend
        if self.backend == "torch":
            self.framework = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        elif self.backend == "tf":
            self.framework = tf
        else:
            raise ValueError("Unsupported backend: choose 'torch' or 'tf'.")

    def predict(self, x: np.ndarray, input_data_format="tf") -> np.ndarray:
        assert input_data_format in ["tf", "torch"]

        # Append
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)
        if self.backend == "torch":
            if input_data_format == "tf":
                # Swap axes: from (B, T, F) to (B, F, T)
                x = np.transpose(x, (0, 2, 1))
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
                output = torch.nn.functional.softmax(output, dim=1)
            return output.detach().cpu().numpy()

        elif self.backend == "tf":
            if input_data_format == "torch":
                # Swap axes: from (B, F, T) to (B, T, F)
                x = np.transpose(x, (0, 2, 1))
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            output = self.model.predict(x_tensor, verbose=0)
            return output
        
    def to(self, device):
        self.model = self.model.to(device)