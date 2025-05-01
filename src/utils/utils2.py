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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# from models.ResNet import ResNet
from ..models.FCN import FCN, ElasticFCN, DilatedFCN, DynamicFCN

# from models.DisjointCNN import DisjointCNN
# from models.DWSCNN import DWSCNN, ElasticDWSCNN, DilatedDWSCNN
# from models.InceptionTime import InceptionModel, DilatedInceptionModel
# from models.TapNet import TapNet
# from models.ProtoPNet import ProtoPNet, ProtoPNetLearner
# from models.CorrectedProtoPNet import CorrectedProtoPNet, CorrectedProtoPNetLearner
# from models.PIPNet import PIPNet, get_optimizer_nn, PIPNetLearner
# from models.ProtoPool import ProtoPool, ProtoPoolLearner
# from models.ProtoPoolMod import ProtoPoolLearnerMod
# from models.MultiScaleProto import MultiScaleProto, MultiScaleClassProto, MultiScaleProtoMoE, MultiScaleProtoMoETime, MSConvProtoLearner
# from models.MultiScaleProto import MultiScaleProtoParts
# from models.ConvTran import ConvTran, ConvTranLearner, RAdam
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


def model_selector(
    dataset, in_channels, ts_len, n_classes, params, model_type, results_path
):
    # Model trainer
    if "weight_decay" in params:
        weight_decay = params["weight_decay"]
    else:
        weight_decay = 0

    # Select criterion
    try:
        if params["criterion"] == "NLL":
            criterion = nn.NLLLoss(reduction="mean")
        elif params["criterion"] == "CE":
            criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError("Not valid criterion")
    except KeyError:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    # Create model and learner
    if model_type == "FCN":
        model = FCN(
            in_channels=in_channels,
            channels=params["channels"],
            kernel_sizes=params["kernel_sizes"],
            num_classes=n_classes,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=params["lr_patience"], factor=0.5
        )
        trainer = BasicLearner(
            model,
            criterion,
            num_epochs=params["epochs"],
            es_patience=params["es_patience"],
        )
    # elif model_type == "ResNet":
    #     model = ResNet(
    #         in_channels=in_channels, mid_channels=params["mid_channels"], kernel_sizes=params["kernel_sizes"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "InceptionTime":
    #     model = InceptionModel(
    #         depth=params["depth"],
    #         in_channels=in_channels, out_channels=params["out_channels"],
    #         bottleneck_channels=params["bottleneck_channels"], kernel_sizes=params["kernel_sizes"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "TapNet":
    #     model = TapNet(
    #         nfeat=in_channels, len_ts=ts_len, nclass=n_classes, dropout=params["dropout"],
    #         filters=params["filters"], kernels=params["kernel_sizes"], dilation=params["dilation"],
    #         layers=params["layers"],
    #         use_rp=params["use_rp"], rp_params=params["rp_params"],
    #         use_att=params["use_att"], use_ss=False, use_metric=False, use_muse=False,
    #         use_lstm=params["use_lstm"], use_cnn=params["use_cnn"]
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     if n_classes == 2:
    #         criterion = nn.BCELoss(reduction="mean")
    #     else:
    #         criterion = nn.CrossEntropyLoss(reduction="mean")
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "ProtoPNet":
    #     # Load Backbone
    #     model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    #     pretrained_model = infer_model_folder(
    #         dataset, ts_len, in_channels,
    #         scaling=params["scaling"], seed=params["seed"],
    #         model_type=params["back_bone_type"], model_params=model_params)
    #     back_bone = pretrained_model.back_bone
    #     model = ProtoPNet(
    #         ts_len=ts_len, back_bone=back_bone,
    #         n_prototypes_per_class=params["n_prototypes_per_class"],
    #         prototype_activation_function=params["prototype_activation_function"],
    #         prototype_dim=params["prototype_dim"],
    #         num_classes=n_classes
    #     )
    #     # Create optimizer and schedulers
    #     lr_features_addon_prototype_last = params["lr_features_addon_prototype_last"]
    #     lr_features = lr_features_addon_prototype_last[0]
    #     lr_add_on_layers = lr_features_addon_prototype_last[1]
    #     lr_prototype_vectors = lr_features_addon_prototype_last[2]
    #     lr_last_layer = lr_features_addon_prototype_last[3]
    #     joint_optimizer_specs = [
    #         {'params': model.back_bone.parameters(), 'lr': lr_features, 'weight_decay': weight_decay},
    #         {'params': model.add_on_layers.parameters(), 'lr': lr_add_on_layers, 'weight_decay': weight_decay},
    #         {'params': model.prototype_vectors, 'lr': lr_prototype_vectors},
    #     ]
    #     joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    #     joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params["lr_joint_step"], gamma=0.1)

    #     last_layer_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': lr_last_layer}]
    #     last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    #     last_layer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         last_layer_optimizer, 'min',
    #         patience=params["lr_patience_last_layer"],
    #         factor=0.5
    #     )
    #     optimizer = {"joint": joint_optimizer, "last_layer": last_layer_optimizer}
    #     scheduler = {"joint": joint_lr_scheduler, "last_layer": last_layer_scheduler}

    #     coefs = {"crs_ent": params["coefs"][0], "clst": params["coefs"][1], "sep": params["coefs"][2], "l1": params["coefs"][3]}
    #     push_interval = params["push_interval_last_layer_epochs"][0]
    #     last_layer_epochs_per_push = params["push_interval_last_layer_epochs"][1]
    #     trainer = ProtoPNetLearner(
    #         model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False,
    #         class_specific=params["class_specific"], coefs=coefs, push_start=params["push_start"],
    #         push_interval=push_interval, last_layer_epochs_per_push=last_layer_epochs_per_push,
    #     )
    # elif model_type == "CorrectedProtoPNet":
    #     back_bone = load_back_bone(params["back_bone_type"], params, dataset)
    #     model = CorrectedProtoPNet(
    #         ts_len=ts_len, back_bone=back_bone,
    #         n_prototypes_per_class=params["n_prototypes_per_class"],
    #         prototype_activation_function=params["prototype_activation_function"],
    #         prototype_dim=params["prototype_dim"],
    #         num_classes=n_classes
    #     )
    #     # Create optimizer and schedulers
    #     lr_features_addon_prototype_actweight = params["lr_features_addon_prototype_actweight"]
    #     lr_features = lr_features_addon_prototype_actweight[0]
    #     lr_add_on_layers = lr_features_addon_prototype_actweight[1]
    #     lr_prototype_vectors = lr_features_addon_prototype_actweight[2]
    #     lr_activation_weight = lr_features_addon_prototype_actweight[3]
    #     joint_optimizer_specs = [
    #         {'params': model.back_bone.parameters(), 'lr': lr_features, 'weight_decay': weight_decay},
    #         {'params': model.add_on_layers.parameters(), 'lr': lr_add_on_layers, 'weight_decay': weight_decay},
    #         {'params': model.prototype_vectors, 'lr': lr_prototype_vectors},
    #         {'params': model.activation_weight, 'lr': lr_activation_weight},
    #     ]
    #     joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    #     joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params["lr_joint_step"], gamma=0.1)

    #     optimizer = {"joint": joint_optimizer}
    #     scheduler = {"joint": joint_lr_scheduler}

    #     coefs = {"crs_ent": params["coefs"][0], "orth": params["coefs"][1], "clst": params["coefs"][2], "sep": params["coefs"][3]}
    #     trainer = CorrectedProtoPNetLearner(
    #         model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False,
    #         coefs=coefs,
    #     )
    # elif model_type == "ProtoPool":
    #     # Load Backbone
    #     model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    #     pretrained_model = infer_model_folder(
    #         dataset, ts_len, in_channels,
    #         scaling=params["scaling"], seed=params["seed"],
    #         model_type=params["back_bone_type"], model_params=model_params)
    #     back_bone = pretrained_model.back_bone
    #     model = ProtoPool(
    #         ts_len, back_bone,
    #         num_prototypes=params["num_prototypes"], num_descriptive=params["num_descriptive"],
    #         proto_depth=params["proto_depth"], prototype_activation_function=params["prototype_activation_function"],
    #         use_last_layer=params["use_last_layer"],
    #         num_classes=n_classes
    #     )

    #     joint_optimizer = torch.optim.Adam(
    #         [{'params': model.features.parameters(), 'lr': params["lr"] / 10, 'weight_decay': weight_decay},
    #          {'params': model.add_on_layers.parameters(), 'lr': 3 * params["lr"], 'weight_decay': weight_decay},
    #          {'params': model.proto_presence, 'lr': 3 * params["lr"]},
    #          {'params': model.prototype_vectors, 'lr': 3 * params["lr"]}]
    #     )
    #     joint_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         joint_optimizer, 'min', patience=params["lr_patience_joint"], factor=0.5)
    #     push_optimizer = torch.optim.Adam(
    #         [{'params': model.last_layer.parameters(), 'lr': params["lr"] / 10, 'weight_decay': weight_decay}])
    #     push_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         push_optimizer, 'min', patience=params["lr_patience_last_layer"], factor=0.5)
    #     optimizer = {"joint": joint_optimizer, "push": push_optimizer}
    #     scheduler = {"joint": joint_scheduler, "push": push_scheduler}

    #     trainer = ProtoPoolLearner(
    #         model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False,
    #         start_val=params["start_val"], end_val=params["end_val"],
    #         gumbel_time=params["gumbel_time"], pp_gumbel=params["pp_gumbel"],
    #         class_specific=params["class_specific"], mixup_data_bool=params["mixup_data_bool"], criterion=criterion,
    #         pp_ortho=params["pp_ortho"], cls_weight=params["cls_weight"], sep_weight=params["sep_weight"],
    #         proto_es_patience=params["proto_es_patience"], fine_tuning_epochs=params["fine_tuning_epochs"]
    #     )
    # elif model_type == "ProtoPoolMod":
    #     # Load Backbone
    #     model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    #     pretrained_model = infer_model_folder(
    #         dataset, ts_len, in_channels,
    #         scaling=params["scaling"], seed=params["seed"],
    #         model_type=params["back_bone_type"], model_params=model_params)
    #     back_bone = pretrained_model.back_bone
    #     model = ProtoPool(
    #         ts_len, back_bone,
    #         num_prototypes=params["num_prototypes"], num_descriptive=params["num_descriptive"],
    #         proto_depth=params["proto_depth"], prototype_activation_function=params["prototype_activation_function"],
    #         use_last_layer=params["use_last_layer"],
    #         num_classes=n_classes
    #     )

    #     lr_features_addon_prototype_last = params["lr_features_addon_prototype_last"]
    #     lr_features = lr_features_addon_prototype_last[0]
    #     lr_add_on_layers = lr_features_addon_prototype_last[1]
    #     lr_prototype_vectors = lr_features_addon_prototype_last[2]
    #     lr_last_layer = lr_features_addon_prototype_last[3]
    #     joint_optimizer = torch.optim.Adam(
    #         [{'params': model.features.parameters(), 'lr': lr_features, 'weight_decay': weight_decay},
    #          {'params': model.add_on_layers.parameters(), 'lr': lr_add_on_layers, 'weight_decay': weight_decay},
    #          {'params': model.proto_presence, 'lr': lr_prototype_vectors},
    #          {'params': model.prototype_vectors, 'lr': lr_prototype_vectors}]
    #     )
    #     joint_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params["lr_joint_step"], gamma=0.1)

    #     push_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': lr_last_layer}]
    #     push_optimizer = torch.optim.Adam(push_optimizer_specs)
    #     push_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         push_optimizer, 'min',
    #         patience=params["lr_patience_last_layer"],
    #         factor=0.5
    #     )
    #     optimizer = {"joint": joint_optimizer, "push": push_optimizer}
    #     scheduler = {"joint": joint_scheduler, "push": push_scheduler}

    #     push_interval = params["push_interval_last_layer_epochs"][0]
    #     last_layer_epochs_per_push = params["push_interval_last_layer_epochs"][1]
    #     trainer = ProtoPoolLearnerMod(
    #         model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False,
    #         start_val=params["start_val"], end_val=params["end_val"],
    #         gumbel_time=params["gumbel_time"], pp_gumbel=params["pp_gumbel"],
    #         class_specific=params["class_specific"], mixup_data_bool=params["mixup_data_bool"], criterion=criterion,
    #         pp_ortho=params["pp_ortho"], cls_weight=params["cls_weight"], sep_weight=params["sep_weight"],
    #         push_start=params["push_start"],
    #         push_interval=push_interval, last_layer_epochs_per_push=last_layer_epochs_per_push,
    #     )
    # elif model_type == "PIPNet":
    #     # Load Backbone
    #     model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    #     pretrained_model = infer_model_folder(
    #         dataset, ts_len, in_channels,
    #         scaling=params["scaling"], seed=params["seed"],
    #         model_type=params["back_bone_type"], model_params=model_params)
    #     back_bone = pretrained_model.back_bone
    #     model = PIPNet(
    #         ts_len=ts_len,
    #         back_bone=back_bone, num_prototypes=params["num_prototypes"], cls_bias=params["cls_bias"],
    #         num_classes=n_classes
    #     )

    #     # Optimizer and scheduler are create inside fit method for PIPNet
    #     lr_last_block_net = params["lr_last_block_net"]
    #     lr = lr_last_block_net[0]
    #     lr_block = lr_last_block_net[1]
    #     lr_net = lr_last_block_net[2]
    #     if params["lr_scheduler"] == "cosine":
    #         optimizer, scheduler = None, None
    #     elif params["lr_scheduler"] == "val_loss":
    #         optimizer_net, optimizer_classifier, _, _, _ = get_optimizer_nn(
    #             model, lr_net, lr_block, lr, params["weight_decay"], params["cls_bias"])
    #         scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer_net, 'min', patience=params["lr_patience"], factor=0.5)
    #         scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer_net, 'min', patience=params["lr_patience"], factor=0.5)
    #         optimizer = {"net": optimizer_net, "classifier": optimizer_classifier}
    #         scheduler = {"net": scheduler_net, "classifier": scheduler_classifier}
    #     else:
    #         raise ValueError("Not valid lr_scheduler")

    #     # Define trainer
    #     trainer = PIPNetLearner(
    #         model,  criterion,
    #         num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False,
    #         epochs_pretrain=params["epochs_pretrain"], epochs_to_finetune=params["epochs_to_finetune"], freeze_epochs=params["freeze_epochs"],
    #         lr_net=lr_net, lr_block=lr_block, lr=lr,
    #         weight_decay=weight_decay, cls_bias=params["cls_bias"],
    #         data_augmentation=params["data_augmentation"]
    #     )
    elif model_type == "ElasticFCN":
        model = ElasticFCN(
            in_channels=in_channels,
            channels=params["channels"],
            kernel_sizes=params["kernel_sizes"],
            kernel_scales=params["kernel_scales"],
            pool_type=params["pool_type"],
            num_classes=n_classes,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=params["lr_patience"], factor=0.5
        )
        trainer = BasicLearner(
            model,
            criterion,
            num_epochs=params["epochs"],
            es_patience=params["es_patience"],
        )
    elif model_type == "DilatedFCN":
        model = DilatedFCN(
            in_channels=in_channels,
            channels=params["channels"],
            kernel_sizes=params["kernel_sizes"],
            dilations=params["dilations"],
            pool_type=params["pool_type"],
            num_classes=n_classes,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=params["lr_patience"], factor=0.5
        )
        trainer = BasicLearner(
            model,
            criterion,
            num_epochs=params["epochs"],
            es_patience=params["es_patience"],
        )
    # elif model_type == "DisjointCNN":
    #     model = DisjointCNN(
    #         in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    elif model_type == "DynamicFCN":
        model = DynamicFCN(
            in_channels=in_channels,
            channels=params["channels"],
            kernel_sizes=params["kernel_sizes"],
            num_classes=n_classes,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=params["lr_patience"], factor=0.5
        )
        trainer = BasicLearner(
            model,
            criterion,
            num_epochs=params["epochs"],
            es_patience=params["es_patience"],
        )
    # elif model_type == "DWSCNN":
    #     model = DWSCNN(
    #         in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
    #         include_bn_relu=params["include_bn_relu"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "ElasticDWSCNN":
    #     model = ElasticDWSCNN(
    #         in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
    #         include_bn_relu=params["include_bn_relu"],
    #         kernel_scales=params["kernel_scales"], pool_type=params["pool_type"], num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "DilatedDWSCNN":
    #     model = DilatedDWSCNN(
    #         in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
    #         include_bn_relu=params["include_bn_relu"],
    #         dilations=params["dilations"], pool_type=params["pool_type"], num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "DilatedInception":
    #     out_channels = params["out_channels_bottleneck_channels"][0]
    #     bottleneck_channels = params["out_channels_bottleneck_channels"][1]
    #     model = DilatedInceptionModel(
    #         depth=params["depth"],
    #         in_channels=in_channels, out_channels=out_channels,
    #         bottleneck_channels=bottleneck_channels, kernel_sizes=params["kernel_sizes"],
    #         dilations=params["dilations"], pool_type=params["pool_type"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #     trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    # elif model_type == "MSProto":
    #     # Load Backbone
    #     back_bone = load_back_bone(params["back_bone_type"], params, dataset)
    #     model = MultiScaleProto(
    #         ts_len, back_bone,
    #         params["num_prototypes_per_layer"], params["activation"], params["ms_activation"],
    #         params["entropy_temp_weighting"],
    #         max_pool_time=params["pre_att_max_pool_time"], embedding_dim=params["pre_att_embedding_dim"],
    #         att_type=params["att_type"], att_dropout=params["att_dropout"],
    #         final_pool_op=params["final_pool_op"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)

    #     plot_proto_info = True if params["seed"] == 0 else False
    #     trainer = MSConvProtoLearner(
    #         model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False, k=3, l1_mult=params["l1_mult"],
    #         plot_proto_info=False, save_dir=results_path
    #     )
    # elif model_type == "MSClassProto":
    #     # Load Backbone
    #     back_bone = load_back_bone(params["back_bone_type"], params, dataset)
    #     model = MultiScaleClassProto(
    #         ts_len, back_bone,
    #         params["num_prototypes_per_class_layer"], params["activation"], params["ms_activation"],
    #         params["entropy_temp_weighting"],
    #         max_pool_time=params["pre_att_max_pool_time"], embedding_dim=params["pre_att_embedding_dim"],
    #         att_type=params["att_type"], att_dropout=params["att_dropout"],
    #         final_pool_op=params["final_pool_op"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)

    #     plot_proto_info = True if params["seed"] == 0 else False
    #     trainer = MSConvProtoLearner(
    #         model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False, k=3, l1_mult=None,
    #         plot_proto_info=False, save_dir=results_path
    #     )
    # elif model_type == "MSProtoMoE":
    #     # Load Backbone
    #     back_bone = load_back_bone(params["back_bone_type"], params, dataset)
    #     model = MultiScaleProtoMoE(
    #         ts_len, back_bone,
    #         params["num_prototypes_per_layer"], params["activation"], params["ms_activation"],
    #         params["entropy_temp_weighting"],
    #         num_experts=n_classes, num_selected_protos=params["num_selected_protos"], top_k=params["top_k"],
    #         max_pool_time=params["pre_att_max_pool_time"], embedding_dim=params["pre_att_embedding_dim"],
    #         att_type=params["att_type"], att_dropout=params["att_dropout"],
    #         final_pool_op=params["final_pool_op"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"],
    #                                                            factor=0.5)

    #     plot_proto_info = True if params["seed"] == 0 else False
    #     trainer = MSConvProtoLearner(
    #         model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False, k=3, l1_mult=None,
    #         plot_proto_info=False, save_dir=results_path
    #     )
    # elif model_type == "MSProtoMoETime":
    #     # Load Backbone
    #     back_bone = load_back_bone(params["back_bone_type"], params, dataset)
    #     model = MultiScaleProtoMoETime(
    #         ts_len, back_bone,
    #         params["num_prototypes_per_layer"], params["activation"], params["ms_activation"],
    #         params["entropy_temp_weighting"],
    #         num_experts=n_classes, num_selected_protos=params["num_selected_protos"], top_k=params["top_k"],
    #         max_pool_time=params["pre_att_max_pool_time"], embedding_dim=params["pre_att_embedding_dim"],
    #         att_type=params["att_type"], att_dropout=params["att_dropout"],
    #         final_pool_op=params["final_pool_op"],
    #         num_classes=n_classes
    #     )
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"],
    #                                                            factor=0.5)

    #     plot_proto_info = True if params["seed"] == 0 else False
    #     trainer = MSConvProtoLearner(
    #         model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False, k=3, l1_mult=None,
    #         plot_proto_info=False, save_dir=results_path
    #     )
    # elif model_type == "MSProtoParts":
    #     # Load Backbone
    #     model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    #     pretrained_model = infer_model_folder(
    #         dataset, ts_len, in_channels,
    #         scaling=params["scaling"], seed=params["seed"],
    #         model_type=params["back_bone_type"], model_params=model_params)
    #     back_bone = pretrained_model.back_bone
    #     model = MultiScaleProtoParts(
    #         ts_len, back_bone,
    #         n_prototypes_per_layer_class=params["n_prototypes_per_layer_class"],
    #         prototype_activation_function=params["prototype_activation_function"],
    #         prototype_dim=params["prototype_dim"],
    #         num_classes=n_classes
    #     )
    #     if params["learner"] == "basic":
    #         optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
    #         trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    #     elif params["learner"] == "protop":
    #         lr_features_addon_prototype_last = params["lr_prototype_last"]
    #         lr_prototypes = lr_features_addon_prototype_last[0]
    #         lr_last_layer = lr_features_addon_prototype_last[1]
    #         joint_optimizer_specs = [
    #             {'params': model.prototype_blocks.parameters(), 'lr': lr_prototypes},
    #         ]
    #         joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    #         joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params["lr_joint_step"],
    #                                                              gamma=0.1)

    #         last_layer_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': lr_last_layer}]
    #         last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    #         last_layer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             last_layer_optimizer, 'min',
    #             patience=params["lr_patience_last_layer"],
    #             factor=0.5
    #         )
    #         optimizer = {"joint": joint_optimizer, "last_layer": last_layer_optimizer}
    #         scheduler = {"joint": joint_lr_scheduler, "last_layer": last_layer_scheduler}

    #         coefs = {"crs_ent": params["coefs"][0], "clst": params["coefs"][1], "sep": params["coefs"][2], "l1": params["coefs"][3]}
    #         push_interval = params["push_interval_last_layer_epochs"][0]
    #         last_layer_epochs_per_push = params["push_interval_last_layer_epochs"][1]
    #         trainer = ProtoPNetLearner(
    #             model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #             include_labels_in_forward=False,
    #             class_specific=params["class_specific"], coefs=coefs, push_start=params["push_start"],
    #             push_interval=push_interval, last_layer_epochs_per_push=last_layer_epochs_per_push,
    #         )

    #     else:
    #         raise NotImplementedError
    # elif model_type == "ConvTran":
    #     model = ConvTran(
    #         in_channels=in_channels, ts_len=ts_len,
    #         kernel_size=params["kernel_size"], emb_size=params["emb_size"], num_heads=params["num_heads"],
    #         dim_ff=params["dim_ff"], dropout=params["dropout"],
    #         num_classes=n_classes
    #     )
    #     optimizer = RAdam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    #     scheduler = None

    #     trainer = ConvTranLearner(
    #         model, num_epochs=params["epochs"], es_patience=params["es_patience"],
    #         include_labels_in_forward=False, l2_reg=params["l2_reg"]
    #     )
    else:
        raise ValueError("Not valid model type")

    return model, optimizer, scheduler, trainer


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
    torch.save(model, f"{results_path}/model.pth")

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
