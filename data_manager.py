import os
import typing as tp
import numpy as np
from utils import predict_proba

try:
    from torch import tensor, nn
except ImportError:
    tensor = None
try:
    from tensorflow import convert_to_tensor
except ImportError:
    convert_to_tensor = None


class DataManager:
    def __init__(
        self,
        dataset: str,
        model: nn.Module,
        scaling: str = "none",
        backend: str = "torch",
    ):
        self.X_train, self.y_train_true, self.X_test, self.y_test_true = (
            self.load_dataset(dataset, scaling, backend)
        )
        self.name = dataset
        self.model = model
        self.y_train_model = predict_proba(self.model, self.X_train)[1]
        self.y_test_model = predict_proba(self.model, self.X_test)[1]
        self.nuns_train = self.compute_nuns(train=True, preds=True)
        self.nuns_test = self.compute_nuns(train=False, preds=True)

    def load_dataset(
        self, dataset: str, scaling: str = "none", backend: str = "torch"
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the dataset from the specified path and applies optional scaling and backend handling.

        :param `dataset`: Name of the dataset. The files must follow the path `./data/<dataset>`
        :param `scaling`: Scaling method ('min_max', 'standard', 'none'), default is 'none'
        :param `backend`: Backend to format the data ('torch' or 'tf'), default is 'torch'
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
            data[file.split(".")[0]] = np.load(file_path, allow_pickle=True)

        X_train, y_train, X_test, y_test = (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
        )

        if scaling == "min_max":
            X_train, X_test = self.min_max_scale_data(X_train, X_test)
        elif scaling == "standard":
            X_train, X_test = self.standard_scale_data(X_train, X_test)
        elif scaling != "none":
            raise ValueError(
                "Invalid scaling method. Choose 'min_max', 'standard', or 'none'."
            )

        if backend == "torch":
            X_train = tensor(X_train).permute(0, 2, 1)
            X_train = X_train.float()
            X_test = tensor(X_test).permute(0, 2, 1)
            X_test = X_test.float()
        elif backend == "tf":
            X_train = convert_to_tensor(X_train)
            X_test = convert_to_tensor(X_test)
        else:
            raise ValueError("Invalid backend. Choose 'torch' or 'tf'.")

        return X_train, y_train, X_test, y_test

    def min_max_scale_data(self, X_train, X_test):
        """
        Scales the training and test data using Min-Max normalization.

        :param `X_train`: Training data to be scaled
        :param `X_test`: Test data to be scaled using training data parameters
        :return: Tuple of scaled training and test data (X_train_scaled, X_test_scaled)
        """
        min_val, max_val = 0, 1
        data_max = X_train.max()
        data_min = X_train.min()

        X_train_scaled = (X_train - data_min) / (data_max - data_min)
        X_train_scaled = X_train_scaled * (max_val - min_val) + min_val

        X_test_scaled = (X_test - data_min) / (data_max - data_min)
        X_test_scaled = X_test_scaled * (max_val - min_val) + min_val

        return X_train_scaled, X_test_scaled

    def standard_scale_data(self, X_train, X_test):
        """
        Scales the training and test data using standardization (Z-score).

        :param `X_train`: Training data to be standardized
        :param `X_test`: Test data to be standardized using training data parameters
        :return `X_train`, `X_test`: Tuple of standardized training and test data (X_train, X_test)
        """
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()

        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

        return X_train, X_test

    def compute_nuns(self, train=True, preds=False):
        """
        Computes the Nearest Unlike Neighbors (NUNs) for each sample in the dataset.
        For each sample, finds the closest samples with a different label and returns them in an ordered list based on their distance.

        :param `train`: If True, computes NUNs for training data, otherwise for test data, default is True
        :param `preds`: If True, uses model predictions as labels, otherwise uses true labels, default is False
        :return `nuns`: A dictionary mapping each sample index to its ordered list of NUNs
        """
        data = self.X_train if train else self.X_test

        if train:
            labels = self.y_train_model if preds else self.y_train_true
        else:
            labels = self.y_test_model if preds else self.y_test_true

        nuns = dict()
        for i, sample in enumerate(data):
            sample_label = (
                self.get_true_label(sample, True if train else False)
                if not preds
                else self.get_predicted_label(sample, True if train else False)
            )
            unlike_indices = np.where(labels != sample_label)[0]
            distances = np.linalg.norm(data[unlike_indices] - sample, ord=2, axis=2)
            sorted_idxs = unlike_indices[np.argsort(distances, axis=0)[::-1]].flatten()
            nuns[i] = data[sorted_idxs]

        return nuns

    def get_sample(self, label=1, index=None, test=False, failed=False):
        """
        Gets a sample of the data belonging to the class specified by label.

        :param `label`: The label of the class from which a sample is to be obtained, default is 1
        :param `index`: The specific index of the sample to retrieve. If None, a random sample is returned, default is None
        :param `test`: If True, gets the sample from the test data, otherwise from training data, default is False
        :param `failed`: Parameter is not used in the current implementation
        :return `sample`: A sample of the specified class (converted to a NumPy Array)
        :raises `ValueError`: If there is no data for the specified class or if the index is out of bounds
        """
        data = self.X_test if test else self.X_train
        labels = self.y_test_true if test else self.y_train_true
        if label not in labels:
            raise ValueError(f"{label} not in labels: {list(set(labels))}")

        class_indices = np.where(labels == label)[0]

        if index is not None:
            if index not in class_indices:
                raise ValueError(f"Index {index} not found in class {label}")
            sample = data[index : index + 1]
            return sample

        index = np.random.choice(class_indices)
        sample = data[index : index + 1]
        return sample.detach().cpu().numpy()

    def get_nun(self, sample=None, sample_index=None, train=True, k=1):
        """
        Finds the Nearest Unlike Neighbor(s) (NUN) for a given sample.

        :param `sample`: The sample for which to find the NUN. If None, sample_index must be provided, default is None
        :param `sample_index`: The index of the sample in the dataset. If None, sample must be provided, default is None
        :param `train`: Whether to search in the training set (True) or test set (False), default is True
        :param `k`: Number of nearest unlike neighbors to return, default is 1
        :return `nun`: The k nearest unlike neighbors of the sample (converted to a NumPy Array)
        :raises `ValueError`: If neither sample nor sample_index is provided or if the sample index is not found
        """
        if sample is None and sample_index is None:
            raise ValueError("Either `sample` or `sample_index` must be provided")

        if sample_index is None:
            data = self.X_train if train else self.X_test
            if hasattr(data, "dim"):
                sample_index = (
                    (data == sample).all(dim=2).nonzero(as_tuple=True)[0][0].item()
                )
            else:
                sample_index = np.where(np.all(data == sample, axis=2))[0][0]

        nuns_dict = self.nuns_train if train else self.nuns_test
        if sample_index not in nuns_dict:
            raise ValueError(
                f"Sample index {sample_index} not found in the {'training' if train else 'test'} set"
            )

        return nuns_dict[sample_index][k - 1 : k].detach().cpu().numpy()

    def get_true_label(self, sample, train=True):
        """
        Returns the true label of the given sample by matching it with the dataset.

        :param `sample`: The sample for which the label is to be retrieved
        :param `train`: If True, searches in the training set, otherwise in the test set, default is True
        :return `label`: The true label corresponding to the given sample
        """
        data = self.X_train if train else self.X_test
        labels = self.y_train_true if train else self.y_test_true
        label = labels[(data == sample).all(dim=2).squeeze()][0]
        return label

    def get_predicted_label(self, sample, train=True):
        """
        Gets the label predicted by the model for the given sample.

        :param `sample`: The sample for which the prediction is to be obtained.
        :param `train`: If True, searches the training set, otherwise the test set.
        :return `label`: The predicted label for the given sample.
        :raises `ValueError`: If there are no predictions available for the specified set
        """
        if train and self.y_train_model is None:
            raise ValueError(
                "There are no train labels predicted by a model. You must then call `compute_predictions`"
            )
        elif not train and self.y_test_model is None:
            raise ValueError(
                "There are no test labels predicted by a model. You must then call `compute_predictions`"
            )
        data = self.X_train if train else self.X_test
        labels = self.y_train_model if train else self.y_test_model
        label = labels[(data == sample).all(dim=2).squeeze()]
        return label

    # TODO: Make a method that returns a sample and its NUNs
