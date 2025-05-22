import os
import typing as tp
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeries
from ..utils import label_encoder, predict_proba

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
        self.X_train, y_train, self.X_test, y_test = self.load_dataset(
            dataset, scaling, backend
        )
        self.y_train_true, self.y_test_true = label_encoder(y_train, y_test)
        self.name = dataset.split("/")[-1]
        self.scaling = scaling
        self.backend = backend
        self.model = model
        self.y_train_model = predict_proba(self.model, self.X_train)[1]
        self.y_test_model = predict_proba(self.model, self.X_test)[1]
        self.nuns_train, self.nuns_train_distances = self.compute_nuns(
            train=True, preds=True
        )
        self.nuns_test, self.nuns_test_distances = self.compute_nuns(
            train=False, preds=True
        )

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
        For each sample, finds the 10 closest samples with a different label, using a k-NN classifier based on Euclidean distance.

        The NUNs of the training data are computed within the training set itself.
        The NUNs of the test data are also computed using the training set as the reference base, ensuring consistency across evaluations.

        :param `train`: If True, computes NUNs for training data; otherwise, for test data. Default is True
        :param `preds`: If True, uses model predictions as labels; otherwise, uses true labels. Default is False
        :return `nuns`: A dictionary mapping each sample index to its ordered list of NUNs (as arrays)
        :return `nuns_distances`: The distances to their 10 NUNs
        """
        import warnings

        warnings.filterwarnings(
            "ignore", category=FutureWarning
        )  # The line with the `.fit` causes lots of WARNING messages

        data = self.X_train if train else self.X_test
        labels = labels = self.y_train_model if preds else self.y_train_true

        nuns = dict()
        nuns_distances = dict()
        for i, sample in enumerate(data):
            knn = KNeighborsTimeSeries(n_neighbors=1, metric="euclidean")
            label = (
                self.get_predicted_label(sample, train)
                if preds
                else self.get_true_label(sample, train)
            )
            unlike_labels = [i for i, l in enumerate(labels) if l != label]
            unlike_samples = self.X_train[unlike_labels]
            knn = knn.fit(X=unlike_samples, y=unlike_labels)
            distances, idxs = knn.kneighbors(sample, n_neighbors=10)
            idxs = [unlike_labels[x] for x in idxs.squeeze()]
            nuns[i] = self.X_train[idxs]
            nuns_distances[i] = distances

        return nuns, nuns_distances

    def get_sample(self, label=0, index=None, train=True, failed=False):
        """
        Gets a sample of the data belonging to the class specified by label.

        :param `label`: The label of the class from which a sample is to be obtained, default is 1
        :param `index`: The specific index of the sample to retrieve. If None, a random sample is returned, default is None
        :param `train`: If True, gets the sample from the train data, otherwise from test data, default is False
        :param `failed`: Parameter is not used in the current implementation
        :return `sample`: A sample of the specified class
        :raises `ValueError`: If there is no data for the specified class or if the index is out of bounds
        """
        data = self.X_train if train else self.X_test
        labels = self.y_train_true if train else self.y_test_true
        if label not in labels:
            raise ValueError(f"{label} not in labels: {list(set(labels))}")

        class_indices = np.where(labels == label)[0]

        if index is not None:
            if index not in class_indices:
                raise ValueError(f"Index {index} not found in class {label}")
            sample = data[index]
            return sample

        # TODO: Implement ´failed´ functionality.
        # This will return a sample that the model failed predicting it.

        index = np.random.choice(class_indices)
        sample = data[index]
        return sample

    def get_nun(
        self, sample=None, sample_index=None, train=True, k=1, weighted_random=False
    ):
        """
        Finds the Nearest Unlike Neighbor(s) (NUN) for a given sample.

        :param `sample`: The sample for which to find the NUN. If None, sample_index must be provided, default is None
        :param `sample_index`: The index of the sample in the dataset. If None, sample must be provided, default is None
        :param `train`: Whether to search in the training set (True) or test set (False), default is True
        :param `k`: Number of nearest unlike neighbors to return, default is 1
        :param `weighted_random`: If True, selects a neighbor based on weighted random selection using inverse distances, default is False
        :return: The k nearest unlike neighbors of the sample (converted to a NumPy Array)
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

        if weighted_random:
            distances = self.nuns_train_distances if train else self.nuns_test_distances
            distances = 1 / distances[sample_index].flatten()[:4]
            probs = distances / distances.sum()
            k = np.random.choice(4, p=probs)

        return nuns_dict[sample_index][k]

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

    def get_test_samples(self, label=0, n_samples=100, random_seed=42, k=1):
        """
        Returns a fixed set of random samples from the test dataset belonging to a specific class,
        along with their corresponding Nearest Unlike Neighbors (NUNs).

        :param `label`: The class label for which to get samples, default is 0
        :param `n_samples`: Number of samples to return (default: 100)
        :param `random_seed`: Random seed to ensure reproducibility (default: 42)
        :param `k`: The k-th nearest unlike neighbor to return (default: 1)
        :return: A tuple containing (X_samples, y_samples, nuns) - the samples, their true labels, and their NUNs
        :raises `ValueError`: If no samples with the specified label are found
        """
        class_indices = np.where(self.y_test_model == label)[0]

        if len(class_indices) == 0:
            raise ValueError(f"No samples with label {label} found in the test dataset")

        if n_samples >= len(class_indices):
            selected_indices = class_indices
        else:
            np.random.seed(random_seed)

            selected_indices = np.random.choice(
                class_indices, size=n_samples, replace=False
            )

        X_samples = self.X_test[selected_indices]
        y_samples = self.y_test_model[selected_indices]

        nuns = []
        for idx in selected_indices:
            try:
                nun = self.get_nun(sample_index=idx, train=False, k=k)
                nuns.append(nun)
            except ValueError:
                nuns.append(None)

        return X_samples, y_samples, nuns

    def get_shape(self):
        """
        Gets the shape of the samples.

        :return: The shape of the data
        """
        return self.X_train[0].shape

    def get_num_samples(self):
        """
        Gets the number of samples in the training and test sets.

        :return: A tuple with the number of training and test samples
        """
        return self.X_train.shape[0], self.X_test.shape[0]

    def get_dim(self):
        """
        Gets the number of dimensions in the data.

        :return: The number of dimensions
        """
        return self.X_train.shape[1]

    def get_len(self):
        """
        Gets the number of temporal instances.

        :return: The number of temporal instances
        """
        return self.X_train.shape[2]

    def __str__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def __repr__(self):
        return f"<{self.__class__.__name__}(dataset='{self.name}', model='{self.model.__class__.__name__}')>"

    # TODO: Make a method that returns a sample and its NUNs
