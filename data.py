import gzip
import mlx.core as mx
import os
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
from torchvision.datasets import MNIST

def download_data(mnist_zip_url='https://data.deepai.org/mnist.zip', data_dir='data'):
    """
    Download and extract the MNIST dataset.
    
    Parameters:
    mnist_zip_url (str): URL to the MNIST zip file.
    data_dir (str): Directory to save the extracted data.
    """
    # Define paths
    mnist_dir = os.path.join(data_dir, "MNIST")
    raw_mnist_dir = os.path.join(mnist_dir, "raw")
    
    # Create directories if they don't exist
    os.makedirs(raw_mnist_dir, exist_ok=True)
    
    # Download and unzip the dataset
    with urlopen(mnist_zip_url) as zip_response:
        with ZipFile(BytesIO(zip_response.read())) as zfile:
            zfile.extractall(raw_mnist_dir)
    
    # Unzip .gz files
    for fname in os.listdir(path=raw_mnist_dir):
        if fname.endswith(".gz"):
            fpath = os.path.join(raw_mnist_dir, fname)
            with gzip.open(fpath, 'rb') as f_in:
                fname_unzipped = fname.replace(".gz", "")
                with open(os.path.join(raw_mnist_dir, fname_unzipped), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


class MnistDataset:
    def __init__(self, path):
        self._X_train = MNIST(root=path, train=True, download=True).data.numpy().reshape(-1, 28, 28, 1).astype(float).tolist()
        self._X_test = MNIST(root=path, train=False, download=True).data.numpy().reshape(-1, 28, 28, 1).astype(float).tolist()
        self._y_train = MNIST(root=path, train=True, download=True).targets.numpy().tolist()
        self._y_test = MNIST(root=path, train=False, download=True).targets.numpy().tolist()
        self._labels = list(range(10))
    
    def X_train(self):
        return self._X_train
    
    def y_train(self):
        return self._y_train

    def X_test(self):
        return self._X_test

    def y_test(self):
        return self._y_test

    def labels(self):
        return self._labels
    
    def channels(self):
        return 1
    
    def dim(self):
        return 28


class Dataset:
    def __init__(self, X, y):
        self.X = mx.array(X)
        self.y = mx.array(y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
  
    def __iter__(self):
        for batch_id in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[batch_id:batch_id+self.batch_size]
  
    def __len__(self):
        return len(self.dataset) // self.batch_size
    

def split_training_data(X, y, training_split):
    import random
    paired = list(zip(X, y))
    random.shuffle(paired)

    X, y = zip(*paired)
    X = mx.array(X)
    y = mx.array(y)

    split_index = int(training_split * len(X))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_validation = X[split_index:]
    y_validation = y[split_index:]

    return X_train, y_train, X_validation, y_validation


class SplitDataset:
    def __init__(self, data, training_split):
        X_train, y_train, X_validation, y_validation = split_training_data(data.X_train(), data.y_train(), training_split)
        self.training_data = Dataset(X_train, y_train)
        self.validation_data = Dataset(X_validation, y_validation)
        self.test_dataset = Dataset(data.X_test(), data.y_test())
