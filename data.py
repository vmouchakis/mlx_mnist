import gzip
import os
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
from mnist import MNIST

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


def load_data(data_path='data/MNIST/raw'):
    """
    Load the MNIST dataset.

    Parameters:
    data_path (str): Path to the directory containing the raw MNIST data files.
    
    Returns:
    tuple: Four NumPy arrays: train_images, train_labels, test_images, test_labels
    """
    mndata = MNIST(data_path)
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Convert lists to NumPy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print(train_images.shape)
    return train_images, train_labels, test_images, test_labels
