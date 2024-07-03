import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
import mlflow
from mlp import MLP
from utils import *
from data import load_data


num_layers = 2
hidden_dim = 64
num_classes = 10
batch_size = 64
num_epochs = 50
learning_rate = 1e-1

train_images, train_labels, test_images, test_labels = map(
    mx.array, load_data()
)

model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
mx.eval(model.parameters())

import sys; sys.exit()
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.SGD(learning_rate=learning_rate)

mlflow.set_experiment("MNIST Experiment")

with mlflow.start_run():
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)

    for e in tqdm(range(num_epochs)):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)

            optimizer.update(model, grads)

            mx.eval(model.parameters(), optimizer.state)

        accuracy = eval_fn(model, test_images, test_labels)
        print(f"Epoch {e}: Test accuracy {accuracy.item():.3f} | Loss {loss.item():.3f}")
        mlflow.log_metric("loss", loss.item(), step=e)
        mlflow.log_metric("test_accuracy", accuracy.item(), step=e)

# mlflow.log_artifact('mnist_model.params')
model.save_weights('mnist_model.npz')
