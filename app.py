import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
from tqdm import tqdm
import mlflow
from net import Net
from functools import partial
from utils import loss_and_acc
from data import DataLoader, Dataset, MnistDataset, SplitDataset


hidden_dim = 64
num_classes = 10
batch_size = 32
epochs = 50
learning_rate = 1e-3

data = MnistDataset('data/')

ds = SplitDataset(data, training_split=0.9)
training_data_loader = DataLoader(ds.training_data, batch_size=batch_size)
validation_data_loader = DataLoader(ds.validation_data, batch_size=batch_size)
test_data_loader = DataLoader(ds.test_dataset, batch_size=batch_size)

model = Net(data.channels(), data.dim(), len(data.labels()))
optimizer = optim.Adam(learning_rate)
state = [model.state, optimizer.state, mx.random.state]

mx.eval(model.parameters())

@partial(mx.compile, inputs=state, outputs=state)
def step(X, y):
    train_step_fn = nn.value_and_grad(model, loss_and_acc)
    (loss, acc), grads = train_step_fn(model, X, y)
    optimizer.update(model, grads)
    return loss, acc

mlflow.set_experiment("MNIST Experiment")

with mlflow.start_run():
    mlflow.log_param("channels", data.channels())
    mlflow.log_param("dim", data.dim())
    mlflow.log_param("num_classes", data.labels())
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)

    total_loss = 0.0
    for i in tqdm(range(epochs)):
        total_loss = 0.0
        accuracy = 0
        model.train()

        for _, (X, y) in enumerate(training_data_loader):
            loss, acc = step(X, y)
            total_loss += loss
            accuracy += acc
        
        avg_loss = total_loss.item() / len(training_data_loader)
        accuracy_pct = (accuracy.item() / len(training_data_loader.dataset)) * 100
        mlflow.log_metric("loss", avg_loss, step=i)
        mlflow.log_metric("accuracy", accuracy_pct, step=i)
        print(f"training epoch {i + 1} avg loss: {avg_loss} accuracy: {accuracy_pct:0.3f}%")

model.save_weights('mnist_model.npz')
