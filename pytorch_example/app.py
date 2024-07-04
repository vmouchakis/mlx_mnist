import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import mlflow

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device)

lr = 1e-3
epochs = 50
batch_size = 64
comp = False

trainset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net().to(device)
if comp:
    # The operator 'aten::native_dropout' is not currently implemented for the MPS device
    # As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`
    # to use the CPU as a fallback for this op
    model = torch.compile(model, mode='default', backend='aot_eager')
optimizer = optim.Adadelta(model.parameters(), lr=lr)

mlflow.set_experiment("MNIST Experiment (PyTorch)")

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        loss_epoch = []
        correct = 0
        total = 0
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        avg_loss = sum(loss_epoch) / len(loss_epoch)
        accuracy = correct / total

        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for idx, (data, target) in enumerate(testloader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    test_loss += loss.item()
                    
                    # Calculate test accuracy
                    _, predicted = torch.max(output, 1)
                    correct_test += (predicted == target).sum().item()
                    total_test += target.size(0)
            
            avg_test_loss = test_loss / len(testloader)
            test_accuracy = correct_test / total_test
            print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)
            
torch.save(model.state_dict(), "mnist_cnn.pt")
