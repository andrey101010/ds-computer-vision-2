# start in jupyter notebook mlflow ui before executing this script
# Full pytorch workflow with MNIST fashion dataset used in the Pytorch learning path at Microsoft
import torch
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch
from config import TRACKING_URI, EXPERIMENT_NAME

logger = getLogger(__name__)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# %%
print(torch.cuda.is_available())

# Increases accuracy to 71 % compared to transform = ToTensor(), which delivers 41 %.
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    #transforms.Resize((100,100)) # if the input images varies in sizes. not the case for this dataset
    ])

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,)

# Create data loaders.
batch_size = 4
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
model = NeuralNetwork()
summary(model,input_size=(1,1,28,28))

# %%
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
# the addition of momentum = .9 improves the accuracy from 80 % to 84 % in 1 epoch
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9) 

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):  
        # possible representation of (X, y) with more lines of code.      
        # is just (X, y) = data an in additional line
        # X, y = data 
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad() # Zero your gradients for every batch!
        loss.backward()
        optimizer.step() # gradient descent

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad(): # disables gradient calculation
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
def run_training():
    mlflow.pytorch.autolog() # or mlflow.tensorflow.autolog() does not work here

    with mlflow.start_run(): # which automatically terminates the run at the end of the with block.
        logger.info(f"Creating model")
        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

run_training()
