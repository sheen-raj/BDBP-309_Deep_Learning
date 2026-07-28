import torch
from torch import nn, device
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import os

os.environ["HTTP_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"

# define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=28*28, out_features=512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        logits = self.linear3(x)
        out = logits
        return out

def load_data():
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    return train_data, test_data

def train(mydataloader, model, loss_fn, optimizer, device, epochs):
    size = len(mydataloader.dataset)
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)

            # Prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(mydataloader, model, loss_fn, device):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in mydataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # Load datasets
    train_data, test_data = load_data()
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    # Load Accelerators
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize the network
    model = NeuralNetwork().to(device)
    print(model)

    # Optimize
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Model training
    train(train_dataloader, model, loss_fn, optimizer, device=device, epochs=10)

    # Model testing
    test(test_dataloader, model, loss_fn, device)

    print('End')


if __name__ == "__main__":
    main()