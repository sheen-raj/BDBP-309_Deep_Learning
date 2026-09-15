
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# ======================================================
# Create data
# ======================================================

X = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
)

Y = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
)


# ======================================================
# Batch size
# ======================================================

batch_size = 64


# ======================================================
# Dataloaders
# ======================================================

X_dataloader = DataLoader(
    X,
    batch_size=batch_size
)

y_dataloader = DataLoader(
    Y,
    batch_size=batch_size
)


# ======================================================
# Device
# ======================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")


# ======================================================
# Define model
# ======================================================

class DeepNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10)

        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits


# ======================================================
# Loss function
# ======================================================

loss_fn = nn.CrossEntropyLoss()


# ======================================================
# Optimizers
# ======================================================

optimizers = {
    "SGD": lambda params: torch.optim.SGD(
        params,
        lr=0.001
    ),

    "Momentum": lambda params: torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9
    ),

    "AdaGrad": lambda params: torch.optim.Adagrad(
        params,
        lr=0.001
    ),

    "Adam": lambda params: torch.optim.Adam(
        params,
        lr=0.001
    )
}


# ======================================================
# Training
# ======================================================

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        if batch % 100 == 0:

            loss_value = loss.item()

            current = (batch + 1) * len(X)

            print(
                f"loss: {loss_value:>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


# ======================================================
# Testing
# ======================================================

def test(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    correct /= size

    print(
        f"Test Error:\n"
        f" Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}\n"
    )


# ======================================================
# Main
# ======================================================

def main():

    epochs = 5

    for optimizer_name, optimizer_function in optimizers.items():

        print("\n====================================")
        print(f"Optimizer: {optimizer_name}")
        print("====================================")

        # Create a fresh model
        model = DeepNetwork().to(device)

        # Create optimizer for this model
        optimizer = optimizer_function(model.parameters())

        # Training
        for epoch in range(epochs):

            print(f"\nEpoch {epoch + 1}/{epochs}")

            train(
                X_dataloader,
                model,
                loss_fn,
                optimizer
            )

            test(
                y_dataloader,
                model,
                loss_fn
            )


if __name__ == "__main__":
    main()