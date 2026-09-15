
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# ======================================================
# Layer normalization from scratch
# ======================================================
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5):

        super().__init__()

        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(
            torch.ones(num_features)
        )

        self.beta = nn.Parameter(
            torch.zeros(num_features)
        )

    def forward(self, x):

        # Calculate mean for each sample
        mean = x.mean(dim=1, keepdim=True)

        # Calculate variance for each sample
        variance = x.var(
            dim=1,
            keepdim=True,
            unbiased=False
        )

        # Normalize
        x_normalized = (
            (x - mean) /
            torch.sqrt(variance + self.eps)
        )

        # Scale and shift
        output = (
            self.gamma * x_normalized
            + self.beta
        )

        return output


# =======================================================
# Deep Network with LayerNorm
# =======================================================
class DeepNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28 * 28, 512),

            LayerNorm(512),

            nn.ReLU(),

            nn.Linear(512, 512),

            LayerNorm(512),

            nn.ReLU(),

            nn.Linear(512, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits


# ========================================================
# Create data
# ========================================================

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


# Defining batch size
batch_size = 64


# Dataloaders
train_dataloader = DataLoader(
    X,
    batch_size=batch_size
)

test_dataloader = DataLoader(
    Y,
    batch_size=batch_size
)


# ========================================================
# Implementing accelerators
# ========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")


model = DeepNetwork().to(device)


# ========================================================
# Loss function and optimizer
# ========================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3
)


# ========================================================
# Training
# ========================================================

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if batch % 100 == 0:

            loss, current = (
                loss.item(),
                (batch + 1) * len(X)
            )

            print(
                f"loss: {loss:>7f}  "
                f"[{current:>5d}/{size:>5d}]"
            )


# ========================================================
# Testing
# ========================================================

def test(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(
                pred,
                y
            ).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    correct /= size

    print(
        f"Test Error: \n"
        f" Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )


# ========================================================
# Main
# ========================================================

def main():

    # Number of epochs
    epochs = 5

    for epoch in range(epochs):

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        # Training
        train(
            train_dataloader,
            model,
            loss_fn,
            optimizer
        )

        # Testing
        test(
            test_dataloader,
            model,
            loss_fn
        )


if __name__ == "__main__":
    main()

