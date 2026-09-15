import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# ======================================================
# CREATE DATA
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
# BATCH SIZE
# ======================================================

batch_size = 64


# ======================================================
# DATALOADERS
# ======================================================

X_dataloader = DataLoader(
    X,
    batch_size=batch_size
)

y_dataloader = DataLoader(
    Y,
    batch_size=batch_size
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ======================================================
# DROPOUT FROM SCRATCH
# ======================================================

class Dropout(nn.Module):

    def __init__(self, p=0.5):

        super().__init__()

        self.p = p

    def forward(self, x):

        # Evaluation mode
        if not self.training:
            return x

        # Generate random mask
        mask = torch.rand_like(x) > self.p

        # Inverted dropout
        output = x * mask / (1 - self.p)

        return output


# ======================================================
# DEEP NETWORK
# ======================================================

class DeepNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28 * 28, 512),

            nn.ReLU(),

            Dropout(p=0.5),

            nn.Linear(512, 512),

            nn.ReLU(),

            Dropout(p=0.5),

            nn.Linear(512, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits


# ======================================================
# CREATE MODEL
# ======================================================

model = DeepNetwork().to(device)


# ======================================================
# LOSS FUNCTION
# ======================================================

loss_fn = nn.CrossEntropyLoss()


# ======================================================
# OPTIMIZER
# ======================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3
)


# ======================================================
# TRAINING
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

            loss, current = loss.item(), (batch + 1) * len(X)

            print(
                f"loss: {loss:>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


# ======================================================
# TESTING
# ======================================================

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
        f"Test Error:\n"
        f" Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}\n"
    )


# ======================================================
# MAIN SCRIPT
# ======================================================

def main():

    epochs = 5

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