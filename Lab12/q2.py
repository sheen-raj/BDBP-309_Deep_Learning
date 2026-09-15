import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os

os.environ["HTTP_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"

# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# CIFAR-10 dataset
# ============================================================

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# Plain CNN
# ============================================================

class PlainCNN(nn.Module):

    def __init__(self, num_layers):

        super().__init__()

        layers = []

        in_channels = 3

        out_channels = 64


        # ----------------------------------------------------
        # Convolutional layers
        # ----------------------------------------------------

        for layer in range(num_layers):

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.ReLU()
            )

            in_channels = out_channels


        self.conv_layers = nn.Sequential(*layers)


        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.fc = nn.Linear(
            out_channels,
            10
        )


    def forward(self, x):

        x = self.conv_layers(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc(x)

        return x


# ============================================================
# Training function
# ============================================================

def train_model(model, epochs=10):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )


    model.train()


    for epoch in range(epochs):

        correct = 0
        total = 0
        total_loss = 0


        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Forward pass
            outputs = model(images)


            # Loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # Statistics
            total_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Accuracy: {accuracy:.2f}%"
        )


    train_error = 1 - (correct / total)

    return train_error


# ============================================================
# Experiment
# ============================================================

depths = [
    1,
    2,
    4,
    8,
    12,
    16
]


train_errors = []


for depth in depths:

    print("\n" + "=" * 60)

    print(
        f"Training CNN with {depth} convolutional layers"
    )

    print("=" * 60)


    model = PlainCNN(
        num_layers=depth
    ).to(device)


    train_error = train_model(
        model,
        epochs=10
    )


    train_errors.append(
        train_error
    )


    print(
        f"Final training error: "
        f"{train_error:.4f}"
    )


# ============================================================
# Print results
# ============================================================

print("\nFinal results:")

for depth, error in zip(
    depths,
    train_errors
):

    print(
        f"{depth} layers → "
        f"training error = {error:.4f}"
    )


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_errors,
    marker="o"
)

plt.xlabel(
    "Number of convolutional layers"
)

plt.ylabel(
    "Training error"
)

plt.title(
    "Training Error vs Network Depth"
)

plt.grid()

plt.show()