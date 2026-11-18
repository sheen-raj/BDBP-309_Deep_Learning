## Implement vanilla RNN from scratch -

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Generating sequence dataset
torch.manual_seed(42)

num_samples = 10000
sequence_length = 10

# Generate random sequences
X = torch.randn(num_samples, sequence_length, 1)

# Calculate the sum of each sequence
sequence_sum = X.sum(dim=1)

# Assign labels:
# 1 = Positive sum
# 0 = Negative sum

y = (sequence_sum > 0).float()

# Split into training and testing datasets

train_size = 8000

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


batch_size = 64

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# Vanilla RNN from scratch
class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.hidden_size = hidden_size

        # Input-to-hidden weights
        self.W_xh = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.1
        )

        # Hidden-to-hidden weights
        self.W_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )

        # Hidden layer bias
        self.b_h = nn.Parameter(
            torch.zeros(hidden_size)
        )

        # Hidden-to-output weights
        self.W_hy = nn.Parameter(
            torch.randn(hidden_size, output_size) * 0.1
        )

        # Output layer bias
        self.b_y = nn.Parameter(
            torch.zeros(output_size)
        )

    def forward(self, x):

        # Get batch size
        batch_size = x.size(0)

        # Initialize hidden state with zeros
        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        # Process each time step
        for t in range(x.size(1)):

            # Get input at current time step
            x_t = x[:, t, :]

            # Calculate new hidden state
            h = torch.tanh(
                torch.matmul(x_t, self.W_xh)
                + torch.matmul(h, self.W_hh)
                + self.b_h
            )

        # Generate final output
        output = torch.matmul(h, self.W_hy) + self.b_y

        return output


# Initialize model

input_size = 1
hidden_size = 16
output_size = 1

model = VanillaRNN(
    input_size,
    hidden_size,
    output_size
).to(device)

# loss and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# RNN training
def train(dataloader, model, criterion, optimizer):

    model.train()

    running_loss = 0.0

    for batch, (sequences, labels) in enumerate(dataloader):

        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(sequences)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


#  RNN testing
def test(dataloader, model, criterion):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():

        for sequences, labels in dataloader:

            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(sequences)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)

            # Convert logits into probabilities
            probabilities = torch.sigmoid(outputs)

            # Predict class
            predictions = (probabilities >= 0.5).float()

            # Count correct predictions
            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    test_loss = running_loss / total

    return test_loss, accuracy



# Run training
epochs = 20
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):

    train_loss = train(
        train_loader,
        model,
        criterion,
        optimizer
    )

    test_loss, accuracy = test(
        test_loader,
        model,
        criterion
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Training Loss: {train_loss:.4f}, "
        f"Testing Loss: {test_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )


# Plot training and testing loss
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, epochs + 1),
    test_losses,
    marker="s",
    label="Testing Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Vanilla RNN: Loss vs Epochs")

plt.legend()
plt.grid(True)
plt.show()

# Plot testing accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    test_accuracies,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Vanilla RNN: Testing Accuracy")

plt.grid(True)
plt.show()

# Test individual predictions
model.eval()

with torch.no_grad():

    sample_sequences = X_test[:10].to(device)

    outputs = model(sample_sequences)

    probabilities = torch.sigmoid(outputs)

    predictions = (probabilities >= 0.5).float()

    print("\nSample Predictions:")

    for i in range(10):

        actual = int(y_test[i].item())
        predicted = int(predictions[i].item())

        print(
            f"Sequence: {X_test[i].squeeze().tolist()}, "
            f"Actual: {actual}, "
            f"Predicted: {predicted}"
        )