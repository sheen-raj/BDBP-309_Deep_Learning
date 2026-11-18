import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/accessibility_dataset.csv"
OUTPUT_DIR = "results"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

TEST_SIZE = 0.20
VAL_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nFirst few rows:")
print(df.head())

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# CLEAN SEQUENCES
# ============================================================

df["sequence"] = (
    df["sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# Keep only valid DNA sequences
valid_bases = set("ACGT")

df = df[
    df["sequence"].apply(
        lambda x: set(x).issubset(valid_bases)
    )
].copy()

df["label"] = df["label"].astype(int)

print("\nAfter sequence filtering:")
print("Number of samples:", len(df))


# ============================================================
# CHECK SEQUENCE LENGTH
# ============================================================

sequence_lengths = df["sequence"].str.len()

print("\nSequence length:")
print("Minimum:", sequence_lengths.min())
print("Maximum:", sequence_lengths.max())
print("Mean:", sequence_lengths.mean())


# ============================================================
# KEEP ONLY ONE SEQUENCE LENGTH
# ============================================================

sequence_length = sequence_lengths.mode()[0]

df = df[
    df["sequence"].str.len() == sequence_length
].copy()

print("\nUsing sequence length:", sequence_length)
print("Samples remaining:", len(df))


# ============================================================
# ONE-HOT ENCODING
# ============================================================

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def one_hot_encode(sequence):

    encoded = np.zeros(
        (4, len(sequence)),
        dtype=np.float32
    )

    for position, base in enumerate(sequence):

        index = BASE_TO_INDEX[base]

        encoded[index, position] = 1.0

    return encoded


# ============================================================
# DATASET
# ============================================================

class DNADataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = sequences
        self.labels = labels

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        sequence = self.sequences[index]
        label = self.labels[index]

        sequence = one_hot_encode(sequence)

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return sequence, label


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

sequences = df["sequence"].values
labels = df["label"].values


# First: train+validation vs test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    sequences,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)


# Then: training vs validation
validation_fraction = VAL_SIZE / (1 - TEST_SIZE)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=validation_fraction,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)


print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = DNADataset(
    X_train,
    y_train
)

val_dataset = DNADataset(
    X_val,
    y_val
)

test_dataset = DNADataset(
    X_test,
    y_test
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CNN MODEL
# ============================================================

class AccessibilityCNN(nn.Module):

    def __init__(self, sequence_length):

        super().__init__()

        # ----------------------------------------------------
        # Convolutional layers
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=15
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=10
        )

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        # ----------------------------------------------------
        # Adaptive pooling
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            256,
            64
        )

        self.fc2 = nn.Linear(
            64,
            1
        )

        self.dropout = nn.Dropout(
            0.3
        )

    def forward(self, x):

        # x:
        # batch × 4 × sequence_length

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        # Global max pooling
        x = self.global_pool(x)

        # batch × 256 × 1
        x = x.squeeze(-1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        # Return logits
        return x.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

model = AccessibilityCNN(
    sequence_length
)

model = model.to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for sequences_batch, labels_batch in train_loader:

        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)

        # Forward pass
        outputs = model(
            sequences_batch
        )

        # Calculate loss
        loss = criterion(
            outputs,
            labels_batch
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Predictions
        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    average_loss = (
        total_loss / len(train_loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for sequences_batch, labels_batch in val_loader:

            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(
                sequences_batch
            )

            loss = criterion(
                outputs,
                labels_batch
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += labels_batch.size(0)

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    average_loss = (
        total_loss / len(val_loader)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_probabilities)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting training...\n")

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = validate()

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            f"{OUTPUT_DIR}/best_model.pt"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        f"{OUTPUT_DIR}/best_model.pt",
        map_location=device
    )
)


# ============================================================
# TEST SET EVALUATION
# ============================================================

model.eval()

all_test_labels = []
all_test_probabilities = []

with torch.no_grad():

    for sequences_batch, labels_batch in test_loader:

        sequences_batch = sequences_batch.to(device)

        outputs = model(
            sequences_batch
        )

        probabilities = torch.sigmoid(
            outputs
        )

        all_test_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_test_labels.extend(
            labels_batch.numpy()
        )


all_test_labels = np.array(
    all_test_labels
)

all_test_probabilities = np.array(
    all_test_probabilities
)

all_test_predictions = (
    all_test_probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_test_labels,
    all_test_predictions
)

precision = precision_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)

recall = recall_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)

f1 = f1_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_test_labels,
    all_test_probabilities
)

pr_auc = average_precision_score(
    all_test_labels,
    all_test_probabilities
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("TEST RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

metrics.to_csv(
    f"{OUTPUT_DIR}/metrics.csv",
    index=False
)


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loss_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/accuracy_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    all_test_labels,
    all_test_probabilities
)

plt.figure(
    figsize=(6, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Inaccessible",
        "Accessible"
    ]
)

disp.plot()

plt.title(
    "CNN Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "sequence": X_test,
    "true_label": all_test_labels,
    "predicted_probability": all_test_probabilities,
    "predicted_label": all_test_predictions
})

prediction_df.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("Finished.")
print("========================================")

print(
    f"Results saved in: {OUTPUT_DIR}/"
)