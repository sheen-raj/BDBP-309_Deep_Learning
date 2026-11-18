# ============================================================
# Image Captioning using CNN + LSTM in PyTorch
# Dataset: Flickr8k
# ============================================================

import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ==========================================
# 1. SETTINGS
# ==========================================

# Change these paths if your dataset has a different structure.
dataset_folder = "./image_captioning/Flickr8k"

image_folder = os.path.join(dataset_folder, "Flicker8k_Dataset")
caption_file = os.path.join(dataset_folder, "Flickr8k.token.txt")

train_file = os.path.join(dataset_folder, "Flickr_8k.trainImages.txt")
val_file = os.path.join(dataset_folder, "Flickr_8k.devImages.txt")
test_file = os.path.join(dataset_folder, "Flickr_8k.testImages.txt")

batch_size = 32
epochs = 5
hidden_size = 256
learning_rate = 0.001
num_workers = 2
max_caption_length = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Check dataset paths before attempting to open files.
required_paths = {
    "Image folder": image_folder,
    "Caption file": caption_file,
    "Training split": train_file,
    "Validation split": val_file,
    "Testing split": test_file,
}

print("\nChecking dataset paths...")
for name, path in required_paths.items():
    exists = os.path.exists(path)
    print(f"{name}: {exists} ({path})")
    if not exists:
        raise FileNotFoundError(
            f"{name} was not found at: {path}\n"
            "Please correct the corresponding path in section 1."
        )


# ==========================================
# 2. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD CAPTIONS
# ==========================================

captions = {}

with open(caption_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        image_name, caption = line.split("\t", 1)
        image_name = image_name.split("#")[0]

        caption = caption.lower()
        caption = re.sub(r"[^a-z ]", "", caption)
        caption = caption.split()

        if image_name not in captions:
            captions[image_name] = []

        captions[image_name].append(caption)


# ==========================================
# 4. LOAD TRAIN / VALIDATION / TEST IMAGES
# ==========================================

def read_images(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


train_images = read_images(train_file)
val_images = read_images(val_file)
test_images = read_images(test_file)

print("\nTraining images:", len(train_images))
print("Validation images:", len(val_images))
print("Testing images:", len(test_images))


# Keep only image names that have captions.
train_images = [name for name in train_images if name in captions]
val_images = [name for name in val_images if name in captions]
test_images = [name for name in test_images if name in captions]

print("\nImages with captions:")
print("Training:", len(train_images))
print("Validation:", len(val_images))
print("Testing:", len(test_images))


# ==========================================
# 5. CREATE VOCABULARY
# ==========================================

word_count = Counter()

for image_name in train_images:
    for caption in captions[image_name]:
        word_count.update(caption)

vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

for word, count in word_count.items():
    if count >= 2:
        vocab.append(word)

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

vocab_size = len(vocab)

PAD = word_to_id["<PAD>"]
SOS = word_to_id["<SOS>"]
EOS = word_to_id["<EOS>"]
UNK = word_to_id["<UNK>"]

print("\nVocabulary size:", vocab_size)


def encode_caption(caption):
    # Keep the end token even when a caption is truncated.
    caption = caption[:max_caption_length - 2]
    tokens = ["<SOS>"] + caption + ["<EOS>"]

    return [
        word_to_id.get(word, UNK)
        for word in tokens
    ]


# ==========================================
# 6. CREATE DATASET
# ==========================================

class FlickrDataset(Dataset):

    def __init__(self, image_names):
        self.data = []

        for image_name in image_names:
            for caption in captions[image_name]:
                self.data.append((image_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, caption = self.data[index]

        image_path = os.path.join(image_folder, image_name)

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        caption = torch.tensor(
            encode_caption(caption),
            dtype=torch.long
        )

        return image, caption


# Pad captions so that all captions in a batch have equal length.
def collate_fn(batch):
    images, captions_batch = zip(*batch)

    images = torch.stack(images)

    captions_batch = pad_sequence(
        captions_batch,
        batch_first=True,
        padding_value=PAD
    )

    return images, captions_batch


train_dataset = FlickrDataset(train_images)
val_dataset = FlickrDataset(val_images)
test_dataset = FlickrDataset(test_images)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda")
)

print("\nTraining caption samples:", len(train_dataset))
print("Validation caption samples:", len(val_dataset))
print("Testing caption samples:", len(test_dataset))


# ==========================================
# 7. CNN + LSTM MODEL
# ==========================================

class ImageCaptioningLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        # CNN: extract image features using pretrained ResNet18.
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove the final classification layer.
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Freeze CNN parameters.
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        # ResNet18 produces 512 image features.
        # Project them to the LSTM hidden-state dimension.
        self.image_fc_h = nn.Linear(512, hidden_size)
        self.image_fc_c = nn.Linear(512, hidden_size)

        # Convert word IDs into dense vectors.
        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=PAD
        )

        # LSTM decoder.
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict the next word in the vocabulary.
        self.fc = nn.Linear(hidden_size, vocab_size)

    # Extract image features and initialize both LSTM states.
    def get_states(self, images):

        # Keep the frozen CNN in evaluation mode.
        self.cnn.eval()

        with torch.no_grad():
            features = self.cnn(images)
            features = features.view(features.size(0), -1)

        # LSTM requires both hidden state (h0) and cell state (c0).
        h0 = torch.tanh(self.image_fc_h(features)).unsqueeze(0)
        c0 = torch.tanh(self.image_fc_c(features)).unsqueeze(0)

        return h0, c0

    # Forward pass during training.
    def forward(self, images, captions):

        h0, c0 = self.get_states(images)

        embeddings = self.embedding(captions)

        output, (hidden, cell) = self.lstm(
            embeddings,
            (h0, c0)
        )

        output = self.fc(output)

        return output


model = ImageCaptioningLSTM(
    vocab_size,
    hidden_size
).to(device)


# ==========================================
# 8. LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)


# ==========================================
# 9. TRAINING FUNCTION
# ==========================================

def train():

    model.train()
    total_loss = 0.0

    for images, captions_batch in train_loader:

        images = images.to(device, non_blocking=True)
        captions_batch = captions_batch.to(device, non_blocking=True)

        # Teacher forcing:
        # Input contains all caption tokens except the last.
        # Target contains all caption tokens except the first.
        inputs = captions_batch[:, :-1]
        targets = captions_batch[:, 1:]

        optimizer.zero_grad()

        outputs = model(images, inputs)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()

        # Reduce the risk of exploding gradients in recurrent networks.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ==========================================
# 10. VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for images, captions_batch in val_loader:

            images = images.to(device, non_blocking=True)
            captions_batch = captions_batch.to(device, non_blocking=True)

            inputs = captions_batch[:, :-1]
            targets = captions_batch[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# ==========================================
# 11. TRAINING LOOP
# ==========================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(epochs):

    train_loss = train()
    val_loss = validate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "word_to_id": word_to_id,
                "id_to_word": id_to_word,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size
            },
            "image_captioning_lstm_best.pth"
        )


# ==========================================
# 12. PLOT TRAINING AND VALIDATION LOSS
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("CNN-LSTM Image Captioning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("image_captioning_lstm_loss.png", dpi=300)
plt.show()


# ==========================================
# 13. GENERATE CAPTION
# ==========================================

def generate_caption(image, max_length=20):

    model.eval()

    image = image.unsqueeze(0).to(device)

    generated_words = []

    with torch.no_grad():

        h, c = model.get_states(image)

        current_word = torch.tensor(
            [[SOS]],
            dtype=torch.long,
            device=device
        )

        for step in range(max_length):

            embedding = model.embedding(current_word)

            output, (h, c) = model.lstm(
                embedding,
                (h, c)
            )

            prediction = model.fc(output[:, -1, :])

            predicted_id = prediction.argmax(dim=1).item()

            if predicted_id == EOS:
                break

            word = id_to_word[predicted_id]

            if word not in ["<PAD>", "<SOS>", "<UNK>"]:
                generated_words.append(word)

            current_word = torch.tensor(
                [[predicted_id]],
                dtype=torch.long,
                device=device
            )

    return " ".join(generated_words)


# ==========================================
# 14. TEST ON AN UNSEEN IMAGE
# ==========================================

test_image_name = random.choice(test_images)

test_image_path = os.path.join(
    image_folder,
    test_image_name
)

test_image = Image.open(test_image_path).convert("RGB")

actual_captions = captions[test_image_name]
image_tensor = transform(test_image)

predicted_caption = generate_caption(image_tensor)

print("\nImage:", test_image_name)

print("\nActual captions:")
for caption in actual_captions:
    print(" -", " ".join(caption))

print("\nGenerated caption:")
print(predicted_caption)

plt.figure(figsize=(8, 6))
plt.imshow(test_image)
plt.axis("off")
plt.title("Generated caption:\n" + predicted_caption)
plt.tight_layout()
plt.savefig("image_captioning_lstm_example.png", dpi=300)
plt.show()