## Implement vanilla RNN using PyTorch for image captioning. You can use the PyTorch library.

import os
os.environ["HTTP_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"

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

# SETTINGS
# Dataset directory
dataset_folder = "./image_captioning/Flickr8k"

# Image directory
image_folder = os.path.join(
    dataset_folder,
    "Flicker8k_Dataset"
)

# Caption file
caption_file = os.path.join(
    dataset_folder,
    "Flickr8k.token.txt"
)

# Dataset split files
train_file = os.path.join(
    dataset_folder,
    "Flickr_8k.trainImages.txt"
)

val_file = os.path.join(
    dataset_folder,
    "Flickr_8k.devImages.txt"
)

test_file = os.path.join(
    dataset_folder,
    "Flickr_8k.testImages.txt"
)

batch_size = 32
epochs = 5
hidden_size = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# IMAGE TRANSFORMATION
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# LOAD CAPTIONS
captions = {}

with open(caption_file, "r") as file:

    for line in file:

        image_name, caption = line.strip().split("\t")

        image_name = image_name.split("#")[0]

        caption = caption.lower()
        caption = re.sub(r"[^a-z ]", "", caption)
        caption = caption.split()

        if image_name not in captions:
            captions[image_name] = []

        captions[image_name].append(caption)


# LOAD TRAIN / VALIDATION / TEST IMAGES
def read_images(filename):

    with open(filename, "r") as file:
        return file.read().splitlines()


train_images = read_images(train_file)
val_images = read_images(val_file)
test_images = read_images(test_file)

print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Testing images:", len(test_images))

# Checks
print("\nChecking dataset paths...")

print("Image folder:", os.path.exists(image_folder))
print("Caption file:", os.path.exists(caption_file))
print("Training file:", os.path.exists(train_file))
print("Validation file:", os.path.exists(val_file))
print("Testing file:", os.path.exists(test_file))

# CREATE VOCABULARY
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

print("Vocabulary size:", vocab_size)


# Convert a caption into numbers
def encode_caption(caption):

    tokens = ["<SOS>"] + caption + ["<EOS>"]

    return [
        word_to_id.get(word, word_to_id["<UNK>"])
        for word in tokens
    ]

# CREATE DATASET
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


# Pad captions so they have equal lengths within a batch
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
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2
)


# CNN + VANILLA RNN MODEL
class ImageCaptioningRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        # CNN: extract image features
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Freeze CNN parameters
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        # Convert image features into hidden state
        self.image_fc = nn.Linear(512, hidden_size)

        # Word embedding
        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=PAD
        )

        # Vanilla RNN
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Predict next word
        self.fc = nn.Linear(hidden_size, vocab_size)


    # Extract image features and initialize hidden state
    def get_hidden(self, images):

        self.cnn.eval()

        with torch.no_grad():

            features = self.cnn(images)

            features = features.view(features.size(0), -1)

        hidden = torch.tanh(self.image_fc(features))

        return hidden.unsqueeze(0)


    # Forward pass

    def forward(self, images, captions):

        hidden = self.get_hidden(images)

        embeddings = self.embedding(captions)

        output, hidden = self.rnn(embeddings, hidden)

        output = self.fc(output)

        return output


model = ImageCaptioningRNN(
    vocab_size,
    hidden_size
).to(device)


# LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)


# TRAINING FUNCTION
def train():

    model.train()

    total_loss = 0

    for images, captions_batch in train_loader:

        images = images.to(device)
        captions_batch = captions_batch.to(device)

        # Input and expected output

        inputs = captions_batch[:, :-1]
        targets = captions_batch[:, 1:]


        optimizer.zero_grad()

        # Forward pass

        outputs = model(images, inputs)


        # Calculate loss

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )


        # Backpropagation

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(train_loader)


# VALIDATION FUNCTION
def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for images, captions_batch in val_loader:

            images = images.to(device)
            captions_batch = captions_batch.to(device)

            inputs = captions_batch[:, :-1]
            targets = captions_batch[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_loss += loss.item()


    return total_loss / len(val_loader)


# TRAINING LOOP
for epoch in range(epochs):

    train_loss = train()
    val_loss = validate()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )


# GENERATE CAPTION
def generate_caption(image):

    model.eval()

    image = image.unsqueeze(0).to(device)

    generated_words = []

    with torch.no_grad():

        hidden = model.get_hidden(image)

        current_word = torch.tensor(
            [[SOS]],
            dtype=torch.long
        ).to(device)


        for step in range(20):

            embedding = model.embedding(current_word)

            output, hidden = model.rnn(embedding, hidden)

            prediction = model.fc(output)

            predicted_id = prediction.argmax(dim=-1).item()

            if predicted_id == EOS:
                break

            word = id_to_word[predicted_id]

            if word not in ["<PAD>", "<SOS>"]:
                generated_words.append(word)

            current_word = torch.tensor(
                [[predicted_id]],
                dtype=torch.long
            ).to(device)


    return " ".join(generated_words)


# TEST ON AN UNSEEN IMAGE
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