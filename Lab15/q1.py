## Implement vanilla RNN using PyTorch for image captioning. You can use the PyTorch library.
# This script uses a single image and pretrained ResNet for studying purpose

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import os

os.environ["HTTP_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary
words = [
    "<PAD>", "<SOS>", "<EOS>",
    "a", "dog", "is", "running",
    "cat", "sitting", "on", "mat"
]

word_to_id = {word: i for i, word in enumerate(words)}
id_to_word = {i: word for word, i in word_to_id.items()}

vocab_size = len(words)

# Example caption
caption = ["<SOS>", "a", "dog", "is", "running", "<EOS>"]

caption_ids = [word_to_id[word] for word in caption]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# CNN Encoder
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove final classification layer
cnn = nn.Sequential(*list(resnet.children())[:-1])

cnn = cnn.to(device)
cnn.eval()


# Vanilla RNN Model
class ImageCaptioningRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size=256):

        super().__init__()

        # Convert image features to RNN hidden state
        self.image_fc = nn.Linear(512, hidden_size)

        # Convert word IDs into embeddings
        self.embedding = nn.Embedding(vocab_size, 100)

        # Vanilla RNN
        self.rnn = nn.RNN(
            input_size=100,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict next word
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):

        # Initialize hidden state using image features
        hidden = self.image_fc(image_features)
        hidden = hidden.unsqueeze(0)

        # Convert caption words into embeddings
        embeddings = self.embedding(captions)

        # Pass embeddings through RNN
        output, hidden = self.rnn(embeddings, hidden)

        # Predict words
        output = self.fc(output)

        return output


model = ImageCaptioningRNN(vocab_size).to(device)

# Sample image
# Replace with your image path
image_path = "dog_running.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Extract image features
with torch.no_grad():
    image_features = cnn(image)

image_features = image_features.view(1, 512)


# Prepare caption input and target
caption_tensor = torch.tensor(
    [caption_ids],
    dtype=torch.long
).to(device)

# Input:  <SOS> a dog is running
# Target: a dog is running <EOS>

input_caption = caption_tensor[:, :-1]
target_caption = caption_tensor[:, 1:]


# Training setup
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training loop
model.train()

for epoch in range(100):

    optimizer.zero_grad()

    # Predict next words
    output = model(image_features, input_caption)

    # Calculate loss
    loss = criterion(
        output.reshape(-1, vocab_size),
        target_caption.reshape(-1)
    )

    # Backpropagation
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")


# Generate caption / Evaluation
model.eval()

generated_caption = ["<SOS>"]

current_word = torch.tensor(
    [[word_to_id["<SOS>"]]],
    dtype=torch.long
).to(device)

hidden = model.image_fc(image_features).unsqueeze(0)

with torch.no_grad():

    for _ in range(10):

        embedding = model.embedding(current_word)

        output, hidden = model.rnn(embedding, hidden)

        prediction = model.fc(output)

        predicted_id = prediction.argmax(dim=-1).item()

        predicted_word = id_to_word[predicted_id]

        if predicted_word == "<EOS>":
            break

        generated_caption.append(predicted_word)

        current_word = torch.tensor(
            [[predicted_id]],
            dtype=torch.long
        ).to(device)

print("Generated Caption:", " ".join(generated_caption[1:]))
