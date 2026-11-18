## Implement GAN  using PyTorch for image caption

"""
A very simple GAN for IMAGE CAPTIONING (PyTorch)

THE IDEA
--------
  Generator (G)      : looks at an IMAGE and writes a CAPTION (word by word).
  Discriminator (D)  : looks at an (image, caption) PAIR and says
                       "real pair"  (caption written by a human)  or
                       "fake pair"  (caption written by G).
  G is trained to fool D, so over time G learns to write captions that
  match the image.

TOY DATA (so it runs in seconds, no download)
---------------------------------------------
  Images   : 8x8 handwritten digits from scikit-learn
  Captions : "this is the number seven"  (always 5 words)

THE TRICKY PART: WORDS ARE DISCRETE
-----------------------------------
Picking a word (argmax) is not differentiable, so D's feedback cannot flow
back into G. The fix is the Gumbel-Softmax trick: G still outputs a real
one-hot word, but gradients are passed through as if it were a smooth
probability ("straight-through"). That is what F.gumbel_softmax(hard=True) does.

Run:
    pip install torch scikit-learn matplotlib numpy
    python gan_image_captioning.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-4
HIDDEN = 128

# ----------------------------------------------------------------------
# 2. DATA: images + captions
# ----------------------------------------------------------------------
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
vocab = ["<start>", "this", "is", "the", "number"] + digit_words      # 15 words
word_to_id = {w: i for i, w in enumerate(vocab)}
VOCAB = len(vocab)
CAP_LEN = 5                                                            # words per caption

# One caption per digit, stored as one-hot: shape (10 digits, 5 words, 15 vocab)
caption_ids = torch.tensor(
    [[word_to_id[w] for w in ["this", "is", "the", "number", digit_words[d]]] for d in range(10)]
)
caption_table = F.one_hot(caption_ids, VOCAB).float()


def ids_to_text(ids):
    return " ".join(vocab[i] for i in ids)


digits = load_digits()
images = torch.tensor(digits.data / 16.0, dtype=torch.float32)         # (1797, 64), values 0..1
labels = torch.tensor(digits.target)
img_train, img_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=0, stratify=labels
)
loader = DataLoader(TensorDataset(img_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
print("Train images:", len(img_train), "| Test images:", len(img_test))
print("Example real caption:", ids_to_text(caption_ids[7].tolist()))


# ----------------------------------------------------------------------
# 3. GENERATOR: image -> caption
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, HIDDEN), nn.ReLU())  # image -> feature vector
        self.embed = nn.Linear(VOCAB, HIDDEN, bias=False)                # one-hot word -> word vector
        self.gru = nn.GRUCell(HIDDEN, HIDDEN)                            # remembers the caption so far
        self.out = nn.Linear(HIDDEN, VOCAB)                              # scores for the next word

    def forward(self, img, sample=True):
        h = self.encoder(img)                                            # start memory = image info
        word = F.one_hot(torch.zeros(len(img), dtype=torch.long), VOCAB).float()   # <start>
        caption = []
        for _ in range(CAP_LEN):
            h = self.gru(self.embed(word), h)
            logits = self.out(h)
            if sample:
                word = F.gumbel_softmax(logits, tau=1.0, hard=True)      # random word, but trainable
            else:
                word = F.one_hot(logits.argmax(dim=1), VOCAB).float()    # best word (for testing)
            caption.append(word)
        return torch.stack(caption, dim=1)                               # (batch, 5, 15)


# ----------------------------------------------------------------------
# 4. DISCRIMINATOR: (image, caption) -> real or fake score
# ----------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 + CAP_LEN * VOCAB, HIDDEN), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, img, caption):
        return self.net(torch.cat([img, caption.flatten(1)], dim=1))     # one score per pair


G = Generator()
D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()


def caption_accuracy():
    """Percent of test images where G writes exactly the right caption."""
    G.eval()
    with torch.no_grad():
        generated = G(img_test, sample=False).argmax(dim=2)
    G.train()
    correct = (generated == caption_ids[y_test]).all(dim=1).float().mean().item()
    return 100 * correct


# ----------------------------------------------------------------------
# 5. TRAIN
# ----------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    d_total, g_total = 0.0, 0.0
    for img, y in loader:
        real_cap = caption_table[y]                  # the correct human caption
        ones = torch.ones(len(img), 1)
        zeros = torch.zeros(len(img), 1)

        # ---- Train D: real pair -> 1, fake pair -> 0, mismatched pair -> 0 ----
        fake_cap = G(img).detach()                   # detach: do not update G in this step
        wrong_img = img[torch.randperm(len(img))]    # real caption with a WRONG image
        loss_D = (bce(D(img, real_cap), ones)
                  + bce(D(img, fake_cap), zeros)
                  + bce(D(wrong_img, real_cap), zeros))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train G: make D believe the generated pair is real ----
        fake_cap = G(img)
        loss_G = bce(D(img, fake_cap), ones)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        d_total += loss_D.item()
        g_total += loss_G.item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | D loss {d_total / len(loader):.3f} | "
              f"G loss {g_total / len(loader):.3f} | test caption accuracy {caption_accuracy():.1f}%")

# ----------------------------------------------------------------------
# 6. SHOW RESULTS
# ----------------------------------------------------------------------
G.eval()
with torch.no_grad():
    generated = G(img_test[:10], sample=False).argmax(dim=2)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(img_test[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(ids_to_text(generated[i].tolist()).replace("this is the ", ""), fontsize=10)
    ax.axis("off")
    print(f"Image {i}: true = {digit_words[y_test[i]]:<6} | generated caption = "
          f"{ids_to_text(generated[i].tolist())}")
plt.suptitle("Captions written by the GAN generator")
plt.tight_layout()
plt.savefig("gan_captions.png", dpi=120)
print("\nSaved: gan_captions.png")