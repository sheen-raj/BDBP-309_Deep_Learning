## Develop a stable diffusion based method for image caption generation.

"""
A very simple DIFFUSION model for IMAGE CAPTIONING (PyTorch)

THE IDEA
--------
Stable Diffusion makes images by starting from random noise and removing the
noise step by step, guided by a text prompt. It works in a small "latent"
space instead of on raw pixels.

We flip this around to make CAPTIONS:
  * Latent space : each caption word is a small vector (an embedding),
                   so a 5-word caption is a 5 x 16 block of numbers.
  * Forward      : add noise to the caption embeddings, a little more each step.
  * Reverse      : a neural network learns to remove the noise, guided by the
                   IMAGE (the "condition", like the text prompt in Stable Diffusion).
  * Generate     : start from pure noise, denoise step by step, then turn each
                   cleaned vector back into the nearest real word.

HOW THIS TOY DIFFERS FROM REAL STABLE DIFFUSION
-----------------------------------------------
  Real Stable Diffusion        | This script
  -----------------------------|--------------------------------------
  latent = compressed image    | latent = caption word embeddings
  U-Net denoiser               | small MLP denoiser
  text condition (cross-attn.) | image condition (simple concatenation)
  predicts the noise           | predicts the clean caption (x0)

TOY DATA (no download, runs in about a minute)
----------------------------------------------
  Images   : 8x8 handwritten digits from scikit-learn
  Captions : "this is the number seven"  (always 5 words)

Run:
    pip install torch scikit-learn matplotlib numpy
    python diffusion_image_captioning.py
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
T = 100              # number of noising steps
EMB_DIM = 16         # size of each word vector
HIDDEN = 128
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

# ----------------------------------------------------------------------
# 2. DATA: images + captions
# ----------------------------------------------------------------------
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
vocab = ["this", "is", "the", "number"] + digit_words        # 14 words
word_to_id = {w: i for i, w in enumerate(vocab)}
VOCAB = len(vocab)
CAP_LEN = 5

# caption_ids[d] = the 5 word ids of the caption for digit d
caption_ids = torch.tensor(
    [[word_to_id[w] for w in ["this", "is", "the", "number", digit_words[d]]] for d in range(10)]
)


def ids_to_text(ids):
    return " ".join(vocab[i] for i in ids)


digits = load_digits()
images = torch.tensor(digits.data / 16.0, dtype=torch.float32)        # (1797, 64)
labels = torch.tensor(digits.target)
img_train, img_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=0, stratify=labels
)
loader = DataLoader(TensorDataset(img_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Each word gets a FIXED random vector. Words are far apart from each other,
# so "nearest word" is an easy way to turn a vector back into text.
word_vectors = torch.randn(VOCAB, EMB_DIM)

# ----------------------------------------------------------------------
# 3. THE NOISE SCHEDULE (forward process)
# ----------------------------------------------------------------------
# beta[t] = how much noise is added at step t.
# alpha_bar[t] = how much of the ORIGINAL caption is left after t steps
#                (1 = clean, 0 = pure noise).
beta = torch.linspace(1e-4, 0.1, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

# Numbers needed to take ONE step backwards (standard DDPM formulas)
coef_x0 = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
coef_xt = alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
sigma = (beta * (1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()


# ----------------------------------------------------------------------
# 4. THE DENOISER: (noisy caption, step t, image) -> clean caption
# ----------------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Linear(64, HIDDEN), nn.ReLU())   # the condition
        self.time_embed = nn.Embedding(T, HIDDEN)                              # "which step are we at?"
        self.net = nn.Sequential(
            nn.Linear(CAP_LEN * EMB_DIM + 2 * HIDDEN, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, CAP_LEN * EMB_DIM),
        )

    def forward(self, noisy_caption, t, image):
        x = torch.cat([noisy_caption.flatten(1), self.image_encoder(image), self.time_embed(t)], dim=1)
        return self.net(x).view(-1, CAP_LEN, EMB_DIM)


model = Denoiser()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ----------------------------------------------------------------------
# 5. GENERATE CAPTIONS (reverse process)
# ----------------------------------------------------------------------
def vectors_to_ids(x):
    """Turn caption vectors into word ids by picking the nearest word vector."""
    dist = torch.cdist(x.reshape(-1, EMB_DIM), word_vectors)
    return dist.argmin(dim=1).view(len(x), CAP_LEN)


@torch.no_grad()
def generate(image, show_steps=()):
    model.eval()
    x = torch.randn(len(image), CAP_LEN, EMB_DIM)                # start from pure noise
    history = {}
    for t in reversed(range(T)):
        t_batch = torch.full((len(image),), t, dtype=torch.long)
        x0_guess = model(x, t_batch, image)                      # "what is the clean caption?"
        x = coef_x0[t] * x0_guess + coef_xt[t] * x + sigma[t] * torch.randn_like(x)
        if t in show_steps:
            history[t] = vectors_to_ids(x0_guess)
    model.train()
    return vectors_to_ids(x), history


def caption_accuracy():
    """Percent of test images that get exactly the right caption."""
    generated, _ = generate(img_test)
    return 100 * (generated == caption_ids[y_test]).all(dim=1).float().mean().item()


# ----------------------------------------------------------------------
# 6. TRAIN
# ----------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for img, y in loader:
        clean = word_vectors[caption_ids[y]]                     # (B, 5, 16) real caption vectors

        t = torch.randint(0, T, (len(img),))                     # a random noise level per sample
        noise = torch.randn_like(clean)
        ab = alpha_bar[t].view(-1, 1, 1)
        noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise      # forward process in ONE line

        prediction = model(noisy, t, img)                        # try to recover the clean caption
        loss = F.mse_loss(prediction, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        msg = f"Epoch {epoch:3d}/{EPOCHS} | loss {total / len(loader):.4f}"
        if epoch % 20 == 0:
            msg += f" | test caption accuracy {caption_accuracy():.1f}%"
        print(msg)

# ----------------------------------------------------------------------
# 7. SHOW RESULTS
# ----------------------------------------------------------------------
# Watch a caption appear out of noise for ONE image.
steps_to_show = (99, 75, 50, 25, 0)
final, history = generate(img_test[:1], show_steps=steps_to_show)
print(f"\nDenoising one image (true digit: {digit_words[y_test[0]]})")
for t in steps_to_show:
    print(f"  step t={t:3d}: {ids_to_text(history[t][0].tolist())}")

# Caption 10 test images.
generated, _ = generate(img_test[:10])
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(img_test[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(ids_to_text(generated[i].tolist()).replace("this is the ", ""), fontsize=10)
    ax.axis("off")
    print(f"Image {i}: true = {digit_words[y_test[i]]:<6} | generated caption = "
          f"{ids_to_text(generated[i].tolist())}")
plt.suptitle("Captions generated by the diffusion model")
plt.tight_layout()
plt.savefig("diffusion_captions.png", dpi=120)
print("\nSaved: diffusion_captions.png")