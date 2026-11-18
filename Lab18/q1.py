"""
Representation Learning Lab (simple version)

We learn a 2-number representation (embedding) of each sample using 3 methods
and compare them on 3 kinds of data:

  DATA                              METHODS
  Text    : news posts (TF-IDF)     1. PCA          (linear)
  Image   : handwritten digits      2. Autoencoder  (neural network)
  Biology : breast cancer cells     3. t-SNE        (non-linear, for visualisation)

No labels are used to learn the embeddings. We only use the labels afterwards,
to colour the plots and to check how good each embedding is (kNN accuracy).

Run:
    pip install torch scikit-learn matplotlib numpy
    python representation_learning_simple.py
(The text data is downloaded once, so you need internet the first time.)
"""
import os
os.environ["HTTP_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://sheensagacious%40gmail.com:Freefire%4011@proxy.ibab.ac.in:3128"
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import fetch_20newsgroups, load_digits, load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



np.random.seed(0)
torch.manual_seed(0)


# ----------------------------------------------------------------------
# 1. LOAD THE THREE DATASETS
# Each one becomes a table X (samples x features) and labels y.
# ----------------------------------------------------------------------
def load_text():
    cats = ["sci.space", "sci.med", "rec.sport.hockey", "comp.graphics"]
    news = fetch_20newsgroups(subset="train", categories=cats,
                              remove=("headers", "footers", "quotes"))
    texts, y = news.data[:1500], news.target[:1500]
    # TF-IDF turns each document into numbers (importance of 1000 words)
    X = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(texts).toarray()
    X = X / X.std()                      # rescale so the neural network can learn
    return X.astype(np.float32), y


def load_image():
    digits = load_digits()               # 8x8 images flattened to 64 pixels
    return (digits.data / 16.0).astype(np.float32), digits.target


def load_biology():
    cancer = load_breast_cancer()        # 30 measurements per cell sample
    X = StandardScaler().fit_transform(cancer.data)
    return X.astype(np.float32), cancer.target


datasets = {"Text": load_text(), "Image": load_image(), "Biology": load_biology()}


# ----------------------------------------------------------------------
# 2. THE THREE REPRESENTATION METHODS
# Each takes X and returns a (samples x 2) embedding.
# ----------------------------------------------------------------------
def run_pca(X):
    # Finds the 2 directions with the most variation and projects onto them.
    return PCA(n_components=2).fit_transform(X)


def run_tsne(X):
    # Places points in 2D so that neighbours stay neighbours.
    return TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)


class Autoencoder(nn.Module):
    """input -> encoder -> 2 numbers -> decoder -> rebuilt input"""

    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(X, epochs=100):
    X_t = torch.tensor(X)
    model = Autoencoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        rebuilt = model(X_t)                  # squeeze to 2 numbers, then rebuild
        loss = loss_fn(rebuilt, X_t)          # how different is the rebuilt data?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Autoencoder epoch {epoch + 1}/{epochs}  loss {loss.item():.4f}")

    with torch.no_grad():
        return model.encoder(X_t).numpy()     # the 2-number representation


methods = {"PCA": run_pca, "Autoencoder": run_autoencoder, "t-SNE": run_tsne}


# ----------------------------------------------------------------------
# 3. RUN EVERYTHING: learn embeddings, score them, plot them
# ----------------------------------------------------------------------
fig, axes = plt.subplots(len(datasets), len(methods), figsize=(14, 12))
scores = {}

for row, (data_name, (X, y)) in enumerate(datasets.items()):
    print(f"\n{data_name}: {X.shape[0]} samples, {X.shape[1]} features")
    for col, (method_name, method) in enumerate(methods.items()):
        print(f"  Running {method_name} ...")
        Z = method(X)                                          # the embedding

        # How good is the embedding? 5-fold kNN accuracy using only the 2 numbers.
        acc = cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
        scores[(data_name, method_name)] = acc

        ax = axes[row, col]
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
        ax.set_title(f"{data_name} - {method_name}\nkNN accuracy: {acc:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig("representation_learning.png", dpi=120)
print("\nSaved plot: representation_learning.png")

# ----------------------------------------------------------------------
# 4. RESULTS TABLE
# ----------------------------------------------------------------------
print("\nkNN accuracy on the 2-D embeddings (higher = better)")
print(f"{'':<10}" + "".join(f"{m:>14}" for m in methods))
for data_name in datasets:
    print(f"{data_name:<10}" + "".join(f"{scores[(data_name, m)]:>14.2f}" for m in methods))