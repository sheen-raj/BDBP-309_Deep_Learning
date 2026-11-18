## Demonstrate how both LSTM and GRU solves the vanishing gradient problem.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# Experiment 1
SEQ_LEN_DEMO1 = 100     # how many time steps the gradient travels back
INPUT_DIM = 16
HIDDEN_DIM = 64
TRIALS = 20             # average over this many random initialisations

# Experiments 2 and 3
MEMORY_LENGTHS = [10, 30, 60]   # sequence lengths to test
NUM_CLASSES = 5                 # the first token is one of 5 classes
NOISE_TOKENS = 5                # the other tokens are random noise
VOCAB = NUM_CLASSES + NOISE_TOKENS
EMBED_DIM = 16
MEM_HIDDEN = 32
TRAIN_STEPS = 500               # raise to 1000+ if LSTM/GRU do not learn the long task
BATCH = 64
LR = 0.01
SEEDS = [0, 1, 2]               # repeat each training run with 3 seeds


# ----------------------------------------------------------------------
# HELPER: build an RNN / LSTM / GRU layer
# ----------------------------------------------------------------------
def make_rnn(kind, input_dim, hidden_dim, gate_bias_init=True):
    """
    kind = "RNN", "LSTM" or "GRU".

    gate_bias_init=True applies a standard trick: start the "keep memory"
    gate with a bias of +1 so that it begins around 0.73 instead of 0.5.
      * LSTM: the forget gate (slice H:2H of the bias vector)
      * GRU : the update gate z (slice H:2H of the bias vector; in PyTorch
              z is the gate that KEEPS the old memory)
    The gates are still LEARNED during training. This just gives them a
    good starting point.
    """
    layer = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
    rnn = layer(input_dim, hidden_dim, batch_first=True)
    if gate_bias_init and kind in ("LSTM", "GRU"):
        H = hidden_dim
        with torch.no_grad():
            rnn.bias_ih_l0[H:2 * H].fill_(1.0)
            rnn.bias_hh_l0[H:2 * H].fill_(0.0)
    return rnn


# ======================================================================
# EXPERIMENT 1: How far does the gradient travel? (before training)
# ======================================================================
# We feed a random sequence of 100 steps, take the LAST hidden state, and
# ask: "how much does the input at each earlier step influence it?"
# That influence is the gradient of the last hidden state w.r.t. each input.
def gradient_per_step(kind, gate_bias_init):
    total = torch.zeros(SEQ_LEN_DEMO1)
    for trial in range(TRIALS):
        torch.manual_seed(trial)
        rnn = make_rnn(kind, INPUT_DIM, HIDDEN_DIM, gate_bias_init)
        x = torch.randn(1, SEQ_LEN_DEMO1, INPUT_DIM, requires_grad=True)
        out, _ = rnn(x)
        last_state = out[:, -1, :].sum()
        grad, = torch.autograd.grad(last_state, x)
        total += grad[0].norm(dim=1)           # size of the gradient at every time step
    average = (total / TRIALS).numpy()
    return average[::-1].copy()                # index 0 = last step, index 99 = 99 steps back


print("=" * 60)
print("EXPERIMENT 1: gradient size vs. distance back in time")
print("=" * 60)

curves = {
    "RNN":                         gradient_per_step("RNN", False),
    "LSTM (default init)":         gradient_per_step("LSTM", False),
    "LSTM (forget bias = 1)":      gradient_per_step("LSTM", True),
    "GRU (default init)":          gradient_per_step("GRU", False),
    "GRU (update bias = 1)":       gradient_per_step("GRU", True),
}

distances = [0, 10, 25, 50, 99]
print(f"{'Model':<26}" + "".join(f"{d:>12}" for d in distances) + "   <- steps back")
for name, c in curves.items():
    print(f"{name:<26}" + "".join(f"{c[d]:>12.2e}" for d in distances))
print("\nRead each row left to right: how much of the gradient is left after N steps.")
print("Note: with default init the gates start near 0.5, so even LSTM/GRU shrink the")
print("gradient. The point of the gates is that they can be LEARNED towards 1")
print("(see Experiments 2 and 3). A plain RNN has no such mechanism.")

# ======================================================================
# EXPERIMENT 2: "Remember the first token" task
# ======================================================================
# Each sequence = [class label, noise, noise, ..., noise].
# The model must output the class label after seeing the whole sequence.
# Random guessing gives 20% accuracy (5 classes).
# To succeed, information from step 0 must survive until the final step,
# and the gradient must flow all the way back to step 0 to learn this.
def make_batch(batch_size, length):
    labels = torch.randint(0, NUM_CLASSES, (batch_size,))
    noise = torch.randint(NUM_CLASSES, VOCAB, (batch_size, length - 1))
    return torch.cat([labels.unsqueeze(1), noise], dim=1), labels


class MemoryClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, EMBED_DIM)
        self.rnn = make_rnn(kind, EMBED_DIM, MEM_HIDDEN, gate_bias_init=True)
        self.fc = nn.Linear(MEM_HIDDEN, NUM_CLASSES)

    def from_embedding(self, e):
        out, _ = self.rnn(e)
        return self.fc(out[:, -1, :])          # use only the LAST time step

    def forward(self, x):
        return self.from_embedding(self.embed(x))


def train_memory_model(kind, length, seed):
    torch.manual_seed(seed)
    model = MemoryClassifier(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(TRAIN_STEPS):
        x, y = make_batch(BATCH, length)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x, y = make_batch(1000, length)
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    return model, accuracy


print("\n" + "=" * 60)
print("EXPERIMENT 2: remember the first token (chance level = 20%)")
print("=" * 60)

kinds = ["RNN", "LSTM", "GRU"]
accuracy = {k: {L: [] for L in MEMORY_LENGTHS} for k in kinds}
saved_models = {}          # keep the first-seed model of the middle length for Experiment 3
GRAD_LEN = MEMORY_LENGTHS[len(MEMORY_LENGTHS) // 2]

for length in MEMORY_LENGTHS:
    for kind in kinds:
        for seed in SEEDS:
            model, acc = train_memory_model(kind, length, seed)
            accuracy[kind][length].append(acc)
            if length == GRAD_LEN and seed == SEEDS[0]:
                saved_models[kind] = model
        mean, std = np.mean(accuracy[kind][length]), np.std(accuracy[kind][length])
        print(f"length {length:>3} | {kind:<5} | accuracy {100 * mean:5.1f}% +/- {100 * std:4.1f}")
    print()

# ======================================================================
# EXPERIMENT 3: After training, how much does step 0 still matter?
# ======================================================================
# We take the trained models (sequence length = GRAD_LEN) and measure the
# gradient of the CORRECT class score with respect to the input at every
# time step. A model that remembers the first token must be sensitive to
# step 0, so the gradient there must not have vanished.
def trained_gradient_per_step(model, length):
    torch.manual_seed(123)
    x, y = make_batch(200, length)
    e = model.embed(x).detach().requires_grad_(True)
    scores = model.from_embedding(e).gather(1, y.unsqueeze(1)).sum()
    grad, = torch.autograd.grad(scores, e)
    return grad.norm(dim=2).mean(dim=0).numpy()       # one number per time step


print("=" * 60)
print(f"EXPERIMENT 3: gradient at each time step after training (length {GRAD_LEN})")
print("=" * 60)
trained_grads = {k: trained_gradient_per_step(saved_models[k], GRAD_LEN) for k in kinds}
for k in kinds:
    g = trained_grads[k]
    print(f"{k:<5} | gradient at step 0: {g[0]:.2e} | at last step: {g[-1]:.2e} "
          f"| ratio first/last: {g[0] / (g[-1] + 1e-30):.2e}")
print("\nA ratio close to 0 means the first input has (almost) no influence on the")
print("output, i.e. the gradient vanished and the model could not learn to use it.")

# ======================================================================
# PLOTS
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
styles = {
    "RNN": ("tab:red", "-"),
    "LSTM (default init)": ("tab:blue", "--"),
    "LSTM (forget bias = 1)": ("tab:blue", "-"),
    "GRU (default init)": ("tab:green", "--"),
    "GRU (update bias = 1)": ("tab:green", "-"),
}
for name, c in curves.items():
    color, ls = styles[name]
    axes[0].plot(np.maximum(c, 1e-30), label=name, color=color, linestyle=ls)
axes[0].set_yscale("log")
axes[0].set_xlabel("Steps back in time")
axes[0].set_ylabel("Gradient size (log scale)")
axes[0].set_title("Exp 1: gradient vs. distance (untrained)")
axes[0].legend(fontsize=8)

bar_colors = {"RNN": "tab:red", "LSTM": "tab:blue", "GRU": "tab:green"}
x = np.arange(len(MEMORY_LENGTHS))
for i, k in enumerate(kinds):
    means = [100 * np.mean(accuracy[k][L]) for L in MEMORY_LENGTHS]
    stds = [100 * np.std(accuracy[k][L]) for L in MEMORY_LENGTHS]
    axes[1].bar(x + (i - 1) * 0.27, means, width=0.27, yerr=stds, label=k, color=bar_colors[k])
axes[1].axhline(100 / NUM_CLASSES, color="gray", linestyle=":", label="chance (20%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"length {L}" for L in MEMORY_LENGTHS])
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Exp 2: remember the first token")
axes[1].legend(fontsize=8)

for k in kinds:
    axes[2].plot(np.maximum(trained_grads[k], 1e-30), label=k, color=bar_colors[k])
axes[2].set_yscale("log")
axes[2].set_xlabel("Time step of the input (0 = the token to remember)")
axes[2].set_ylabel("Gradient size (log scale)")
axes[2].set_title(f"Exp 3: trained models (length {GRAD_LEN})")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("vanishing_gradient_demo.png", dpi=120)
print("\nSaved plot: vanishing_gradient_demo.png")