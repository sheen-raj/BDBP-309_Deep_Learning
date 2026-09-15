import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Deep Neural Network
# ============================================================

class DeepNetwork(nn.Module):

    def __init__(self, weight_scale):

        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(20):

            layer = nn.Linear(
                10,
                10,
                bias=False
            )

            with torch.no_grad():

                layer.weight.copy_(
                    weight_scale * torch.eye(10)
                )

            self.layers.append(layer)

    def forward(self, x):

        activations = []

        for layer in self.layers:

            x = layer(x)

            # Tell PyTorch to store gradient of this activation
            x.retain_grad()

            activations.append(x)

        return x, activations


# ============================================================
# Calculate gradients
# ============================================================

def calculate_gradients(weight_scale):

    model = DeepNetwork(weight_scale)

    # Input
    x = torch.ones(1, 10)

    # Target
    target = torch.zeros(1, 10)

    # Forward pass
    output, activations = model(x)

    # Loss
    loss = ((output - target) ** 2).mean()

    # Backpropagation
    loss.backward()

    # Store activation gradients
    gradients = []

    for activation in activations:

        gradient = activation.grad.abs().mean().item()

        gradients.append(gradient)

    return gradients


# ============================================================
# Vanishing Gradient
# ============================================================

vanishing_gradients = calculate_gradients(0.5)


# ============================================================
# Exploding Gradient
# ============================================================

exploding_gradients = calculate_gradients(2.0)


# ============================================================
# Print gradients
# ============================================================

print("\nVanishing Gradient")

for i, gradient in enumerate(vanishing_gradients):

    print(
        f"Layer {i + 1:2d}: "
        f"{gradient:.6e}"
    )


print("\nExploding Gradient")

for i, gradient in enumerate(exploding_gradients):

    print(
        f"Layer {i + 1:2d}: "
        f"{gradient:.6e}"
    )


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 21),
    vanishing_gradients,
    marker="o",
    label="Weight scale = 0.5"
)

plt.plot(
    range(1, 21),
    exploding_gradients,
    marker="o",
    label="Weight scale = 2.0"
)

plt.yscale("log")

plt.xlabel("Layer")

plt.ylabel("Mean Absolute Activation Gradient")

plt.title("Vanishing and Exploding Gradients")

plt.legend()

plt.grid()

plt.show()