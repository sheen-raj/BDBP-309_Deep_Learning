#
# This script implements backward pass for a single neuron layer using for loops
#

import numpy as np

inputs = 1, 2, 3, 4
weights = 1, 1, 1, 1
bias = 1
y = 25
lr = 0.01
epochs = 10

# Activation function
def ReLU(z):
    return max(0, z)

# Derivative of Loss function
def loss_der(y_pred):
    return y_pred - y

# Derivative of ReLU function
def ReLU_der(z):
    if z < 0:
        return 0
    else:
        return 1

# Derivative of hypothesis function (Linear)
# def weight_der():
#     return inputs


def for_pass(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    a = ReLU(z)
    y_pred = a

    # Loss
    loss = 0.5 * (y_pred - y)**2
    return z, loss, y_pred


#======= Backward Pass (Crude) =============

def back_pass(y_pred, z, weights, bias):
    loss_d = loss_der(y_pred)
    # loss_d_global = loss_d
    z_d = ReLU_der(z)
    # z_d_global = z_d * loss_d_global (The global gradients at each step og back-pass should be updated like this for memory efficiency)
    delta = loss_d * z_d  # Global gradient at this step
    bias_gradient = delta  # or 1 * z_d_global
    weights_updated = []
    updated_bias = bias - lr * bias_gradient
    for i in range(len(weights)):
        gradient = delta * inputs[i]
        updated_weight = weights[i] - (lr * gradient)
        weights_updated.append(updated_weight)
    return weights_updated, updated_bias

for epoch in range(epochs):
    z, loss, y_pred = for_pass(inputs, weights, bias)
    weights_updated, updated_bias = back_pass(y_pred, z, weights, bias)
    print(f"\nEpoch {epoch + 1}")
    print(f"Prediction : {y_pred:.4f}")
    print(f"Loss       : {loss:.4f}")
    print(f"Weights    : {weights_updated}")
    print(f"Bias       : {updated_bias:.4f}")
    weights = weights_updated
    bias = updated_bias

