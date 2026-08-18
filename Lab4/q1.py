# Implement a 1-layer (input - output layer) neural network from scratch for the following dataset.
# This includes implementing forward and backward passes from scratch.
# Print the training loss and plot it over 1000 iterations.

import numpy as np

inputs = [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]
weights = 1, 1, 1
bias = 1
y = [0], [1], [1], [0]
lr = 0.01
epochs = 1000

# Converting the inputs and weights into vectors
inputs = np.array(inputs).reshape(4,3)
weights = np.array(weights).reshape(3, 1)
bias = np.array(bias).reshape(1,1)

# Activation function
def ReLU(z):
    return np.maximum(0, z)

# Derivative of Loss function
def loss_der(y_pred):
    return y_pred - y

# Derivative of ReLU function
def ReLU_der(z):
    return (z > 0).astype(float)

def for_pass(inputs, weights, bias):
    z = inputs @ weights + bias
    a = ReLU(z)
    y_pred = a
    # Loss
    loss = 0.5 * (y_pred - y)**2
    return z, loss, y_pred

def back_pass(inputs, weights, bias, y_pred, z):
    loss_d = loss_der(y_pred)
    relu_d = ReLU_der(z)
    delta = loss_d * relu_d
    # print(inputs.shape)
    # print(delta.shape)
    weight_gradient = inputs.T @ delta
    # print(weight_gradient.shape)
    bias_gradient = np.sum(delta, axis=0, keepdims=True)
    weights = weights - lr * weight_gradient
    bias = bias - lr * bias_gradient
    return weights, bias


for epoch in range(epochs):
    z, loss, y_pred = for_pass(inputs, weights, bias)
    print(y_pred.shape)
    weights_updated, updated_bias = back_pass(inputs, weights, bias, y_pred, z)
    print(f"\nEpoch {epoch + 1}")
    print(f"Prediction : {y_pred}")
    print(f"Loss       : {loss}")
    print(f"Weights    : {weights_updated}")
    print(f"Bias       : {updated_bias}")
    weights = weights_updated
    bias = updated_bias
