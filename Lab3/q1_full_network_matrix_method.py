#
# This script is the backward pass in FFN from scratch
#

import numpy as np

inputs = 1, 2, 3, 4
y = 25
lr = 0.01
epochs = 100
num_layers = 3
num_neurons_1 = 3
num_neurons_2 = 2
num_neurons_3 = 1

# Converting the inputs and weights into vectors
X = np.array(inputs).reshape(1,4)
Y = np.array([[y]])

# He initialization for weights
W1 = np.random.randn(4, 3) * np.sqrt(2 / 4)
B1 = np.random.rand(1, 3)

W2 = np.random.randn(3, 2) * np.sqrt(2 / 3)
B2 = np.random.rand(1, 2)

W3 = np.random.randn(2, 1) * np.sqrt(2 / 2)
B3 = np.random.rand(1, 1)



# Activation function
def ReLU(z):
    return np.maximum(0, z)

# Derivative of Loss function
def loss_der(y_pred):
    return y_pred - Y

# Derivative of ReLU function
def ReLU_der(z):
    return (z > 0).astype(float)

def for_pass(X, W1, B1, W2, B2, W3, B3):
    Z1 = X @ W1 + B1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2 + B2
    A2 = ReLU(Z2)
    Z3 = A2 @ W3 + B3
    A3 = Z3  # Final output doesnt require the  activation function

    y_pred = A3
    # Loss
    loss = 0.5 * (y_pred - Y)**2
    return Z1, A1, Z2, A2, Z3, A3, y_pred, loss

def back_pass(X, W1, B1, W2, B2, W3, B3, y_pred, Z1, A1, Z2, A2):
    # Output layer
    dA3 = loss_der(y_pred)
    dZ3 = dA3  # Final output doesnt require the  activation function

    dW3 = A2.T @ dZ3
    dB3 = dZ3

    # Hidden layer 2
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * ReLU_der(Z2)

    dW2 = A1.T @ dZ2
    dB2 = dZ2

    # Hidden layer 1
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * ReLU_der(Z1)

    dW1 = X.T @ dZ1
    dB1 = dZ1

    # Update
    W3 = W3 - lr * dW3
    B3 = B3 - lr * dB3

    W2 = W2 - lr * dW2
    B2 = B2 - lr * dB2

    W1 = W1 - lr * dW1
    B1 = B1 - lr * dB1
    return W1, B1, W2, B2, W3, B3


for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2, Z3, A3, y_pred, loss = for_pass(X,W1, B1,
        W2, B2,W3, B3)

    # Backward pass
    W1, B1, W2, B2, W3, B3 = back_pass(X, W1, B1, W2, B2,
        W3, B3, y_pred, Z1, A1, Z2, A2)

    print(f"\nEpoch {epoch + 1}")
    print(f"Prediction : {y_pred}")
    print(f"Loss       : {loss}")

    # print(f"W1 shape   : {W1.shape}")
    # print(f"W2 shape   : {W2.shape}")
    # print(f"W3 shape   : {W3.shape}")
    #
    # print(f"B1 shape   : {B1.shape}")
    # print(f"B2 shape   : {B2.shape}")
    # print(f"B3 shape   : {B3.shape}")