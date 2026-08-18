#
# This script is the backward pass in FFN from scratch
#

import numpy as np

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0], [1], [1], [0]
lr = 0.1
epochs = 5000
num_layers = 3
num_neurons_1 = 2
num_neurons_2 = 1


# Converting the inputs and weights into vectors
X = np.array(inputs).reshape(4, 2)
Y = np.array(y).reshape(4, 1)

# He initialization for weights
W1 = np.random.randn(2, 2) * np.sqrt(2 / 2)
B1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1) * np.sqrt(2 / 2)
B2 = np.zeros((1, 1))



# Activation function
def ReLU(z):
    return np.maximum(0, z)

# Derivative of Loss function
def loss_der(y_pred):
    return (y_pred - Y) / Y.shape[0]

# Derivative of ReLU function
def ReLU_der(z):
    return (z > 0).astype(float)

def for_pass(X, W1, B1, W2, B2):
    Z1 = X @ W1 + B1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2 + B2
    A2 = Z2  # Final output doesnt require the  activation function

    y_pred = A2
    # Loss
    loss = np.mean(0.5 * (y_pred - Y)**2)
    return Z1, A1, Z2, A2, y_pred, loss

def back_pass(X, W1, B1, W2, B2, y_pred, Z1, A1, Z2, A2):
    # Output layer
    dA2 = loss_der(y_pred)
    dZ2 = dA2  # Final output doesnt require the  activation function

    dW2 = A1.T @ dZ2
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer 2
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * ReLU_der(Z1)

    dW1 = X.T @ dZ1
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W2 = W2 - lr * dW2
    B2 = B2 - lr * dB2

    W1 = W1 - lr * dW1
    B1 = B1 - lr * dB1
    return W1, B1, W2, B2


for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2,  y_pred, loss = for_pass(X,W1, B1,
        W2, B2)

    # Backward pass
    W1, B1, W2, B2, = back_pass(X, W1, B1, W2, B2,
         y_pred, Z1, A1, Z2, A2)

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