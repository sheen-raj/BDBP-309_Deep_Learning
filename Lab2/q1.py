'''Consider the following two networks.  W is a matrix, x is a vector, z is a vector, and a is a vector. y^ is a scalar
and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).  Initialize X and W
randomly. Every neuron has a bias term.

1. Implement forward pass for the above two networks. Print activation values for each neuron at each layer.
   Print the loss value (y^).

2. Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z and a are vectors.
   The implementation should not contain any loops.
'''

import numpy as np
import random

print("\n====================\nQ1\n====================")
print("\n===========Network 1===================")

x1, x2, x3, x4, w11, w12, w13, w14 = [round(random.uniform(-1, 1), 2) for _ in range(8)]

# Bias term
b1 = round(random.uniform(-1, 1), 2)
print("Bias =",b1)

# Activation Function
def ReLU(z):
    if z < 0:
        return 0
    elif z > 0:
        return z
    else:
        return z

z1 = (x1 * w11) + (x2 * w12) + (x3 * w13) + (x4 * w14) + b1
a = ReLU(z1)
y = a

print(f"x = {x1:.2f}, {x2:.2f}, {x3:.2f}, {x4:.2f}")
print(f"w = {w11:.2f}, {w12:.2f}, {w13:.2f}, {w14:.2f}")
print(f"z = {z1:.4f}")
print(f"y^ = {y:.4f}")

# Doing the same thing with Network 2 having more layers and neurons

print("\n===========Network 2===================")

# LAYER 1
# Inputs
x1, x2, x3, x4 = [round(random.uniform(-1, 1), 2) for _ in range(4)]
X = [x1, x2, x3, x4]
print("Inputs:", X)

# Initializing weights and Bias terms per layer
w1 = [[random.uniform(-1,1) for _ in range(4)] for _ in range(3)] #neuron 1
b1 = [random.uniform(-1,1) for _ in range(3)]
w2 = [[random.uniform(-1,1) for _ in range(3)] for _ in range(2)] #neuron 2
b2 = [random.uniform(-1,1) for _ in range(2)]
w3 = [[random.uniform(-1,1) for _ in range(2)]] #neuron 3
b3 = [random.uniform(-1,1)]


# Activation function for each node computation with bias term
# z1 = (x1 * w11) + (x2 * w12) + (x3 * w13) + (x4 * w14) + b1
# a1 = ReLU(z1)
#
# z2 = (x1 * w21) + (x2 * w22) + (x3 * w23) + (x4 * w24) + b2
# a2 = ReLU(z2)
#
# z3 = (x1 * w31) + (x2 * w32) + (x3 * w33) + (x4 * w34) + b3
# a3 = ReLU(z3)

def activation_per_layer(inputs, weights, biases):
    activations = []
    for i in range(len(weights)):
        z = 0
        for j in range(len(inputs)):
            z += inputs[j] * weights[i][j]  # compute activation values for each neuron
        z += biases[i]  # add bias term to each activation values

        a = ReLU(z)
        activations.append(a)
    # print(activations)
    return activations

a1 = activation_per_layer(X, w1, b1)
# print(a1)

a2 = activation_per_layer(a1, w2, b2)
# print(a2)

a3 = activation_per_layer(a2, w3, b3)
# print(a3)

y = a3[0]

print(f"Layer 1 activations: {a1}")
print(f"Layer 2 activations: {a2}")
print(f"Output activation : {a3}")
print(f"y^ = {y:.4f}")
# -----------------------------------------------------------
# -----------------------------------------------------------
# ======================= Q2 ================================
# -----------------------------------------------------------
# -----------------------------------------------------------
print("\n====================\nQ2 (Using Matrix Method)\n====================")
# Input Layer
X = np.random.randn(1, 4)
# Initializing weights and bias terms
W1 = np.random.randn(4, 3)
B1 = np.random.randn(1, 3)
W2 = np.random.randn(3, 2)
B2 = np.random.randn(1, 2)
W3 = np.random.randn(2, 1)
B3 = np.random.randn(1, 1)

def activation_per_layer_matrix_method(inputs, weights, biases):
    z = np.dot(inputs, weights) + biases
    a = np.maximum(0, z)  #ReLU
    return a

A1 = activation_per_layer_matrix_method(X, W1, B1)
A2 = activation_per_layer_matrix_method(A1, W2, B2)
A3 = activation_per_layer_matrix_method(A2, W3, B3)

# print(f"{A1}\n{A2}\n{A3}")
print("A1 =", A1)
print("A2 =", A2)
print("A3 =", A3)
print("Output =", A3[0, 0])

