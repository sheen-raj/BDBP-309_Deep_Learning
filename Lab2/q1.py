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