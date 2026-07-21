'''Implement the following functions in Python from scratch. Do not use any library functions. You are allowed to use
numpy and matplotlib. Generate 100 equally spaced values between -10 and 10. Call this list as  z. Implement the
following functions and its derivative. Use class notes to find the expression for these functions. Use z as input and
plot both the function outputs and its derivative outputs.  Upload your code into Github and share it with me.
Sigmoid
Tanh
ReLU (Rectified Linear Unit)
Leaky ReLU
Softmax (no need for visualization)'''
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)

def sigmoid(z):
    a = []
    a_d = []

    for value in z:
        f = 1 / (1 + np.exp(-value))
        a.append(f)
        f_d = (np.exp(-value)) / ((1 + np.exp(-value))**2)
        a_d.append(f_d)

    plt.plot(z, a, label="Sigmoid")
    plt.plot(z, a_d, label="Sigmoid Derivative")
    plt.xlabel("Z")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
# sigmoid(z)

'''
What are the min and max values for the functions?
>>> 0 and 1
Is the output of the function zero-centred?
>>> No
What happens to the gradient when the input values are too small or too big?
>>> The output values goes closer to Zero
'''

def tanh(z):
    a = []
    a_d = []

    for value in z:
        f = (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value))
        a.append(f)
        f_d = 1 - (f**2)
        a_d.append(f_d)

    plt.plot(z, a, label="tan h")
    plt.plot(z, a_d, label="tan h Derivative")
    plt.xlabel("Z")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
# tanh(z)

'''
What are the min and max values for the functions?
>>> tanh : -1 and 1  ,  Derivative of tanh : 0 and 1
Is the output of the function zero-centred?
>>> Yes
What happens to the gradient when the input values are too small or too big?
>>> The output values are Zero
What is the relationship between sigmoid and tanh?
>>> tanh curve is similar to the sigmoid curve but is much more in a larger scale and also shifted in such a way that 
it is zero-centered while sigmoid is not.
'''

def ReLU(z):
    a = []
    a_d = []

    for value in z:
        if value < 0:
            a.append(0)
            a_d.append(0)
        elif value > 0:
            a.append(value)
            a_d.append(1)
        else:
            a.append(value)
            a_d.append(0)

    plt.plot(z, a, label="ReLU")
    plt.plot(z, a_d, label="ReLU Derivative")
    plt.xlabel("Z")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
# ReLU(z)

'''
What are the min and max values for the functions?
>>> 0 and 10
Is the output of the function zero-centred?
>>> No
What happens to the gradient when the input values are too small or too big?
>>> The output of the function is Binary
'''

def leakyReLU(z):
    alpha = 0.1
    a = []
    a_d = []

    for value in z:
        if value < 0:
            a.append(alpha*value)
            a_d.append(alpha)
        elif value > 0:
            a.append(value)
            a_d.append(1)
        else:
            a.append(value)
            a_d.append(1)

    plt.plot(z, a, label="Leaky ReLU")
    plt.plot(z, a_d, label="Leaky ReLU Derivative")
    plt.xlabel("Z")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
leakyReLU(z)

'''
What are the min and max values for the functions?
>>> 0 and 10
Is the output of the function zero-centred?
>>> No
What happens to the gradient when the input values are too small or too big?
>>> The output values are Binary
'''

# ================== SOFTMAX =================
def softmax(z):
    exp_z = np.exp(z)
    probabilities = exp_z / sum(exp_z)
    return probabilities

# output = softmax(z)
# print(output)
# print(np.sum(output))