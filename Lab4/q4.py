from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# 3 layer neural network for this classification

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential

# model = Sequential([
#     Input(shape=(2,)),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(5, "relu"),
#     Dense(1, "sigmoid")
# ])
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
# model.fit(X, y, batch_size=32, epochs=100, verbose=0)
# print(model.evaluate(X, y))

# The accuracy goes down after certain number of neuron layers

# Look at weights in each layer

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import RandomNormal

class WeightCapture(Callback):
    "Capture the weights of each layer of the model"

    def __init__(self, model):
        super().__init__()
        # self.model = model
        self.weights = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)  # remember the epoch axis
        weight = {}
        for layer in model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)


def make_mlp(activation, initializer, name):
    "Create a model with specified activation and initalizer"
    model = Sequential([
        Input(shape=(2,), name=name+"0"),
        Dense(5, activation=activation, kernel_initializer=initializer, name=name+"1"),
        Dense(5, activation=activation, kernel_initializer=initializer, name=name+"2"),
        Dense(5, activation=activation, kernel_initializer=initializer, name=name+"3"),
        Dense(5, activation=activation, kernel_initializer=initializer, name=name+"4"),
        Dense(1, activation="sigmoid", kernel_initializer=initializer, name=name+"5")
    ])
    return model

initializer = RandomNormal(mean=0.0, stddev=1.0)
batch_size = 32
n_epochs = 100

model = make_mlp("sigmoid", initializer, "sigmoid")
capture_cb = WeightCapture(model)
capture_cb.on_epoch_end(-1)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
model.fit(X, y, batch_size=batch_size, epochs=n_epochs, callbacks=[capture_cb], verbose=1)

print(model.evaluate(X,y))


def plotweight(capture_cb):
    "Plot the weights' mean and s.d. across epochs"
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 10))
    ax[0].set_title("Mean weight")
    for key in capture_cb.weights[0]:
        ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in capture_cb.weights[0]:
        ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
    ax[1].legend()
    plt.show()


plotweight(capture_cb)