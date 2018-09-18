import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.paramas
        out = x @ W + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        In, Hidden, Out = input_size, hidden_size, output_size

        W1 = np.random.randn(In, Hidden)
        b1 = np.random.randn(Hidden)
        W2 = np.random.randn(Hidden, Out)
        b2 = np.random.randn(Out)

        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
