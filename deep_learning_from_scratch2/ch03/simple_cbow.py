import sys
import numpy as np

sys.path.append("..")
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.i_layer0 = MatMul(W_in)
        self.i_layer1 = MatMul(W_in)
        self.o_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.i_layer0, self.i_layer1, self.o_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in
