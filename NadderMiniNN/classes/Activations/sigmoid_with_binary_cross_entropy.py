from ..layer2 import Layer2
import numpy as np


class SigmoidWithBinaryCrossEntropy(Layer2):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = binary_cross_entropy(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def binary_cross_entropy(y, t):
    eps = 1e-7
    return -np.mean(t * np.log(y + eps) + (1 - t) * np.log(1 - y + eps))
