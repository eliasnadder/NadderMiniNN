import numpy as np
from ..Layer import Layer


class Sigmoid(Layer):
    """Sigmoid activation function"""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
