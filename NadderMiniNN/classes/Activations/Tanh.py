import numpy as np
from ..Layer import Layer


class Tanh(Layer):
    """Tanh activation function"""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx
