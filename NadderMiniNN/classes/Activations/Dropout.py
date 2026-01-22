import numpy as np
from ..layer1 import Layer1


class Dropout(Layer1):
    """Dropout layer for regularization"""

    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
