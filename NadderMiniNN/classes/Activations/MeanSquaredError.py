import numpy as np
from ..Layer import Layer


class MeanSquaredError(Layer):
    """Mean Squared Error loss"""

    def __init__(self):
        super().__init__()
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = x
        self.t = t
        loss = 0.5 * np.sum((self.y - self.t)**2) / x.shape[0]
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
