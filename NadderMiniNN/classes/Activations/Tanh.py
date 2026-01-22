import numpy as np
from ..layer1 import Layer1

# بتخلي التمركز بين 1 و -1
# بتعالج شوي مشكلة التلاشي للمشتقات


class Tanh(Layer1):
    """Tanh activation function"""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        # self.out = np.tanh(x)
        self.out = ( 2 / ( 1 + np.exp(-2 * x)) ) - 1
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx
