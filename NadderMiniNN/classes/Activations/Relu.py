from ..Layer import Layer


class Relu(Layer):
    """ReLU activation function"""

    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx