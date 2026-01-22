from ..layer1 import Layer


class Linear(Layer):
    """Linear activation (identity function)"""

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout