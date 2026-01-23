from ..layer1 import Layer1


class Linear(Layer1):
    """Linear activation (identity function)"""

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout