from ..layer1 import Layer1

# هي الدالة هيي يلي بتعدم القيم السالبة و بتترك القيم الموجبة
# هون هيي بتوفع بمشكلة Dying
# لانو بتعدم عصبونات


class Relu(Layer1):
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
