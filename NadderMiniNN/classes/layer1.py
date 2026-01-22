class Layer1:
    """Base class for all layers"""

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError
