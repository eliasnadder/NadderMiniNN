class Optimizer:
    """Base class for all optimizers"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError
