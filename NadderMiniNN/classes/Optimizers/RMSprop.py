import numpy as np
from .Optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSprop optimizer"""

    def __init__(self, lr=0.01, decay_rate=0.99, epsilon=1e-8):
        super().__init__(lr)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = self.decay_rate * self.h[key] + \
                (1 - self.decay_rate) * grads[key]**2
            params[key] -= self.lr * grads[key] / \
                (np.sqrt(self.h[key]) + self.epsilon)
