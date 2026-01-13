import numpy as np
from ..Layer import Layer


class BatchNormalization(Layer):
    """Batch Normalization layer"""

    def __init__(self, input_size, momentum=0.9, epsilon=1e-8):
        super().__init__()
        self.params['gamma'] = np.ones(input_size)
        self.params['beta'] = np.zeros(input_size)
        self.momentum = momentum
        self.epsilon = epsilon

        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)

        self.batch_size = None
        self.xc = None
        self.std = None
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + self.epsilon)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.std = std

            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.epsilon)

        out = self.params['gamma'] * xn + self.params['beta']
        return out

    def backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xc / self.std * dout, axis=0)

        dxn = self.params['gamma'] * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.grads['gamma'] = dgamma
        self.grads['beta'] = dbeta

        return dx
