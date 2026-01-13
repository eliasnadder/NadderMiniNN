import numpy as np

from NadderMiniNN.classes.Layer import Layer



class Dense(Layer):
    """Fully connected layer (Affine transformation)"""

    def __init__(self, input_size, output_size, weight_init_std=0.01):
        super().__init__()
        self.params['W'] = weight_init_std * \
            np.random.randn(input_size, output_size)
        self.params['b'] = np.zeros(output_size)
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.params['W']) + self.params['b']
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params['W'].T)
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx
