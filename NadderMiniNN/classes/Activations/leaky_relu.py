from ..layer1 import Layer1

# هون بتعالج مشكلة ال Dying
# بدال ما نعدم العصبون عالاخير و نموتو
# نحن منصفر القيمة تبع المشتقة


class LeakyReLU(Layer1):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.alpha = 0.01

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, dout):
        dout[self.mask] *= self.alpha
        dx = dout
        return dx
