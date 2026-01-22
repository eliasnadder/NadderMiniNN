import numpy as np
from ..layer1 import Layer1

# هون نحنا عنا مشكلة Vanishing
# القيم بتصير بين ال 0 و 1
# لانو عند المشتقات الكبير و المشتقات الصغيرةبتصير القيم شبه ثابته و هيك المشتقات معرضة للاختفاء
# ببطئ التعلم لانو غير متمركز حول الصغر و القيم موجبة


class Sigmoid(Layer1):
    """Sigmoid activation function"""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
