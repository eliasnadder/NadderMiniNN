import numpy as np
from .Optimizer import Optimizer

# هيي عبارة عن حجم الخطوة
#  باراميتر بيتخدث كتير منصفر معدل التعلم
# باراميتر بيتحدث قليل منكبر معدل التعلم
# بدال ما يكون عنا خطوة ثابتة للوزن بصير عنا خطوة مختلفة
# منعمل مصفوفة بعدد اعمدة مصفوفة الباراميتر
# و كل مرة بيتخدث الباراميتر على اساس حجم الاتجاه السابق


class AdaGrad(Optimizer):
    """AdaGrad optimizer"""

    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.h = None
        self.epsilon = epsilon

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / \
                (np.sqrt(self.h[key]) + self.epsilon)
