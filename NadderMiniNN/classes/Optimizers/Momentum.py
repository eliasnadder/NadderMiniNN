import numpy as np
from .Optimizer import Optimizer


# هلا ال momentum
# هيي عبارة عن ذاكرة للاتجاه
# بدال ما نحنا نعتم علة الاتجاه الحالي
# منراكم الاتجاهات يلي صارت بالماضي
# هاد الشي بقلل التذبذب عن طريق اذا في اتجاه نوصح ببطئ الحركة بهاد الاتجاه
# و بسرع الحركة بالاتجاه الصح
# منعمل مصفوفة لكل وزن نفس عدد اعمدة مصفوفة الاوزان
# منخزن بهي المصفوفة ذاكرة الحركة الخاصة  بكل وزن
# و كل مرة بدنا نحدث منحدث كل قيمة للوزن بالمصفوفة مناخد نسبة من الحركة القديمة و منضيفا

class Momentum(Optimizer):
    """Momentum optimizer"""

    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
