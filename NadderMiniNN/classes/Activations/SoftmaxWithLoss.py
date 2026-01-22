import numpy as np

from ..layer2 import Layer2

# Output and Loss
# هلا نحنا عنا المخرجات اما تصنيفات ثنائية او اكثر
# هلا نحنا عم نستخدم السوفت ماكس منشان نعمل الخرج ك احتمالات
# و بعد ما نشوف الاحتمالات منعرف شو الخرج و منشوف الخسارة بالنسبة لهاد الخرج يلي عطيتو الشبكة
# و من اجل التصنيفات الثنائية ممكن سبغمويد
# هلا لنحسب الخسارة نحنا عنا cross entropy , mean squared , binary cross entropy
# هلا مع المشاكل التصنيفية cross entropy انسب مع الاحتمالات
# لانو لما بطالع الخرج بقدر يعاقب عقاب شديد لما بكون النموذج واثق بالخطأ
# منلاحظ الفرق بالعقابات
# اما Mean هون مافي منطق لانو العقاب على الاجابة الصح ما بيخف كتير عن العقاب للاجابة الخطأ
# لهيك المشاكل الخطية بتناسبها اكتر Mean
# منلاحظ انو نحن عن ندمج السوفت ماكس مع cross entro
# sigmoid with binary cross
# لانو هيك بصيروا بطبقة وحدة منخفف ذاكرة بدال ما خزن مخرجات السوفت ماكس لحال و بعدا الخسارة
# بخفف ذاكرة بصير اسرع و اقل اخطاء
# و طبعا منلاحظ دالة backward يلي فيه بدنا نبلش نرجع لورا
# رح نستق دالة الهسارة بالنسبة ل x و نرجع بالطبقات لورا
# لحتى نشتق دالة الخسارة بالنسبة للاوزان و الانحيازات


class SoftmaxWithLoss(Layer2):
    """Softmax activation with Cross Entropy loss"""

    def __init__(self):
        super().__init__()
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = self._softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        self.loss = self._cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    def _softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def _cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
