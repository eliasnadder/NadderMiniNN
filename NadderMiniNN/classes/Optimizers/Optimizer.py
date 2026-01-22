# كل كلاس فيه دالة update
# هيك وحدنا الشكل و سهلنا تبديل بين ال optimizers

# هلا اهم مشكلتين بحلوهن optimizers
# هنن Overshooting and slow convergence

class Optimizer:
    """Base class for all optimizers"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError
