# NadderMiniNN - Mini Neural Network Library

A mini neural network library built from scratch using Python and NumPy only.

مكتبة شبكة عصبونية مصغرة مبنية من الصفر باستخدام Python و NumPy فقط.

## 📁 Project Structure / هيكل المشروع

```txt
NadderMiniNN/
├── NadderMiniNN/
│   ├── classes                         # All layers (Dense, Activations, etc.)
│   │   ├── Activitions 
│   │   │   ├── BatchNormalization.py
│   │   │   ├── Dropout.py
│   │   │   ├── Linear.py
│   │   │   ├── MeanSquaredError.py
│   │   │   ├── Relu.py
│   │   │   ├── Sigmoid.py
│   │   │   ├── SoftmaxWithLoss.py
│   │   │   └── Tanh.py
│   │   │   
│   │   ├── Optimizers                  # Optimization algorithms (SGD, Adam, etc.)
│   │   │   ├── AdaGrad.py
│   │   │   ├── Adam.py
│   │   │   ├── Momentum.py
│   │   │   ├── Optimizer
│   │   │   ├── RMSprop.py
│   │   │   └── SGD.py
│   │   │
│   │   ├── Dense.py
│   │   └── Layer.py
│   │   
│   ├── __init__.py
│   ├── neural_network.py               # Basic neural network structure
│   ├── trainer.py                      # Network trainer
│   └── hyperparameter_tuning.py        # Hyperparameter tuning
│   
├── examples/                           # Example usage of the library
│   ├── example_iris.py
│   └── example_mnist.py
│
├── tests/                              # Tests
│   └── test_library.py
│
├── setup.py 
├── requirements.txt
├── README.md
├── LICENSE
├── MANIFEST.in
└── .gitignore
```

## 🚀 Features / المميزات

### Available Layers / الطبقات المتاحة

- **Dense**: Fully Connected Layer
- **Activation Functions** / دوال التفعيل:
  - Linear
  - ReLU
  - Sigmoid
  - Tanh
- **Regularization** / التنظيم:
  - Dropout
  - Batch Normalization
- **Loss Functions** / دوال الخسارة:
  - Mean Squared Error
  - Softmax with Cross Entropy

### Optimization Algorithms / خوارزميات التحسين

- SGD (Stochastic Gradient Descent)
- Momentum
- AdaGrad
- Adam
- RMSprop

### Additional Features / ميزات إضافية

- Complete training system with accuracy and loss tracking
- Hyperparameter tuning (Grid Search & Random Search)
- Batch Normalization and Dropout support
- Weight initialization (He, Xavier)

## 💻 Requirements / متطلبات التشغيل

```bash
numpy>=1.19.0
scikit-learn>=0.24.0  # Only for the example
```

## 📖 Usage / كيفية الاستخدام

### Simple Example / مثال بسيط

```python
from layers import Dense, Relu, SoftmaxWithLoss
from neural_network import NeuralNetwork
from optimizers import Adam
from trainer import Trainer

# Build the network
network = NeuralNetwork()
network.add_layer('dense1', Dense(4, 16))
network.add_layer('relu1', Relu())
network.add_layer('dense2', Dense(16, 3))
network.set_loss_layer(SoftmaxWithLoss())
network.init_weights('he')

# Training
optimizer = Adam(lr=0.01)
trainer = Trainer(network, optimizer)
trainer.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

# Prediction
predictions = network.predict(X_test, train_mode=False)
```

### Complex Network / بناء شبكة معقدة

```python
# Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
network = NeuralNetwork()
network.add_layer('dense1', Dense(input_size, 64))
network.add_layer('sigmoid1', Sigmoid())
network.add_layer('batchnorm1', BatchNormalization(64))
network.add_layer('dense2', Dense(64, 32))
network.add_layer('relu1', Relu())
network.add_layer('dense3', Dense(32, num_classes))
network.set_loss_layer(SoftmaxWithLoss())
```

### Hyperparameter Tuning / ضبط المعاملات الفائقة

```python
from hyperparameter_tuning import HyperparameterTuning

# Define a function to build the network
def network_builder(hidden_size=32, **kwargs):
    network = NeuralNetwork()
    network.add_layer('dense1', Dense(4, hidden_size))
    network.add_layer('relu1', Relu())
    network.add_layer('dense2', Dense(hidden_size, 3))
    network.set_loss_layer(SoftmaxWithLoss())
    network.init_weights('he')
    return network

# Define a function to build the trainer
def trainer_builder(network, lr=0.01, **kwargs):
    optimizer = Adam(lr=lr)
    return Trainer(network, optimizer)

# Search for best parameters
tuner = HyperparameterTuning(network_builder, trainer_builder)
param_grid = {
    'hidden_size': [16, 32, 64],
    'lr': [0.001, 0.01, 0.1]
}
best_params = tuner.grid_search(param_grid, X_train, y_train, X_val, y_val)
```

## 🧪 Running the Example / تشغيل المثال

```bash
python example.py
```

This example does the following:

1. Load Iris dataset
2. Build a neural network
3. Train the network
4. Display results and accuracy

## 📊 Example Output / مثال على الخرج

```txt
============================================================
Testing NadderMiniNN Library on Iris Dataset
============================================================

Loading and preparing data...
Training samples: 120
Testing samples: 30
Features: 4
Classes: 3

Building network...
Network architecture:
  dense1: Dense
  sigmoid1: Sigmoid
  batchnorm1: BatchNormalization
  dense2: Dense
  relu1: Relu
  dense3: Dense

Starting training...
------------------------------------------------------------
Epoch 10/100 - Loss: 0.8234 - Train Acc: 0.6667 - Test Acc: 0.6667
Epoch 20/100 - Loss: 0.4521 - Train Acc: 0.8750 - Test Acc: 0.9000
...
Epoch 100/100 - Loss: 0.1234 - Train Acc: 0.9833 - Test Acc: 0.9667

============================================================
Final Results:
============================================================
Final Train Accuracy: 0.9833
Final Test Accuracy: 0.9667
```

## 🏗️ Architecture / البنية المعمارية

### Forward Propagation / الانتشار الأمامي

Each layer implements a `forward(x)` operation that computes the output from the input.

### Backward Propagation / الانتشار العكسي

Each layer implements a `backward(dout)` operation that computes the gradients.

### Weight Update / تحديث الأوزان

The Optimizer uses the computed gradients to update the network weights.

## 📝 Important Notes / ملاحظات مهمة

1. **Weight Initialization**: Use `he` for ReLU and `xavier` for Sigmoid/Tanh
2. **Batch Normalization**: Improves training speed and stability
3. **Dropout**: Helps prevent Overfitting
4. **Learning Rate**: Start with small values (0.001 - 0.01) with Adam

## 🔧 Extension / التوسعة

You can easily add:

- New layers by inheriting from `Layer` class
- New Optimizers by inheriting from `Optimizer` class
- New Loss functions

Example:

```python
class MyCustomLayer(Layer):
    def forward(self, x):
        # Your implementation here
        return output
    
    def backward(self, dout):
        # Your implementation here
        return dx
```

## 📄 License / الترخيص

This project is open source for academic use.

## ✍️ Author / المؤلف

Elias Nadder - Damascus University - ITE - Fourth Year

---

**Note**: This library is designed for educational purposes to understand how neural networks work internally.
