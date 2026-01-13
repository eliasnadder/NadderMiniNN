"""
Example usage of NadderMiniNN library
Testing on Iris dataset
Network: Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

# Import from NadderMiniNN package
from NadderMiniNN import (
    Dense, Sigmoid, Relu, BatchNormalization, SoftmaxWithLoss,
    NeuralNetwork, Adam, Trainer, Dropout
)


def prepare_mnist_data():
    """تحميل وتحضير بيانات MNIST"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # تحويل الصور لمصفوفة أحادية وتطبيع القيم (Normalization)
    X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255.0

    # تحويل التسميات إلى One-hot encoding
    def to_one_hot(y, num_classes=10):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    return X_train, X_test, y_train, y_test

def prepare_data():
    """Load and prepare Iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to one-hot encoding
    y_onehot = np.zeros((y.size, y.max() + 1))
    y_onehot[np.arange(y.size), y] = 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def build_mnist_network():
    network = NeuralNetwork()

    # المدخلات 784 والمخرجات في الطبقة الأولى 128
    network.add_layer('dense1', Dense(784, 128))
    network.add_layer('batchnorm1', BatchNormalization(128))
    network.add_layer('relu1', Relu())
    network.add_layer('dropout1', Dropout(0.2)) # منع Overfitting

    # طبقة مخفية ثانية
    network.add_layer('dense2', Dense(128, 64))
    network.add_layer('relu2', Relu())

    # الطبقة الأخيرة 10 مخارج (للأرقام من 0-9)
    network.add_layer('dense3', Dense(64, 10))

    network.set_loss_layer(SoftmaxWithLoss())
    network.init_weights('he')

    return network

def build_network():
    """Build neural network as required:
    Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
    """
    network = NeuralNetwork()
    
    # Add layers as required
    network.add_layer('dense1', Dense(4, 16))
    network.add_layer('sigmoid1', Sigmoid())
    network.add_layer('batchnorm1', BatchNormalization(16))
    network.add_layer('dense2', Dense(16, 8))
    network.add_layer('relu1', Relu())
    network.add_layer('dropout1', Dropout(0.5)) # إضافة نسبة 50% حذف
    network.add_layer('dense3', Dense(8, 3))
    
    # Set loss layer
    network.set_loss_layer(SoftmaxWithLoss())
    
    # Initialize weights
    network.init_weights('he')
    
    return network

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def main():
    """Main function"""
    print("="*60)
    print("Testing NadderMiniNN Library on Iris Dataset")
    print("="*60)
    
    # Prepare data
    print("\nLoading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train.shape[1]}")
    
    # Build network
    print("\nBuilding network...")
    print("Architecture: Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss")
    network = build_network()
    
    print("\nNetwork layers:")
    for name, layer in network.layers.items():
        print(f"  {name}: {layer.__class__.__name__}")
    
    # Create optimizer and trainer
    print("\nSetting up training...")
    optimizer = Adam(lr=0.001)
    trainer = Trainer(network, optimizer, verbose=True)
    
    # Train
    print("\nStarting training...")
    print("-"*60)
    trainer.fit(
        X_train, y_train,
        X_test, y_test,
        epochs=100,
        batch_size=16,
        evaluate_interval=10,
        patience=50,
        min_delta=0.0001
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    train_acc = trainer.evaluate(X_train, y_train)
    test_acc = trainer.evaluate(X_test, y_test)
    print(f"Final Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Show predictions on a few test samples
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)
    predictions = network.predict(X_test[:5], train_mode=False)
    probs = softmax(predictions)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:5], axis=1)
    
    iris = load_iris()
    for i in range(5):
        correct = "✓" if pred_classes[i] == true_classes[i] else "✗"
        print(f"Sample {i+1}: {correct}")
        print(f"  Predicted: {iris.target_names[pred_classes[i]]}")
        print(f"  Actual: {iris.target_names[true_classes[i]]}")
        print(f"  Confidence: {probs[i][pred_classes[i]]:.4f}")
        print()


if __name__ == "__main__":
    main()