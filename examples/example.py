"""
Example usage of YourLastNameMiniNN library
Testing on Iris dataset
Network: Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NadderMiniNN.classes.Dense import Dense

from NadderMiniNN.classes.Activations.Sigmoid import Sigmoid
from NadderMiniNN.classes.Activations.Relu import Relu
from NadderMiniNN.classes.Activations.BatchNormalization import BatchNormalization
from NadderMiniNN.classes.Activations.SoftmaxWithLoss import SoftmaxWithLoss

from NadderMiniNN.neural_network import NeuralNetwork
from NadderMiniNN.classes.Optimizers.Adam import Adam
from NadderMiniNN.trainer import Trainer

# Load and prepare data


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


# Build the network: Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
def build_network():
    """Build neural network"""
    network = NeuralNetwork()

    # Add layers as required
    network.add_layer('dense1', Dense(4, 16))
    network.add_layer('sigmoid1', Sigmoid())
    network.add_layer('batchnorm1', BatchNormalization(16))
    network.add_layer('dense2', Dense(16, 8))
    network.add_layer('relu1', Relu())
    network.add_layer('dense3', Dense(8, 3))

    # Set loss layer
    network.set_loss_layer(SoftmaxWithLoss())

    # Initialize weights
    network.init_weights('he')

    return network


def main():
    """Main function"""
    print("="*60)
    print("Testing YourLastNameMiniNN Library on Iris Dataset")
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
    network = build_network()
    print("Network architecture:")
    for name, layer in network.layers.items():
        print(f"  {name}: {layer.__class__.__name__}")

    # Create optimizer and trainer
    print("\nSetting up training...")
    optimizer = Adam(lr=0.01)
    trainer = Trainer(network, optimizer, verbose=True)

    # Train
    print("\nStarting training...")
    print("-"*60)
    trainer.fit(
        X_train, y_train,
        X_test, y_test,
        epochs=100,
        batch_size=16,
        evaluate_interval=10
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
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:5], axis=1)

    iris = load_iris()
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Predicted: {iris.target_names[pred_classes[i]]}")
        print(f"  Actual: {iris.target_names[true_classes[i]]}")
        print(f"  Confidence: {predictions[i][pred_classes[i]]:.4f}")
        print()


if __name__ == "__main__":
    main()
