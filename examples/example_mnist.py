"""
Advanced example using YourLastNameMiniNN library
Testing on MNIST Digits dataset with more complex architecture
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NadderMiniNN.classes.Dense import Dense

from NadderMiniNN.classes.Activations.Relu import Relu
from NadderMiniNN.classes.Activations.Dropout import Dropout
from NadderMiniNN.classes.Activations.BatchNormalization import BatchNormalization
from NadderMiniNN.classes.Activations.SoftmaxWithLoss import SoftmaxWithLoss

from NadderMiniNN.neural_network import NeuralNetwork
from NadderMiniNN.classes.Optimizers.Adam import Adam
from NadderMiniNN.trainer import Trainer
from NadderMiniNN.hyperparameter_tuning import HyperparameterTuning


def prepare_digits_data():
    """Load and prepare MNIST digits dataset"""
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert labels to one-hot encoding
    y_onehot = np.zeros((y.size, 10))
    y_onehot[np.arange(y.size), y] = 1

    # Split data: 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_deep_network(input_size=64, hidden_sizes=[128, 64, 32],
                       output_size=10, dropout_rate=0.3):
    """
    Build a deep neural network with multiple hidden layers
    """
    network = NeuralNetwork()

    # First hidden layer
    network.add_layer('dense1', Dense(input_size, hidden_sizes[0]))
    network.add_layer('batchnorm1', BatchNormalization(hidden_sizes[0]))
    network.add_layer('relu1', Relu())
    network.add_layer('dropout1', Dropout(dropout_rate))

    # Second hidden layer
    network.add_layer('dense2', Dense(hidden_sizes[0], hidden_sizes[1]))
    network.add_layer('batchnorm2', BatchNormalization(hidden_sizes[1]))
    network.add_layer('relu2', Relu())
    network.add_layer('dropout2', Dropout(dropout_rate))

    # Third hidden layer
    network.add_layer('dense3', Dense(hidden_sizes[1], hidden_sizes[2]))
    network.add_layer('batchnorm3', BatchNormalization(hidden_sizes[2]))
    network.add_layer('relu3', Relu())

    # Output layer
    network.add_layer('dense4', Dense(hidden_sizes[2], output_size))

    # Set loss layer
    network.set_loss_layer(SoftmaxWithLoss())

    # Initialize weights with He initialization
    network.init_weights('he')

    return network


def train_with_early_stopping(network, optimizer, X_train, y_train,
                              X_val, y_val, epochs=200, batch_size=32,
                              patience=10):
    """
    Train with early stopping based on validation accuracy
    """
    trainer = Trainer(network, optimizer, verbose=True)

    best_val_acc = 0
    patience_counter = 0

    print("\nTraining with Early Stopping...")
    print("-" * 60)

    for epoch in range(epochs):
        # Train one epoch
        train_size = X_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)

        # Shuffle
        idx = np.random.permutation(train_size)
        X_train_shuffled = X_train[idx]
        y_train_shuffled = y_train[idx]

        epoch_loss = 0
        for i in range(iter_per_epoch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_size)
            x_batch = X_train_shuffled[start_idx:end_idx]
            t_batch = y_train_shuffled[start_idx:end_idx]

            loss = trainer.train_step(x_batch, t_batch)
            epoch_loss += loss

        epoch_loss /= iter_per_epoch

        # Evaluate
        if (epoch + 1) % 5 == 0:
            train_acc = network.accuracy(X_train, y_train)
            val_acc = network.accuracy(X_val, y_val)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f} - "
                  f"Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"  → New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

    return trainer


def hyperparameter_search_example(X_train, y_train, X_val, y_val):
    """
    Example of hyperparameter tuning
    """
    print("\n" + "="*60)
    print("Hyperparameter Tuning")
    print("="*60)

    def network_builder(hidden_size=64, dropout_rate=0.3, **kwargs):
        network = NeuralNetwork()
        network.add_layer('dense1', Dense(64, hidden_size))
        network.add_layer('relu1', Relu())
        network.add_layer('dropout1', Dropout(dropout_rate))
        network.add_layer('dense2', Dense(hidden_size, 10))
        network.set_loss_layer(SoftmaxWithLoss())
        network.init_weights('he')
        return network

    def trainer_builder(network, lr=0.01, **kwargs):
        optimizer = Adam(lr=lr)
        return Trainer(network, optimizer, verbose=False)

    tuner = HyperparameterTuning(network_builder, trainer_builder)

    # Grid search
    param_grid = {
        'hidden_size': [64, 128],
        'lr': [0.001, 0.01],
        'dropout_rate': [0.2, 0.3]
    }

    best_params = tuner.grid_search(
        param_grid, X_train, y_train, X_val, y_val,
        epochs=30, batch_size=32, verbose=True
    )

    return best_params


def main():
    """Main function"""
    print("="*60)
    print("Testing YourLastNameMiniNN Library on MNIST Digits")
    print("="*60)

    # Prepare data
    print("\nLoading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_digits_data()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train.shape[1]}")

    # Option 1: Train with predefined architecture
    print("\n" + "="*60)
    print("Training Deep Network")
    print("="*60)

    network = build_deep_network(
        input_size=64,
        hidden_sizes=[128, 64, 32],
        output_size=10,
        dropout_rate=0.3
    )

    print("\nNetwork architecture:")
    for name, layer in network.layers.items():
        print(f"  {name}: {layer.__class__.__name__}")

    optimizer = Adam(lr=0.001)
    trainer = train_with_early_stopping(
        network, optimizer, X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32, patience=10
    )

    # Final evaluation
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    train_acc = network.accuracy(X_train, y_train)
    val_acc = network.accuracy(X_val, y_val)
    test_acc = network.accuracy(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Show some predictions
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)

    predictions = network.predict(X_test[:10], train_mode=False)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:10], axis=1)

    print(f"{'Sample':<8} {'Predicted':<12} {'Actual':<12} {'Confidence':<12}")
    print("-" * 60)
    for i in range(10):
        confidence = predictions[i][pred_classes[i]]
        status = "✓" if pred_classes[i] == true_classes[i] else "✗"
        print(f"{i+1:<8} {pred_classes[i]:<12} {true_classes[i]:<12} "
              f"{confidence:.4f} {status}")

    # Option 2: Hyperparameter tuning (uncomment to run)
    # best_params = hyperparameter_search_example(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
