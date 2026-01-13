"""
Unit tests for YourLastNameMiniNN library
Testing individual components
"""

import numpy as np
from NadderMiniNN.classes.Dense import Dense
from NadderMiniNN.classes.Activations.Relu import Relu
from NadderMiniNN.classes.Activations.Sigmoid import Sigmoid
from NadderMiniNN.classes.Activations.BatchNormalization import BatchNormalization
from NadderMiniNN.classes.Activations.Dropout import Dropout
from NadderMiniNN.classes.Activations.SoftmaxWithLoss import SoftmaxWithLoss

from NadderMiniNN.neural_network import NeuralNetwork
from NadderMiniNN.classes.Optimizers.SGD import SGD
from NadderMiniNN.classes.Optimizers.Adam import Adam
from NadderMiniNN.trainer import Trainer


def test_dense_layer():
    """Test Dense layer forward and backward pass"""
    print("Testing Dense Layer...")

    # Create layer
    layer = Dense(3, 2)

    # Test forward
    x = np.array([[1, 2, 3], [4, 5, 6]])
    out = layer.forward(x)

    assert out.shape == (2, 2), "Dense forward output shape incorrect"

    # Test backward
    dout = np.ones_like(out)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, "Dense backward output shape incorrect"
    assert 'W' in layer.grads, "Weight gradients not computed"
    assert 'b' in layer.grads, "Bias gradients not computed"

    print("  ✓ Dense layer tests passed")


def test_relu_layer():
    """Test ReLU activation"""
    print("Testing ReLU Layer...")

    layer = Relu()
    x = np.array([[-1, 0, 1], [2, -3, 4]])

    # Forward
    out = layer.forward(x)
    expected = np.array([[0, 0, 1], [2, 0, 4]])

    assert np.allclose(out, expected), "ReLU forward incorrect"

    # Backward
    dout = np.ones_like(out)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, "ReLU backward shape incorrect"

    print("  ✓ ReLU layer tests passed")


def test_sigmoid_layer():
    """Test Sigmoid activation"""
    print("Testing Sigmoid Layer...")

    layer = Sigmoid()
    x = np.array([[0, 1, -1], [2, -2, 0]])

    # Forward
    out = layer.forward(x)

    assert np.all((out >= 0) & (out <= 1)), "Sigmoid output out of range"
    assert out.shape == x.shape, "Sigmoid forward shape incorrect"

    # Backward
    dout = np.ones_like(out)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, "Sigmoid backward shape incorrect"

    print("  ✓ Sigmoid layer tests passed")


def test_batch_normalization():
    """Test Batch Normalization"""
    print("Testing Batch Normalization...")

    layer = BatchNormalization(3)
    x = np.random.randn(10, 3)

    # Forward in training mode
    layer.train_mode = True
    out = layer.forward(x)

    # Check normalization
    mean = np.mean(out, axis=0)
    std = np.std(out, axis=0)

    assert np.allclose(mean, 0, atol=1e-6), "BatchNorm mean not zero"
    assert np.allclose(std, 1, atol=1e-1), "BatchNorm std not one"

    # Backward
    dout = np.ones_like(out)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, "BatchNorm backward shape incorrect"

    print("  ✓ Batch Normalization tests passed")


def test_dropout():
    """Test Dropout layer"""
    print("Testing Dropout Layer...")

    layer = Dropout(dropout_ratio=0.5)
    x = np.ones((100, 10))

    # Training mode
    layer.train_mode = True
    out = layer.forward(x)

    # Check that roughly half the values are zero
    zero_ratio = np.sum(out == 0) / out.size
    assert 0.3 < zero_ratio < 0.7, "Dropout ratio incorrect"

    # Test mode
    layer.train_mode = False
    out_test = layer.forward(x)

    # In test mode, no dropout but scaled
    assert np.allclose(out_test, x * 0.5), "Dropout test mode incorrect"

    print("  ✓ Dropout layer tests passed")


def test_softmax_loss():
    """Test Softmax with Cross Entropy loss"""
    print("Testing Softmax with Cross Entropy...")

    layer = SoftmaxWithLoss()

    # Test with one-hot labels
    x = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
    t = np.array([0, 1])  # Class indices

    loss = layer.forward(x, t)

    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"

    # Backward
    dx = layer.backward()

    assert dx.shape == x.shape, "Softmax backward shape incorrect"

    print("  ✓ Softmax with Cross Entropy tests passed")


def test_neural_network():
    """Test Neural Network construction"""
    print("Testing Neural Network...")

    network = NeuralNetwork()

    # Add layers
    network.add_layer('dense1', Dense(10, 5))
    network.add_layer('relu1', Relu())
    network.add_layer('dense2', Dense(5, 3))
    network.set_loss_layer(SoftmaxWithLoss())

    # Test predict
    x = np.random.randn(4, 10)
    out = network.predict(x)

    assert out.shape == (4, 3), "Network predict shape incorrect"

    # Test loss and gradient
    t = np.array([0, 1, 2, 0])
    loss = network.loss(x, t)

    assert isinstance(loss, (float, np.floating)), "Loss should be scalar"

    grads = network.gradient(x, t)

    assert len(grads) > 0, "Gradients not computed"

    print("  ✓ Neural Network tests passed")


def test_optimizer():
    """Test optimizers"""
    print("Testing Optimizers...")

    # Create simple parameters and gradients
    params = {'W': np.ones((3, 2)), 'b': np.zeros(2)}
    grads = {'W': np.ones((3, 2)) * 0.1, 'b': np.ones(2) * 0.1}

    # Test SGD
    optimizer = SGD(lr=0.1)
    params_copy = {k: v.copy() for k, v in params.items()}
    optimizer.update(params_copy, grads)

    assert not np.allclose(
        params_copy['W'], params['W']), "SGD didn't update parameters"

    # Test Adam
    optimizer = Adam(lr=0.001)
    params_copy = {k: v.copy() for k, v in params.items()}
    optimizer.update(params_copy, grads)

    assert not np.allclose(
        params_copy['W'], params['W']), "Adam didn't update parameters"

    print("  ✓ Optimizer tests passed")


def test_simple_training():
    """Test simple training loop"""
    print("Testing Training Loop...")

    # Create simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # XOR

    # Build network
    network = NeuralNetwork()
    network.add_layer('dense1', Dense(2, 4))
    network.add_layer('relu1', Relu())
    network.add_layer('dense2', Dense(4, 2))
    network.set_loss_layer(SoftmaxWithLoss())
    network.init_weights('he')

    # Train
    optimizer = Adam(lr=0.1)
    trainer = Trainer(network, optimizer, verbose=False)

    initial_loss = network.loss(X, y)
    trainer.fit(X, y, epochs=50, batch_size=4, evaluate_interval=50)
    final_loss = network.loss(X, y)

    assert final_loss < initial_loss, "Training didn't reduce loss"

    print("  ✓ Training loop tests passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running YourLastNameMiniNN Library Tests")
    print("="*60 + "\n")

    tests = [
        test_dense_layer,
        test_relu_layer,
        test_sigmoid_layer,
        test_batch_normalization,
        test_dropout,
        test_softmax_loss,
        test_neural_network,
        test_optimizer,
        test_simple_training
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {str(e)}")
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please review.")


if __name__ == "__main__":
    run_all_tests()
