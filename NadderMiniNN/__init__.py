"""
NadderMiniNN - A mini neural network library

A simple neural network library built from scratch using only NumPy.
Designed for educational purposes to understand how neural networks work internally.
"""

__version__ = '1.0.0'
__author__ = 'Nadder'

# Import main components
from .classes import (Layer, Dense)

from .classes.Activations import (
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    Dropout,
    BatchNormalization,
    MeanSquaredError,
    SoftmaxWithLoss
)

from .classes.Optimizers import (
    Optimizer,
    SGD,
    Momentum,
    AdaGrad,
    Adam,
    RMSprop
)

from .neural_network import NeuralNetwork
from .trainer import Trainer
from .hyperparameter_tuning import HyperparameterTuning

__all__ = [
    # Layers
    'Layer',
    'Dense',
    'Linear',
    'Relu',
    'Sigmoid',
    'Tanh',
    'Dropout',
    'BatchNormalization',
    'MeanSquaredError',
    'SoftmaxWithLoss',

    # Optimizers
    'Optimizer',
    'SGD',
    'Momentum',
    'AdaGrad',
    'Adam',
    'RMSprop',

    # Core components
    'NeuralNetwork',
    'Trainer',
    'HyperparameterTuning'
]
