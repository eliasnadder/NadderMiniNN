"""
NadderMiniNN - A mini neural network library

A simple neural network library built from scratch using only NumPy.
Designed for educational purposes to understand how neural networks work internally.
"""

__version__ = '1.0.0'
__author__ = 'Elias Nadder'

# Import Layer base class
from .classes.Layer import Layer
from .classes.Dense import Dense

# Import Activations
from .classes.Activations.Linear import Linear
from .classes.Activations.Relu import Relu
from .classes.Activations.Sigmoid import Sigmoid
from .classes.Activations.Tanh import Tanh
from .classes.Activations.Dropout import Dropout
from .classes.Activations.BatchNormalization import BatchNormalization
from .classes.Activations.MeanSquaredError import MeanSquaredError
from .classes.Activations.SoftmaxWithLoss import SoftmaxWithLoss

# Import Optimizers
from .classes.Optimizers.Optimizer import Optimizer
from .classes.Optimizers.SGD import SGD
from .classes.Optimizers.Momentum import Momentum
from .classes.Optimizers.AdaGrad import AdaGrad
from .classes.Optimizers.Adam import Adam
from .classes.Optimizers.RMSprop import RMSprop

# Import Core components
from .neural_network import NeuralNetwork
from .trainer import Trainer
from .hyperparameter_tuning import HyperparameterTuning

__all__ = [
    # Base classes
    'Layer',
    'Dense',
    
    # Activations
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