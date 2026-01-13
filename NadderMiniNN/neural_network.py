import numpy as np
from collections import OrderedDict

class NeuralNetwork:
    """Neural Network class"""
    def __init__(self):
        self.layers = OrderedDict()
        self.loss_layer = None
        self.params = {}
        
    def add_layer(self, name, layer):
        """Add a layer to the network"""
        self.layers[name] = layer
        if hasattr(layer, 'params'):
            for key in layer.params.keys():
                param_key = f"{name}_{key}"
                self.params[param_key] = layer.params[key]
                
    def set_loss_layer(self, loss_layer):
        """Set the loss layer"""
        self.loss_layer = loss_layer
        
    def predict(self, x, train_mode=True):
        """Forward pass through the network"""
        for layer in self.layers.values():
            if hasattr(layer, 'train_mode'):
                layer.train_mode = train_mode
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        """Calculate loss"""
        y = self.predict(x, train_mode=True)
        return self.loss_layer.forward(y, t)
        
    def accuracy(self, x, t):
        """Calculate accuracy"""
        y = self.predict(x, train_mode=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        """Calculate gradients using backpropagation"""
        # Forward
        self.loss(x, t)
        
        # Backward
        dout = 1
        dout = self.loss_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # Collect gradients
        grads = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'grads'):
                for key in layer.grads.keys():
                    param_key = f"{layer_name}_{key}"
                    grads[param_key] = layer.grads[key]
                    
        return grads
        
    def init_weights(self, weight_init_std='he'):
        """Initialize weights for all layers"""
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params') and 'W' in layer.params:
                if weight_init_std == 'he':
                    std = np.sqrt(2.0 / layer.params['W'].shape[0])
                elif weight_init_std == 'xavier':
                    std = np.sqrt(1.0 / layer.params['W'].shape[0])
                else:
                    std = weight_init_std
                    
                layer.params['W'] = std * np.random.randn(*layer.params['W'].shape)
                
                # Update global params
                self.params[f"{layer_name}_W"] = layer.params['W']
                
    def get_params(self):
        """Get all parameters"""
        params = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params'):
                for key in layer.params.keys():
                    param_key = f"{layer_name}_{key}"
                    params[param_key] = layer.params[key]
        return params
        
    def set_params(self, params):
        """Set all parameters"""
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params'):
                for key in layer.params.keys():
                    param_key = f"{layer_name}_{key}"
                    if param_key in params:
                        layer.params[key] = params[param_key]