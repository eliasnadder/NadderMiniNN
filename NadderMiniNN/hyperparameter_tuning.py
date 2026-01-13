from itertools import product

class HyperparameterTuning:
    """Hyperparameter tuning class"""
    def __init__(self, network_builder, trainer_builder):
        """
        Args:
            network_builder: Function that builds and returns a network
            trainer_builder: Function that builds and returns a trainer
        """
        self.network_builder = network_builder
        self.trainer_builder = trainer_builder
        self.results = []
        
    def grid_search(self, param_grid, x_train, t_train, x_val, t_val, 
                   epochs=10, batch_size=100, verbose=True):
        """
        Grid search for hyperparameters
        
        Args:
            param_grid: Dictionary of parameter names and lists of values to try
                       Example: {'lr': [0.001, 0.01], 'batch_size': [32, 64]}
            x_train, t_train: Training data
            x_val, t_val: Validation data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            verbose: Whether to print progress
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            if verbose:
                print(f"\nTesting parameters: {params}")
                
            # Build network and trainer with these parameters
            network = self.network_builder(**params)
            trainer = self.trainer_builder(network, **params)
            
            # Train
            trainer.fit(x_train, t_train, x_val, t_val, 
                       epochs=epochs, batch_size=batch_size, 
                       evaluate_interval=epochs, verbose=False)
            
            # Evaluate
            val_acc = trainer.evaluate(x_val, t_val)
            
            result = {
                'params': params,
                'val_accuracy': val_acc,
                'history': trainer.get_history()
            }
            self.results.append(result)
            
            if verbose:
                print(f"Validation Accuracy: {val_acc:.4f}")
                
        # Sort by validation accuracy
        self.results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        if verbose:
            print("\n" + "="*50)
            print("Best hyperparameters:")
            print(f"Parameters: {self.results[0]['params']}")
            print(f"Validation Accuracy: {self.results[0]['val_accuracy']:.4f}")
            
        return self.results[0]
        
    def random_search(self, param_distributions, n_iter, x_train, t_train, 
                     x_val, t_val, epochs=10, batch_size=100, verbose=True):
        """
        Random search for hyperparameters
        
        Args:
            param_distributions: Dictionary of parameter names and sampling functions
                                Example: {'lr': lambda: 10**np.random.uniform(-4, -1)}
            n_iter: Number of random combinations to try
            x_train, t_train: Training data
            x_val, t_val: Validation data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            verbose: Whether to print progress
        """
        for i in range(n_iter):
            # Sample parameters
            params = {key: func() for key, func in param_distributions.items()}
            
            if verbose:
                print(f"\n[{i+1}/{n_iter}] Testing parameters: {params}")
                
            # Build network and trainer with these parameters
            network = self.network_builder(**params)
            trainer = self.trainer_builder(network, **params)
            
            # Train
            trainer.fit(x_train, t_train, x_val, t_val, 
                       epochs=epochs, batch_size=batch_size, 
                       evaluate_interval=epochs, verbose=False)
            
            # Evaluate
            val_acc = trainer.evaluate(x_val, t_val)
            
            result = {
                'params': params,
                'val_accuracy': val_acc,
                'history': trainer.get_history()
            }
            self.results.append(result)
            
            if verbose:
                print(f"Validation Accuracy: {val_acc:.4f}")
                
        # Sort by validation accuracy
        self.results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        if verbose:
            print("\n" + "="*50)
            print("Best hyperparameters:")
            print(f"Parameters: {self.results[0]['params']}")
            print(f"Validation Accuracy: {self.results[0]['val_accuracy']:.4f}")
            
        return self.results[0]
        
    def get_best_params(self):
        """Get the best parameters found"""
        if not self.results:
            return None
        return self.results[0]['params']
        
    def get_all_results(self):
        """Get all results"""
        return self.results