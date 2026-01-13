import numpy as np

class Trainer:
    """Trainer class for neural network"""
    def __init__(self, network, optimizer, verbose=True):
        self.network = network
        self.optimizer = optimizer
        self.verbose = verbose
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
    def train_step(self, x_batch, t_batch):
        """Single training step"""
        # Calculate gradients
        grads = self.network.gradient(x_batch, t_batch)
        
        # Update parameters
        params = self.network.get_params()
        self.optimizer.update(params, grads)
        self.network.set_params(params)
        
        # Calculate loss
        loss = self.network.loss(x_batch, t_batch)
        return loss
        
    def fit(self, x_train, t_train, x_test=None, t_test=None, 
            epochs=20, batch_size=100, evaluate_interval=1):
        """Train the network"""
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        
        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(train_size)
            x_train = x_train[idx]
            t_train = t_train[idx]
            
            epoch_loss = 0
            for i in range(iter_per_epoch):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                
                loss = self.train_step(x_batch, t_batch)
                epoch_loss += loss
                
            epoch_loss /= iter_per_epoch
            self.train_loss_list.append(epoch_loss)
            
            if (epoch + 1) % evaluate_interval == 0:
                train_acc = self.network.accuracy(x_train, t_train)
                self.train_acc_list.append(train_acc)
                
                if x_test is not None and t_test is not None:
                    test_acc = self.network.accuracy(x_test, t_test)
                    self.test_acc_list.append(test_acc)
                    
                    if self.verbose:
                        print(f"Epoch {epoch+1}/{epochs} - "
                              f"Loss: {epoch_loss:.4f} - "
                              f"Train Acc: {train_acc:.4f} - "
                              f"Test Acc: {test_acc:.4f}")
                else:
                    if self.verbose:
                        print(f"Epoch {epoch+1}/{epochs} - "
                              f"Loss: {epoch_loss:.4f} - "
                              f"Train Acc: {train_acc:.4f}")
                              
    def evaluate(self, x, t):
        """Evaluate the network"""
        return self.network.accuracy(x, t)
        
    def get_history(self):
        """Get training history"""
        return {
            'loss': self.train_loss_list,
            'train_accuracy': self.train_acc_list,
            'test_accuracy': self.test_acc_list
        }