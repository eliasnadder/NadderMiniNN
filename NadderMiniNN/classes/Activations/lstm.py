"""
LSTM Layer implementation for NadderMiniNN
Based on research papers and manual implementation guidelines
"""
import numpy as np
from NadderMiniNN.classes.layer1 import Layer1


class LSTM(Layer1):
    """
    Long Short-Term Memory (LSTM) layer for time series prediction
    
    Architecture:
    - Forget Gate: Decides what information to discard from cell state
    - Input Gate: Decides what new information to store in cell state
    - Cell State Update: Updates the cell state
    - Output Gate: Decides what to output based on cell state
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden units
    return_sequences : bool
        If True, returns full sequence. If False, returns only last output
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        # Initialize weights using Xavier initialization for sigmoid/tanh
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        
        # Forget gate parameters
        self.params['Wf'] = scale * np.random.randn(input_size, hidden_size)
        self.params['Uf'] = scale * np.random.randn(hidden_size, hidden_size)
        self.params['bf'] = np.zeros(hidden_size)
        
        # Input gate parameters
        self.params['Wi'] = scale * np.random.randn(input_size, hidden_size)
        self.params['Ui'] = scale * np.random.randn(hidden_size, hidden_size)
        self.params['bi'] = np.zeros(hidden_size)
        
        # Cell state candidate parameters
        self.params['Wc'] = scale * np.random.randn(input_size, hidden_size)
        self.params['Uc'] = scale * np.random.randn(hidden_size, hidden_size)
        self.params['bc'] = np.zeros(hidden_size)
        
        # Output gate parameters
        self.params['Wo'] = scale * np.random.randn(input_size, hidden_size)
        self.params['Uo'] = scale * np.random.randn(hidden_size, hidden_size)
        self.params['bo'] = np.zeros(hidden_size)
        
        # Cache for backward pass
        self.cache = {}
        
    def _sigmoid(self, x):
        """Sigmoid activation with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def forward(self, x):
        """
        Forward pass through LSTM
        
        Parameters:
        -----------
        x : ndarray, shape (batch_size, sequence_length, input_size)
            Input sequences
            
        Returns:
        --------
        output : ndarray
            If return_sequences=True: (batch_size, sequence_length, hidden_size)
            If return_sequences=False: (batch_size, hidden_size)
        """
        batch_size, seq_length, _ = x.shape
        
        # Initialize hidden state and cell state
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        # Store all states for backward pass
        self.cache['x'] = x
        self.cache['h_states'] = []
        self.cache['c_states'] = []
        self.cache['f_gates'] = []
        self.cache['i_gates'] = []
        self.cache['c_candidates'] = []
        self.cache['o_gates'] = []
        
        outputs = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            # Forget gate
            f_t = self._sigmoid(
                np.dot(x_t, self.params['Wf']) + 
                np.dot(h, self.params['Uf']) + 
                self.params['bf']
            )
            
            # Input gate
            i_t = self._sigmoid(
                np.dot(x_t, self.params['Wi']) + 
                np.dot(h, self.params['Ui']) + 
                self.params['bi']
            )
            
            # Cell state candidate
            c_tilde = self._tanh(
                np.dot(x_t, self.params['Wc']) + 
                np.dot(h, self.params['Uc']) + 
                self.params['bc']
            )
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Output gate
            o_t = self._sigmoid(
                np.dot(x_t, self.params['Wo']) + 
                np.dot(h, self.params['Uo']) + 
                self.params['bo']
            )
            
            # Update hidden state
            h = o_t * self._tanh(c)
            
            # Store for backward pass
            self.cache['h_states'].append(h.copy())
            self.cache['c_states'].append(c.copy())
            self.cache['f_gates'].append(f_t)
            self.cache['i_gates'].append(i_t)
            self.cache['c_candidates'].append(c_tilde)
            self.cache['o_gates'].append(o_t)
            
            outputs.append(h)
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]
    
    def backward(self, dout):
        """
        Backward pass through LSTM
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from next layer
            
        Returns:
        --------
        dx : ndarray, shape (batch_size, sequence_length, input_size)
            Gradient with respect to input
        """
        x = self.cache['x']
        batch_size, seq_length, _ = x.shape
        
        # Initialize gradients
        for key in self.params.keys():
            self.grads[key] = np.zeros_like(self.params[key])
        
        dx = np.zeros_like(x)
        
        # Initialize gradients for hidden and cell states
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # If not return_sequences, expand dout to sequence
        if not self.return_sequences:
            dout_seq = np.zeros((batch_size, seq_length, self.hidden_size))
            dout_seq[:, -1, :] = dout
            dout = dout_seq
        
        # Backward through time
        for t in reversed(range(seq_length)):
            x_t = x[:, t, :]
            
            # Get cached values
            h_prev = self.cache['h_states'][t-1] if t > 0 else np.zeros((batch_size, self.hidden_size))
            c_prev = self.cache['c_states'][t-1] if t > 0 else np.zeros((batch_size, self.hidden_size))
            
            f_t = self.cache['f_gates'][t]
            i_t = self.cache['i_gates'][t]
            c_tilde = self.cache['c_candidates'][t]
            o_t = self.cache['o_gates'][t]
            c_t = self.cache['c_states'][t]
            
            # Gradient from current timestep and next timestep
            dh = dout[:, t, :] + dh_next
            
            # Output gate gradient
            do = dh * self._tanh(c_t)
            do_input = do * o_t * (1 - o_t)  # sigmoid derivative
            
            # Cell state gradient
            dc = dh * o_t * (1 - self._tanh(c_t)**2) + dc_next
            
            # Cell candidate gradient
            dc_tilde = dc * i_t
            dc_tilde_input = dc_tilde * (1 - c_tilde**2)  # tanh derivative
            
            # Input gate gradient
            di = dc * c_tilde
            di_input = di * i_t * (1 - i_t)  # sigmoid derivative
            
            # Forget gate gradient
            df = dc * c_prev
            df_input = df * f_t * (1 - f_t)  # sigmoid derivative
            
            # Gradients for weights and biases
            self.grads['Wf'] += np.dot(x_t.T, df_input)
            self.grads['Uf'] += np.dot(h_prev.T, df_input)
            self.grads['bf'] += np.sum(df_input, axis=0)
            
            self.grads['Wi'] += np.dot(x_t.T, di_input)
            self.grads['Ui'] += np.dot(h_prev.T, di_input)
            self.grads['bi'] += np.sum(di_input, axis=0)
            
            self.grads['Wc'] += np.dot(x_t.T, dc_tilde_input)
            self.grads['Uc'] += np.dot(h_prev.T, dc_tilde_input)
            self.grads['bc'] += np.sum(dc_tilde_input, axis=0)
            
            self.grads['Wo'] += np.dot(x_t.T, do_input)
            self.grads['Uo'] += np.dot(h_prev.T, do_input)
            self.grads['bo'] += np.sum(do_input, axis=0)
            
            # Gradient with respect to input
            dx[:, t, :] = (
                np.dot(df_input, self.params['Wf'].T) +
                np.dot(di_input, self.params['Wi'].T) +
                np.dot(dc_tilde_input, self.params['Wc'].T) +
                np.dot(do_input, self.params['Wo'].T)
            )
            
            # Gradient with respect to previous hidden state
            dh_next = (
                np.dot(df_input, self.params['Uf'].T) +
                np.dot(di_input, self.params['Ui'].T) +
                np.dot(dc_tilde_input, self.params['Uc'].T) +
                np.dot(do_input, self.params['Uo'].T)
            )
            
            # Gradient with respect to previous cell state
            dc_next = dc * f_t
        
        return dx