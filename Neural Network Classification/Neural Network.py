import numpy as np
from ativacoes import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax
from Loss import compute_loss

class NeuralNetwork:

    def __init__(self, layer_sizes, seed= 127):
        if len(layer_sizes)<2:
            raise ValueError()
        
        self.layer_sizes = layer_sizes
        self.num_layer = len(layer_sizes)
        self.rng = np.random.default_rng(seed)

        #pesos de cada camada
        self.weights = []
        
        #bias das camadas
        self.bias = []

        for i in range(self.num_layer - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]

            limit = np.sqrt(2.0/n_in)
            w = self.rng.normal(loc=0.0, scale = limit, size=(n_in,n_out))
            b = np.zeros((1,n_out))

            self.weights.append(w)
            self.bias.append(b)

    def __repr__(self):
        arch = " -> ".join(str(n) for n in self.layer_sizes)
        return f"NeuralNet({arch})"
    
    def backward(self, activations, pre_activations, y_true):

        n = y_true.shape[0]
        n_classes = activations[-1].shape[1]
 
        # One-hot encoding das classes reais
        y_one_hot = np.zeros((n, n_classes))
        y_one_hot[np.arange(n), y_true] = 1.0
 
        # Listas para guardar os gradientes (preenchidas de tras para frente)
        grad_weights = [None] * (self.num_layers - 1)
        grad_biases  = [None] * (self.num_layers - 1)
 
        # --- Camada de saida (softmax + loss) ---

        delta = activations[-1] - y_one_hot   # shape: (n, n_classes)
 
        # --- backpropagation ---
        for i in reversed(range(self.num_layers - 1)):
            a_prev = activations[i]   # ativacao da camada anterior
 
            # Gradiente dos pesos: a_prev.T @ delta / n
            grad_weights[i] = a_prev.T @ delta / n
 
            # Gradiente do bias: media dos deltas ao longo das amostras
            grad_biases[i] = delta.mean(axis=0, keepdims=True)
 
            # Propaga o delta para a camada anterior (exceto na camada de entrada)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i - 1])
 
        return grad_weights, grad_biases
