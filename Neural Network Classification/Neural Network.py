import numpy as np
from ativacoes import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax

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
    

