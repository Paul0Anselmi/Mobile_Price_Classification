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
    
    def forward(self, X):
            """
            Propaga as entradas X pela rede (forward pass).
    
            Parameters
            ----------
            X : np.ndarray, shape (n_amostras, n_features)
                Dados de entrada. Cada linha e uma amostra (um celular),
                cada coluna e uma das 20 variaveis do dataset.
    
            Returns
            -------
            activations : list[np.ndarray]
                Ativacoes de cada camada, incluindo a entrada (camada 0)
                e a saida (ultima camada). Guardadas para uso no backprop.
            pre_activations : list[np.ndarray]
                Valores lineares (z = X @ W + b) antes de aplicar a funcao
                de ativacao. Tambem necessarios no backprop.
            """
            activations = [X]       # a[0] = entrada bruta
            pre_activations = []    # z[i] = combinacao linear da camada i
    
            a = X
            for i in range(self.num_layers - 1):
                # Passo 1: combinacao linear z = a @ W + b
                z = a @ self.weights[i] + self.bias[i]
                pre_activations.append(z)
    
                # Passo 2: aplicar ativacao

                if i < self.num_layers - 2:
                    a = relu(z)
                else:
                    a = softmax(z)
    
                activations.append(a)
    
            return activations, pre_activations
    
    def compute_loss(self, y_pred, y_true):
        """
        Calcula a funcao de custo Cross-Entropy para classificacao
        multi-classe.
 
        Parameters
        ----------
        y_pred : np.ndarray, shape (n_amostras, n_classes)
            Probabilidades geradas pelo softmax (saida do forward pass).
        y_true : np.ndarray, shape (n_amostras,)
            Classes reais de cada amostra (valores inteiros: 0, 1, 2 ou 3).
 
        Returns
        -------
        loss : float
            Valor medio do custo para todas as amostras. Quanto menor,
            melhor a rede esta classificando.
        """
        n = y_true.shape[0]
 
        # Converte y_true para one-hot encoding.
        # Ex: classe 2 vira [0, 0, 1, 0] para uma rede com 4 classes.
        n_classes = y_pred.shape[1]
        y_one_hot = np.zeros((n, n_classes))
        y_one_hot[np.arange(n), y_true] = 1.0
 

        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
 
        # funcao de perda
        loss = -np.sum(y_one_hot * np.log(y_pred_clipped)) / n
 
        return loss