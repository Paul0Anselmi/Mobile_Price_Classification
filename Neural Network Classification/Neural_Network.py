import numpy as np
from ativacoes import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax
from Loss import compute_loss

class NeuralNetwork:

    def __init__(self, layer_sizes, seed=127):
        if len(layer_sizes) < 2:
            raise ValueError()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)  # corrigido: era num_layer (sem o s)
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []  # corrigido: era self.bias, padronizado para self.biases

        for i in range(self.num_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            limit = np.sqrt(2.0 / n_in)
            w = self.rng.normal(loc=0.0, scale=limit, size=(n_in, n_out))
            b = np.zeros((1, n_out))

            self.weights.append(w)
            self.biases.append(b)

    def __repr__(self):
        arch = " -> ".join(str(n) for n in self.layer_sizes)
        return f"NeuralNet({arch})"

    def forward(self, X):
        activations = [X]
        pre_activations = []

        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            if i < self.num_layers - 2:
                a = relu(z)
            else:
                a = softmax(z)

            activations.append(a)

        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        n = y_true.shape[0]
        n_classes = activations[-1].shape[1]

        y_one_hot = np.zeros((n, n_classes))
        y_one_hot[np.arange(n), y_true] = 1.0

        grad_weights = [None] * (self.num_layers - 1)
        grad_biases  = [None] * (self.num_layers - 1)

        delta = activations[-1] - y_one_hot

        for i in reversed(range(self.num_layers - 1)):
            a_prev = activations[i]
            grad_weights[i] = a_prev.T @ delta / n
            grad_biases[i]  = delta.mean(axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i - 1])

        return grad_weights, grad_biases

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01,
              verbose=True, print_every=100):

        history = []

        for epoch in range(1, epochs + 1):
            activations, pre_activations = self.forward(X_train)

            # chama compute_loss importado do Loss.py
            loss = compute_loss(activations[-1], y_train)
            history.append(loss)

            grad_weights, grad_biases = self.backward(
                activations, pre_activations, y_train
            )

            for i in range(self.num_layers - 1):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i]  -= learning_rate * grad_biases[i]

            if verbose and epoch % print_every == 0:
                print(f"Epoca {epoch:>5}/{epochs} — Custo: {loss:.4f}")

        return history

    def predict(self, X):
        """
        Gera a predicao de classe para cada amostra em X.

        Faz o forward pass e aplica argmax sobre as probabilidades
        do softmax — retorna a classe com maior probabilidade.

        Parameters
        ----------
        X : np.ndarray, shape (n_amostras, n_features)
            Dados de entrada normalizados.

        Returns
        -------
        y_pred : np.ndarray, shape (n_amostras,)
            Classe prevista para cada amostra (0, 1, 2 ou 3).
        probabilities : np.ndarray, shape (n_amostras, n_classes)
            Probabilidades do softmax para cada classe.
        """
        activations, _ = self.forward(X)
        probabilities = activations[-1]

        # argmax retorna o indice da maior probabilidade em cada linha
        # Ex: [0.05, 0.10, 0.72, 0.13] -> 2 (preco alto)
        y_pred = np.argmax(probabilities, axis=1)

        return y_pred, probabilities