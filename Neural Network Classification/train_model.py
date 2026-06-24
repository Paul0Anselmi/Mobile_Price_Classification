import os
import numpy as np
import pandas as pd
from Neural_Network import NeuralNetwork
import matplotlib.pyplot as plt

# 1) Carregar os dados
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join("..", "Database", "train.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["price_range"]).values.astype(float)
y = df["price_range"].values

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} variaveis")

# 2) Normalizacao min-max
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# 3) Divisao treino / validacao: 70% / 30%
split = int(0.7 * len(X_norm))
X_tr, X_val = X_norm[:split], X_norm[split:]
y_tr, y_val = y[:split], y[split:]

print(f"Treino:    {X_tr.shape[0]} amostras")
print(f"Validacao: {X_val.shape[0]} amostras\n")

# 4) Criar e treinar a rede
nn = NeuralNetwork(layer_sizes=[20, 16, 8, 4], seed=127)
print(nn)

history = nn.train(
    X_tr, y_tr,
    epochs=1000,
    learning_rate=0.01,
    verbose=True,
    print_every=100
)

# ------------------------------------------------------------------
# 5) Grafico da curva de aprendizado
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history, color="royalblue", linewidth=2)
plt.title("Curva de Aprendizado — Cross-Entropy por Epoca", fontsize=14)
plt.xlabel("Epoca", fontsize=12)
plt.ylabel("Custo (Cross-Entropy)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
 
output_path = os.path.join(base_dir, "curva_aprendizado.png")
plt.savefig(output_path, dpi=150)
print(f"\nGrafico salvo em: {output_path}")
plt.show()