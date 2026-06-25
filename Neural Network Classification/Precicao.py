import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Neural_Network import NeuralNetwork

# ------------------------------------------------------------------
# 1) Carregar os dados de treino e teste
# ------------------------------------------------------------------
base_dir  = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "..", "Database", "train.csv")
test_path  = os.path.join(base_dir, "..", "Database", "test.csv")

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print(f"Treino: {df_train.shape[0]} amostras")
print(f"Teste:  {df_test.shape[0]} amostras\n")

# ------------------------------------------------------------------
# 2) Preparar os dados de treino
# ------------------------------------------------------------------
X_train = df_train.drop(columns=["price_range"]).values.astype(float)
y_train = df_train["price_range"].values

# Normalizacao min-max — usando os parametros do TREINO
# (importante: nunca usar os parametros do teste para normalizar)
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)

# ------------------------------------------------------------------
# 3) Preparar os dados de teste com os mesmos parametros do treino
# ------------------------------------------------------------------
# O test.csv tem uma coluna "id" extra que precisa ser removida
X_test = df_test.drop(columns=["id"]).values.astype(float)
X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)

# ------------------------------------------------------------------
# 4) Treinar a rede com TODOS os dados de treino (sem separar validacao)
#    Agora que os hiperparametros estao definidos, usamos tudo para treinar
# ------------------------------------------------------------------
nn = NeuralNetwork(layer_sizes=[20, 16, 8, 4], seed=127)
print(f"{nn}")
print("Treinando com todos os dados de treino...\n")

history = nn.train(
    X_train_norm, y_train,
    epochs=10000,
    learning_rate=0.01,
    verbose=True,
    print_every=1000
)

# ------------------------------------------------------------------
# 5) Gerar predicoes no teste
# ------------------------------------------------------------------
y_pred_test, probabilities = nn.predict(X_test_norm)

class_names = {
    0: "Preco Baixo",
    1: "Preco Medio",
    2: "Preco Alto",
    3: "Preco Muito Alto"
}

print(f"\n{'='*50}")
print(f"  Predicoes geradas para {len(y_pred_test)} celulares")
print(f"{'='*50}")

# Distribuicao das predicoes
print("\nDistribuicao das classes previstas:")
for classe, nome in class_names.items():
    count = np.sum(y_pred_test == classe)
    pct   = count / len(y_pred_test) * 100
    barra = "█" * int(pct / 2)
    print(f"  {nome:<18} ({classe}): {count:>4} amostras ({pct:>5.1f}%) {barra}")

# ------------------------------------------------------------------
# 6) Salvar as predicoes em CSV
# ------------------------------------------------------------------
df_result = df_test.copy()
df_result["price_range_predicted"] = y_pred_test
df_result["classe_nome"] = [class_names[p] for p in y_pred_test]

# Adiciona probabilidades por classe
for i, nome in class_names.items():
    df_result[f"prob_classe_{i}"] = np.round(probabilities[:, i], 4)

output_csv = os.path.join(base_dir, "predicoes_test.csv")
df_result.to_csv(output_csv, index=False)
print(f"\nPredicoes salvas em: predicoes_test.csv")

# ------------------------------------------------------------------
# 7) Graficos
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#1e1e1e")

# --- Curva de aprendizado ---
axes[0].set_facecolor("#1e1e1e")
axes[0].plot(history, color="royalblue", linewidth=2)
axes[0].set_title("Curva de Aprendizado (Treino Completo)",
                   color="white", fontsize=12)
axes[0].set_xlabel("Epoca", color="white")
axes[0].set_ylabel("Custo (Cross-Entropy)", color="white")
axes[0].tick_params(colors="white")
axes[0].grid(True, linestyle="--", alpha=0.3)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#444444")

# --- Distribuicao das predicoes ---
axes[1].set_facecolor("#1e1e1e")
counts = [np.sum(y_pred_test == i) for i in range(4)]
cores  = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]
bars   = axes[1].bar([f"Classe {i}\n{class_names[i]}" for i in range(4)],
                      counts, color=cores, edgecolor="white", linewidth=0.5)

axes[1].set_title("Distribuicao das Predicoes no Teste",
                   color="white", fontsize=12)
axes[1].set_ylabel("Quantidade de Celulares", color="white")
axes[1].tick_params(colors="white")
axes[1].set_facecolor("#1e1e1e")
for spine in axes[1].spines.values():
    spine.set_edgecolor("#444444")

for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 3,
                 str(count), ha="center", va="bottom",
                 color="white", fontsize=11, fontweight="bold")

plt.tight_layout()
output_png = os.path.join(base_dir, "predicoes_test.png")
plt.savefig(output_png, dpi=150, facecolor=fig.get_facecolor())
print(f"Grafico salvo em: predicoes_test.png")
plt.show()