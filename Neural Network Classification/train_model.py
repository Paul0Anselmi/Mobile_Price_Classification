import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Neural_Network import NeuralNetwork

# ------------------------------------------------------------------
# 1) Carregar os dados
# ------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "Database", "train.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["price_range"]).values.astype(float)
y = df["price_range"].values

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} variaveis")

# ------------------------------------------------------------------
# 2) Normalizacao min-max
# ------------------------------------------------------------------
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# ------------------------------------------------------------------
# 3) Divisao treino / validacao: 70% / 30%
# ------------------------------------------------------------------
split = int(0.7 * len(X_norm))
X_tr, X_val = X_norm[:split], X_norm[split:]
y_tr, y_val = y[:split], y[split:]

print(f"Treino:    {X_tr.shape[0]} amostras")
print(f"Validacao: {X_val.shape[0]} amostras\n")

# ------------------------------------------------------------------
# 4) Criar e treinar a rede
# ------------------------------------------------------------------
nn = NeuralNetwork(layer_sizes=[20, 16, 8, 4], seed=127)
print(nn)

history = nn.train(
    X_tr, y_tr,
    epochs=10000,
    learning_rate=0.01,
    verbose=True,
    print_every=100
)

# ------------------------------------------------------------------
# 5) Avaliacao — Acuracia
# ------------------------------------------------------------------
class_names = ["Preco Baixo (0)", "Preco Medio (1)",
               "Preco Alto (2)",  "Preco Muito Alto (3)"]

y_pred_tr,  _ = nn.predict(X_tr)
y_pred_val, _ = nn.predict(X_val)

acc_tr  = np.mean(y_pred_tr  == y_tr)
acc_val = np.mean(y_pred_val == y_val)

print(f"\n{'='*45}")
print(f"  Acuracia no Treino:    {acc_tr  * 100:.2f}%")
print(f"  Acuracia na Validacao: {acc_val * 100:.2f}%")
print(f"  Meta do artigo:        96.31%")
print(f"{'='*45}")

# ------------------------------------------------------------------
# 6) Matriz de Confusao
# ------------------------------------------------------------------
n_classes = 4
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
for real, previsto in zip(y_val, y_pred_val):
    conf_matrix[real][previsto] += 1

print("\nMatriz de Confusao (Validacao):")
print(f"{'':20s} " + "  ".join(f"Prev {i}" for i in range(n_classes)))
for i, row in enumerate(conf_matrix):
    print(f"Real {i} ({class_names[i][:12]}): " +
          "  ".join(f"{v:6d}" for v in row))

# ------------------------------------------------------------------
# 7) Precision, Recall e F1 por classe
# ------------------------------------------------------------------
print(f"\n{'Classe':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Suporte':>10}")
print("-" * 55)

for k in range(n_classes):
    TP = conf_matrix[k][k]
    FP = conf_matrix[:, k].sum() - TP
    FN = conf_matrix[k, :].sum() - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    support   = conf_matrix[k, :].sum()

    print(f"{class_names[k]:<20} {precision:>10.4f} {recall:>10.4f} "
          f"{f1:>10.4f} {support:>10d}")

# ------------------------------------------------------------------
# 8) Graficos
# ------------------------------------------------------------------

# --- Curva de aprendizado ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#1e1e1e")

axes[0].set_facecolor("#1e1e1e")
axes[0].plot(history, color="royalblue", linewidth=2)
axes[0].set_title("Curva de Aprendizado", color="white", fontsize=13)
axes[0].set_xlabel("Epoca", color="white")
axes[0].set_ylabel("Custo (Cross-Entropy)", color="white")
axes[0].tick_params(colors="white")
axes[0].grid(True, linestyle="--", alpha=0.3)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#444444")

# --- Matriz de confusao ---
axes[1].set_facecolor("#1e1e1e")
im = axes[1].imshow(conf_matrix, cmap="Blues")
axes[1].set_title("Matriz de Confusao (Validacao)", color="white", fontsize=13)
axes[1].set_xlabel("Classe Prevista", color="white")
axes[1].set_ylabel("Classe Real", color="white")
axes[1].set_xticks(range(n_classes))
axes[1].set_yticks(range(n_classes))
axes[1].set_xticklabels([f"Classe {i}" for i in range(n_classes)],
                         color="white", rotation=15)
axes[1].set_yticklabels([f"Classe {i}" for i in range(n_classes)], color="white")

for i in range(n_classes):
    for j in range(n_classes):
        axes[1].text(j, i, str(conf_matrix[i][j]),
                     ha="center", va="center",
                     color="white" if conf_matrix[i][j] < conf_matrix.max() * 0.6
                     else "black", fontsize=12, fontweight="bold")

plt.colorbar(im, ax=axes[1])
plt.tight_layout()

output_path = os.path.join(base_dir, "resultado_treinamento.png")
plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
print(f"\nGrafico salvo em: {output_path}")
plt.show()