import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Neural_Network import NeuralNetwork

# ------------------------------------------------------------------
# 1) Carregar os dados
# ------------------------------------------------------------------
base_dir  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "Database", "train.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["price_range"]).values.astype(float)
y = df["price_range"].values

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} variaveis")

# ------------------------------------------------------------------
# 2) Normalizacao min-max
# ------------------------------------------------------------------
X_min  = X.min(axis=0)
X_max  = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# ------------------------------------------------------------------
# 3) Divisao treino / validacao: 70% / 30%
# ------------------------------------------------------------------
split       = int(0.7 * len(X_norm))
X_tr, X_val = X_norm[:split], X_norm[split:]
y_tr, y_val = y[:split], y[split:]

print(f"Treino:    {X_tr.shape[0]} amostras")
print(f"Validacao: {X_val.shape[0]} amostras\n")

# ------------------------------------------------------------------
# 4) Criar e treinar a rede
# ------------------------------------------------------------------
nn = NeuralNetwork(layer_sizes=[20, 16, 8, 4], seed=127)
print(nn)

history_tr = nn.train(
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
n_classes   = 4

y_pred_tr,  _         = nn.predict(X_tr)
y_pred_val, probs_val = nn.predict(X_val)

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

precision_list = []
recall_list    = []
f1_list        = []

for k in range(n_classes):
    TP = conf_matrix[k][k]
    FP = conf_matrix[:, k].sum() - TP
    FN = conf_matrix[k, :].sum() - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    support   = conf_matrix[k, :].sum()

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"{class_names[k]:<20} {precision:>10.4f} {recall:>10.4f} "
          f"{f1:>10.4f} {support:>10d}")

# ------------------------------------------------------------------
# 8) Graficos — painel completo (6 graficos)
# ------------------------------------------------------------------
DARK       = "#1e1e1e"
GRID_COLOR = "#444444"
colors     = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ---- Grafico A: Curva de aprendizado ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(DARK)
ax1.plot(history_tr, color="#4C9BE8", linewidth=2, label="Treino")
ax1.set_title("Curva de Aprendizado", color="white", fontsize=12)
ax1.set_xlabel("Epoca", color="white")
ax1.set_ylabel("Custo (Cross-Entropy)", color="white")
ax1.tick_params(colors="white")
ax1.legend(facecolor="#2b2b2b", labelcolor="white", fontsize=9)
ax1.grid(linestyle="--", alpha=0.2)
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# ---- Grafico B: Matriz de confusao ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(DARK)
im = ax2.imshow(conf_matrix, cmap="Blues")
ax2.set_title("Matriz de Confusao (Validacao)", color="white", fontsize=12)
ax2.set_xlabel("Classe Prevista", color="white")
ax2.set_ylabel("Classe Real", color="white")
ax2.set_xticks(range(n_classes))
ax2.set_yticks(range(n_classes))
ax2.set_xticklabels([f"Classe {i}" for i in range(n_classes)],
                     color="white", rotation=15)
ax2.set_yticklabels([f"Classe {i}" for i in range(n_classes)], color="white")

threshold = conf_matrix.max() / 2
for i in range(n_classes):
    for j in range(n_classes):
        # texto escuro em celulas claras (valores baixos), claro em celulas escuras
        text_color = "white" if conf_matrix[i][j] < threshold else "black"
        ax2.text(j, i, str(conf_matrix[i][j]),
                 ha="center", va="center",
                 color=text_color, fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax2)
ax2.tick_params(colors="white")

# ---- Grafico 1: Metricas por classe ----
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(DARK)
x     = np.arange(n_classes)
width = 0.25
bp = ax3.bar(x - width, precision_list, width, label="Precision",
              color="#4C9BE8", edgecolor="white", linewidth=0.5)
br = ax3.bar(x,          recall_list,   width, label="Recall",
              color="#F4A261", edgecolor="white", linewidth=0.5)
bf = ax3.bar(x + width,  f1_list,       width, label="F1-Score",
              color="#2A9D8F", edgecolor="white", linewidth=0.5)
for bars in [bp, br, bf]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", color="white", fontsize=7.5)
ax3.set_title("Grafico 1 — Metricas por Classe", color="white", fontsize=12)
ax3.set_ylabel("Score", color="white")
ax3.set_xticks(x)
ax3.set_xticklabels([f"Cl.{i} {class_names[i][:10]}" for i in range(n_classes)],
                     color="white", rotation=12, fontsize=8)
ax3.tick_params(colors="white")
ax3.set_ylim(0, 1.12)
ax3.legend(facecolor="#2b2b2b", labelcolor="white", fontsize=9)
ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
ax3.grid(axis="y", linestyle="--", alpha=0.2)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# ---- Grafico 2: Distribuicao das probabilidades ----
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(DARK)
for k in range(n_classes):
    mask  = y_val == k
    probs = probs_val[mask, k]
    ax4.hist(probs, bins=20, alpha=0.7, color=colors[k],
             label=class_names[k][:14], edgecolor="white", linewidth=0.3)
ax4.set_title("Grafico 2 — Probabilidade da Classe Correta", color="white", fontsize=12)
ax4.set_xlabel("Probabilidade atribuida a classe correta", color="white")
ax4.set_ylabel("Frequencia", color="white")
ax4.tick_params(colors="white")
ax4.legend(facecolor="#2b2b2b", labelcolor="white", fontsize=8)
ax4.grid(linestyle="--", alpha=0.2)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# ---- Grafico 5: Mapa de erros ----
ax5 = fig.add_subplot(gs[2, :])
error_matrix = conf_matrix.copy().astype(float)
np.fill_diagonal(error_matrix, 0)
im2 = ax5.imshow(error_matrix, cmap="Reds", aspect="auto")
ax5.set_title("Grafico 5 — Mapa de Erros (diagonal zerada — so erros)",
               color="white", fontsize=12)
ax5.set_xlabel("Classe Prevista (errada)", color="white")
ax5.set_ylabel("Classe Real", color="white")
ax5.set_xticks(range(n_classes))
ax5.set_yticks(range(n_classes))
ax5.set_xticklabels([f"Cl.{i} — {class_names[i]}" for i in range(n_classes)],
                     color="white", fontsize=9)
ax5.set_yticklabels([f"Cl.{i} — {class_names[i]}" for i in range(n_classes)],
                     color="white", fontsize=9)
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            ax5.text(j, i, str(int(error_matrix[i][j])),
                     ha="center", va="center",
                     color="white", fontsize=13, fontweight="bold")
plt.colorbar(im2, ax=ax5, label="Numero de erros")
ax5.tick_params(colors="white")

fig.suptitle("Avaliacao Completa — Mobile Price Classification",
              color="white", fontsize=14, fontweight="bold", y=0.99)

output_path = os.path.join(base_dir, "resultado_treinamento.png")
plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"\nGrafico salvo em: {output_path}")
plt.show()