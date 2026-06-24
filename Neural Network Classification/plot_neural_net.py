import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_neural_network(layer_sizes, layer_names=None, title="Arquitetura da Rede Neural"):
    """
    Gera um diagrama visual da arquitetura da rede neural.

    Parameters
    ----------
    layer_sizes : list[int]
        Numero de neuronios em cada camada.
    layer_names : list[str], opcional
        Nomes das camadas para o rodape.
    title : str
        Titulo do grafico.
    """
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, n_layers + 1)
    ax.set_ylim(0, max_neurons + 1)
    ax.axis("off")

    # Cores para cada tipo de camada
    colors = {
        "input":  "#4C9BE8",   # azul  — entrada
        "hidden": "#F4A261",   # laranja — ocultas
        "output": "#2A9D8F",   # verde  — saida
    }

    neuron_positions = []   # guarda (x, y) de cada neuronio por camada

    for layer_idx, n_neurons in enumerate(layer_sizes):
        # Tipo da camada
        if layer_idx == 0:
            color = colors["input"]
        elif layer_idx == n_layers - 1:
            color = colors["output"]
        else:
            color = colors["hidden"]

        # Centraliza os neuronios verticalmente
        y_positions = np.linspace(
            (max_neurons - n_neurons) / 2 + 1,
            (max_neurons + n_neurons) / 2,
            n_neurons
        )

        layer_pos = []
        x = layer_idx + 1

        for y in y_positions:
            circle = plt.Circle((x, y), 0.3, color=color, zorder=4, linewidth=1.5,
                                 edgecolor="white")
            ax.add_patch(circle)
            layer_pos.append((x, y))

        neuron_positions.append(layer_pos)

    # Conexoes entre camadas
    for layer_idx in range(n_layers - 1):
        for (x1, y1) in neuron_positions[layer_idx]:
            for (x2, y2) in neuron_positions[layer_idx + 1]:
                ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.15,
                        linewidth=0.6, zorder=1)

    # Rotulos de quantidade de neuronios em cada camada
    for layer_idx, (n_neurons, positions) in enumerate(zip(layer_sizes, neuron_positions)):
        x = positions[0][0]
        y_top = positions[-1][1] + 0.65
        ax.text(x, y_top, f"{n_neurons}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7))

    # Rotulos das camadas no rodape
    if layer_names is None:
        layer_names = ["Entrada"] + \
                      [f"Oculta {i+1}" for i in range(n_layers - 2)] + \
                      ["Saida"]

    for layer_idx, (name, positions) in enumerate(zip(layer_names, neuron_positions)):
        x = positions[0][0]
        y_bot = positions[0][1] - 0.75
        ax.text(x, y_bot, name, ha="center", va="top",
                fontsize=10, color="#DDDDDD",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#444444", alpha=0.6))

    # Legenda
    legend_elements = [
        mpatches.Patch(color=colors["input"],  label="Camada de Entrada  (20 variaveis)"),
        mpatches.Patch(color=colors["hidden"], label="Camadas Ocultas  (ReLU)"),
        mpatches.Patch(color=colors["output"], label="Camada de Saida  (Softmax — 4 classes)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              facecolor="#2b2b2b", labelcolor="white", edgecolor="gray")

    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_title(title, fontsize=15, color="white", pad=20)

    plt.tight_layout()
    plt.savefig("arquitetura_rede.png", dpi=150, facecolor=fig.get_facecolor())
    print("Grafico salvo em: arquitetura_rede.png")
    plt.show()


if __name__ == "__main__":
    plot_neural_network(
        layer_sizes=[20, 16, 8, 4],
        title="Rede Neural — Mobile Price Classification"
    )