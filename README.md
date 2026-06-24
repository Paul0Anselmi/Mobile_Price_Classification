# Mobile Price Classification

Implementação de uma Rede Neural Multilayer Perceptron (MLP) do zero com NumPy para classificação de faixas de preço de celulares.

Baseado no artigo:
> Nasser, I. M., Al-Shawwa, M., & Abu-Naser, S. S. (2019). *Developing Artificial Neural Network for Predicting Mobile Phone Price Range*. International Journal of Academic Information Systems Research (IJAISR), Vol. 3, Issue 2, pp. 1-6.

---

## Estrutura do Projeto

O código está organizado em módulos separados por responsabilidade:

```
Mobile_Price_Classification/
│
├── Database/
│   ├── train.csv          # Base de treino (2000 amostras, 20 variáveis)
│   └── test.csv           # Base de teste (1000 amostras)
│
└── Neural Network Classification/
    ├── ativacoes.py       # Funções de ativação: ReLU, Sigmoid, Softmax e derivadas
    ├── Loss.py            # Função de custo: Cross-Entropy
    ├── Neural_Network.py  # Classe principal da rede neural (MLP)
    └── train_model.py     # Script de treinamento: carrega os dados e treina a rede
```

---

## Descrição dos Módulos

**`ativacoes.py`**
Contém as funções de ativação utilizadas pela rede:
- `relu` e `relu_derivative` — usadas nas camadas ocultas
- `sigmoid` e `sigmoid_derivative` — disponível como alternativa
- `softmax` — usada na camada de saída para classificação multi-classe

**`Loss.py`**
Contém a função de custo `compute_loss`, que calcula a Cross-Entropy entre as probabilidades previstas pela rede e as classes reais.

**`Neural_Network.py`**
Classe `NeuralNetwork` com os seguintes métodos:
- `__init__` — inicializa pesos e bias (inicialização He)
- `forward` — propaga os dados da entrada até a saída
- `backward` — calcula os gradientes via backpropagation
- `train` — loop de treinamento com gradient descent

**`train_model.py`**
Script principal para rodar o treinamento. Carrega o `train.csv`, normaliza os dados, divide em treino/validação (70%/30%) e treina a rede.

---

## Como Rodar

1. Clone o repositório:
```bash
git clone https://github.com/Paul0Anselmi/Mobile_Price_Classification.git
cd Mobile_Price_Classification
```

2. Instale a dependência necessária:
```bash
pip install numpy pandas
```

3. Entre na pasta e rode o treinamento:
```bash
cd "Neural Network Classification"
python train_model.py
```

---

## Dataset

**Mobile Price Classification** — disponível no [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).

| Variável | Descrição |
|---|---|
| battery_power | Capacidade da bateria (mAh) |
| blue | Tem Bluetooth? |
| clock_speed | Velocidade do processador |
| dual_sim | Tem dual SIM? |
| fc | Megapixels da câmera frontal |
| four_g | Tem 4G? |
| int_memory | Memória interna (GB) |
| ram | Memória RAM (MB) |
| price_range | **Variável alvo**: 0 (baixo), 1 (médio), 2 (alto), 3 (muito alto) |

---

## Arquitetura da Rede

```
Entrada (20) → Camada Oculta (16) → Camada Oculta (8) → Saída (4)
```

- **Ativação nas camadas ocultas:** ReLU
- **Ativação na saída:** Softmax
- **Função de custo:** Cross-Entropy
- **Algoritmo de treino:** Backpropagation com Gradient Descent
- **Divisão dos dados:** 70% treino / 30% validação
