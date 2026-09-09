# Trabalho Prático 01 — Classificação KNN

Implementação do algoritmo K-Nearest Neighbors do zero, sem bibliotecas de aprendizado de
máquina, e comparação com a implementação do Scikit-learn sobre a base Iris.

## Problema

Classificar amostras da base Iris (150 amostras, 4 atributos, 3 espécies) em suas respectivas
classes, avaliando o impacto do hiperparâmetro k na taxa de reconhecimento.

## Implementação

O classificador desenvolvido do zero usa apenas NumPy, para operações matemáticas básicas.
As quatro etapas:

1. **Distância euclidiana** entre a amostra de teste e todos os padrões de treinamento
2. **Ordenação** das distâncias e seleção dos k vizinhos mais próximos
3. **Voto majoritário** entre as classes desses vizinhos (empate resolvido pelo menor rótulo)
4. **Predição** com a classe vencedora

A matriz de confusão e as métricas de precisão, revocação e acurácia também foram
implementadas manualmente, derivadas da diagonal e das somas de linhas e colunas da matriz.

A base é particionada em 60% treino e 40% teste, com semente fixa em 42 — o que garante que
ambas as implementações recebam exatamente os mesmos conjuntos, tornando a comparação justa e
os resultados reprodutíveis.

## Resultados

| k | Acurácia (do zero) | Acurácia (Sklearn) | Precisão média | Revocação média | Predições divergentes |
|---|---|---|---|---|---|
| 1 | 95,00% | 95,00% | 0,9404 | 0,9472 | 0 / 60 |
| 3 | 96,67% | 96,67% | 0,9617 | 0,9617 | 0 / 60 |
| 5 | **98,33%** | **98,33%** | 0,9861 | 0,9762 | 0 / 60 |
| 7 | 96,67% | 96,67% | 0,9617 | 0,9617 | 0 / 60 |

**Matriz de confusão (k = 5):**

| real \ predito | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 23 | 0 | 0 |
| **versicolor** | 0 | 13 | 1 |
| **virginica** | 0 | 0 | 23 |

### Principais observações

- As duas implementações produziram resultados **rigorosamente idênticos** em todos os valores
  de k, com zero predições divergentes. Reproduzir exatamente a saída de uma biblioteca
  consolidada é uma forma objetiva de validar a implementação própria.
- A diferença está no custo computacional: medindo a média de 10 execuções após aquecimento, o
  Scikit-learn foi cerca de 13× mais rápido. A implementação do zero percorre todos os pontos
  em laço explícito, com custo O(n·m) por predição, enquanto a biblioteca usa código vetorizado
  compilado e índice espacial (KD-Tree, neste caso).
- k = 5 obteve o melhor resultado. Com k = 1 a decisão depende de um único vizinho e fica
  sensível a ruído; valores altos demais reduzem a capacidade de discriminação.
- A classe *setosa* nunca foi confundida, por ser linearmente separável. Os erros
  concentraram-se entre *versicolor* e *virginica*, que se sobrepõem no espaço de atributos.

## Como executar

```bash
pip install numpy scikit-learn matplotlib
python knn.py
```

Ou abra `knn_iris.ipynb` no Jupyter ou no Google Colab — o notebook já contém as saídas e os
gráficos gerados.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `knn_iris.ipynb` | Notebook com implementação, saídas e gráficos |
| `knn.py` | Mesma implementação em formato de script |
| `relatorio.pdf` | Relatório de uma página com a análise comparativa |
| `apresentacao.pptx` | Slides da apresentação |
