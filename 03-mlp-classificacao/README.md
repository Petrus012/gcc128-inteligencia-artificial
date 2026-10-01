# Trabalho Prático 03 — Classificação com MLPClassifier

Rede neural **MLP** (Multi-Layer Perceptron) aplicada às bases **Iris** e **Wine**, usando o
`MLPClassifier` do Scikit-learn, com comparação direta contra o KNN do
[Trabalho Prático 01](../01-knn-classificacao).

## Problema

Classificar amostras de duas bases de características distintas e avaliar em que condições uma
rede neural supera um método mais simples:

- **Iris** — 150 amostras, 4 atributos, 3 espécies
- **Wine** — 178 amostras, 13 atributos, 3 cultivares

## Implementação

```python
make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42))
```

Uma camada oculta de 100 neurônios, ativação ReLU e otimizador Adam. Partição de 70/30
estratificada, com semente fixa em 42 — o MLP e o KNN recebem exatamente as mesmas amostras,
condição para que a comparação seja justa.

### Por que padronizar

A padronização entra dentro de um `Pipeline`, de modo que média e desvio padrão são estimados
apenas no conjunto de treino, sem vazamento para o teste. A decisão foi verificada
empiricamente, repetindo o treino com 10 sementes:

| Base | Sem padronização | Com padronização |
|---|---|---|
| Iris | 0,9778 ± 0,0000 | 0,9178 ± 0,0102 |
| Wine | 0,7481 ± **0,2552** | 0,9815 ± 0,0117 |

Na Wine, o atributo *proline* varia em mais de 1.400 unidades enquanto outros variam menos de
1. Sem padronizar, o gradiente fica dominado por esse atributo e o treinamento desestabiliza —
em algumas sementes a acurácia cai a 27,78%.

## Resultados

| Base | Classificador | Acurácia | Precisão (macro) | Revocação (macro) |
|---|---|---|---|---|
| Iris | MLPClassifier | 91,11% | 0,9155 | 0,9111 |
| Iris | KNN (k = 5) | 91,11% | 0,9298 | 0,9111 |
| Wine | **MLPClassifier** | **100,00%** | **1,0000** | **1,0000** |
| Wine | KNN (k = 5) | 94,44% | 0,9444 | 0,9524 |

### Principais observações

- **Na Iris os dois empatam.** Com 4 atributos e uma classe linearmente separável, a fronteira
  de decisão exigida é simples e a capacidade extra da rede não encontra estrutura adicional a
  explorar. Ambos erram na mesma região, entre *versicolor* e *virginica* — o mesmo
  comportamento do TP01.
- **Na Wine o MLP classifica corretamente todas as 54 amostras de teste.** Com 13 atributos, a
  classe decorre de combinações entre eles, que a rede aprende nos pesos da camada oculta. O
  KNN depende de que amostras da mesma classe permaneçam próximas no espaço original —
  propriedade que se degrada conforme a dimensão aumenta.
- **Naturezas diferentes.** O KNN é baseado em instâncias: sem fase de treino, custo todo na
  classificação, um hiperparâmetro. O MLP constrói um modelo paramétrico (523 iterações na
  Iris, 264 na Wine) e depois classifica quase instantaneamente, mas exige definir arquitetura,
  ativação, otimizador e taxa de aprendizado.
- **O ganho da rede neural não é automático.** Ele aparece quando o problema tem complexidade
  que o justifique.

## Como executar

```bash
pip install numpy scikit-learn matplotlib
```

Abra `mlp_iris_wine.ipynb` no Jupyter ou no Google Colab — o notebook já contém as saídas e os
gráficos gerados.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `mlp_iris_wine.ipynb` | Notebook com implementação, saídas e gráficos |
| `relatorio.pdf` | Relatório de uma página com a análise comparativa |
| `apresentacao_mlp.pptx` | Slides da apresentação |
