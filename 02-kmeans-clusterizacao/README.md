# Trabalho Prático 02 — Agrupamento com K-Means

Aplicação de aprendizado **não supervisionado** com o algoritmo K-Means sobre a base Iris.
O algoritmo foi implementado do zero, incluindo a inicialização k-means++ e as métricas
WCSS e Silhouette.

## Problema

Descobrir a estrutura de grupos da base Iris (150 amostras, 4 atributos) **sem acesso aos
rótulos**, e analisar como a escolha do número de clusters afeta o resultado. Os rótulos das
espécies entram apenas na análise final, para interpretar o que os grupos representam — nunca
como entrada do algoritmo.

## Implementação

O ciclo do algoritmo:

1. **Inicializar** K centróides com **k-means++** — o primeiro é sorteado uniformemente e cada
   seguinte com probabilidade proporcional ao quadrado da distância ao centróide mais próximo,
   o que os espalha e reduz a chance de mínimo local ruim
2. **Atribuir** cada ponto ao centróide mais próximo (distância euclidiana)
3. **Atualizar** cada centróide para a média dos pontos do seu grupo
4. **Repetir** até os centróides estabilizarem

Como o K-Means converge apenas para mínimos locais, cada valor de K é executado 10 vezes,
retendo-se a solução de menor WCSS. Semente fixa em 42.

### Padronização

A base foi processada em **dois cenários**, brutos e padronizados por Z-score. A padronização
não é obrigatória aqui, já que os quatro atributos estão na mesma unidade (cm) e ordem de
grandeza — rodar os dois permite medir o efeito em vez de assumi-lo.

## Resultados

| K | WCSS (brutos) | Silhouette (brutos) | WCSS (padronizados) | Silhouette (padronizados) |
|---|---|---|---|---|
| 2 | 152,348 | **0,6810** | 222,362 | **0,5818** |
| 3 | 78,851 | 0,5528 | 139,820 | 0,4599 |
| 4 | 57,256 | 0,4975 | 114,505 | 0,4133 |
| 5 | 46,446 | 0,4887 | 90,808 | 0,3455 |

**Matriz de contingência (K = 3, dados brutos):**

| espécie \ grupo | grupo 0 | grupo 1 | grupo 2 |
|---|---|---|---|
| setosa | 0 | 50 | 0 |
| versicolor | 48 | 0 | 2 |
| virginica | 14 | 0 | 36 |

ARI = 0,7302 · Silhouette = 0,5528

### Principais observações

- **O Método do Cotovelo aponta K = 3**, coincidindo com o número real de espécies: a passagem
  de K=2 para K=3 reduz o WCSS em 48,2%, contra 27,4% e 18,9% nas transições seguintes.
- **O Silhouette aponta K = 2**, e a discordância é informativa. O cotovelo busca a saturação
  da compactação; a silhueta premia grupos bem separados. Na Iris, *setosa* fica isolada
  enquanto *versicolor* e *virginica* se sobrepõem — separá-las cria uma fronteira no meio de
  uma região densa, o que derruba a silhueta mesmo sendo a partição biologicamente correta.
  Métricas internas descrevem a geometria, não a semântica.
- **Padronizar piorou o resultado** (Silhouette 0,5528 → 0,4599; ARI 0,7302 → 0,6201). O
  Z-score dá peso igual a todos os atributos, e na Iris as medidas da pétala discriminam as
  espécies enquanto a largura da sépala discrimina pouco. Equalizar as escalas amplifica o
  atributo menos informativo. Pré-processamento é decisão a justificar, não passo automático.
- **Validação**: a implementação reproduz exatamente o `KMeans` do Scikit-learn — mesmo WCSS,
  mesmo Silhouette e ARI = 1,0 entre as partições.

## Como executar

```bash
pip install numpy scikit-learn matplotlib
```

Abra `kmeans_iris.ipynb` no Jupyter ou no Google Colab — o notebook já contém as saídas e os
gráficos gerados.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `kmeans_iris.ipynb` | Notebook com implementação, saídas e gráficos |
| `relatorio.pdf` | Relatório com resultados, análise e conclusão |
| `apresentacao_kmeans.pptx` | Slides da apresentação |
