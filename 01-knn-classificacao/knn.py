"""
GCC128 - Inteligencia Artificial
Trabalho Pratico 01 - Classificacao KNN
Implementacao hardcore (do zero) x Sklearn - base Iris
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

VALORES_K = [1, 3, 5, 7]
PROP_TREINO = 0.6
SEMENTE = 42       # fixa para garantir reprodutibilidade dos resultados
REPETICOES = 10    # execucoes por medicao de tempo (media e desvio padrao)


# ----------------------------------------------------------------------
# 1. CARGA E PARTICAO DOS DADOS
# ----------------------------------------------------------------------
def carregar_dados():
    base = load_iris()
    return base.data, base.target, base.target_names


def dividir_treino_teste(X, y, prop_treino, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    corte = int(len(X) * prop_treino)
    tr, te = indices[:corte], indices[corte:]
    return X[tr], y[tr], X[te], y[te]


# ----------------------------------------------------------------------
# 2. KNN HARDCORE
# ----------------------------------------------------------------------
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def voto_majoritario(rotulos):
    valores, contagens = np.unique(rotulos, return_counts=True)
    return valores[np.argmax(contagens)]


def knn_predizer(X_treino, y_treino, X_teste, k):
    predicoes = []
    for amostra in X_teste:
        distancias = np.array([distancia_euclidiana(amostra, t) for t in X_treino])
        vizinhos = np.argsort(distancias)[:k]
        predicoes.append(voto_majoritario(y_treino[vizinhos]))
    return np.array(predicoes)


# ----------------------------------------------------------------------
# 3. METRICAS HARDCORE
# ----------------------------------------------------------------------
def matriz_confusao(y_real, y_pred, n_classes):
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1
    return matriz


def calcular_metricas(matriz):
    diagonal = np.diag(matriz)
    soma_colunas = matriz.sum(axis=0)
    soma_linhas = matriz.sum(axis=1)
    precisao = np.divide(diagonal, soma_colunas,
                         out=np.zeros(len(diagonal)), where=soma_colunas != 0)
    revocacao = np.divide(diagonal, soma_linhas,
                          out=np.zeros(len(diagonal)), where=soma_linhas != 0)
    acuracia = diagonal.sum() / matriz.sum()
    return precisao, revocacao, acuracia


# ----------------------------------------------------------------------
# 4. MEDICAO DE TEMPO COM REPETICOES
# ----------------------------------------------------------------------
def medir(funcao, repeticoes=REPETICOES):
    """Executa uma vez para aquecimento e depois mede N repeticoes.
    Retorna media e desvio padrao em milissegundos."""
    funcao()
    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao()
        tempos.append((time.perf_counter() - inicio) * 1000)
    return float(np.mean(tempos)), float(np.std(tempos))


# ----------------------------------------------------------------------
# 5. PLOTAGEM
# ----------------------------------------------------------------------
def plotar_avaliacao(matriz, precisao, revocacao, acuracia, nomes, k, arquivo=None):
    """Plota a matriz de confusao como mapa de calor e as metricas por classe."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    im = ax1.imshow(matriz, cmap="Purples")
    ax1.set_xticks(range(len(nomes)), nomes, rotation=20)
    ax1.set_yticks(range(len(nomes)), nomes)
    ax1.set_xlabel("Classe predita")
    ax1.set_ylabel("Classe real")
    ax1.set_title(f"Matriz de confusão (k = {k})")
    limite = matriz.max() / 2
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax1.text(j, i, matriz[i, j], ha="center", va="center",
                     color="white" if matriz[i, j] > limite else "black",
                     fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    x = np.arange(len(nomes))
    largura = 0.38
    b1 = ax2.bar(x - largura / 2, precisao, largura, label="Precisão", color="#5B3E8E")
    b2 = ax2.bar(x + largura / 2, revocacao, largura, label="Revocação", color="#7FA88C")
    ax2.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax2.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax2.axhline(acuracia, color="#D99A3E", linestyle="--", linewidth=1.5,
                label=f"Acurácia geral = {acuracia:.4f}")
    ax2.set_xticks(x, nomes)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Valor da métrica")
    ax2.set_title(f"Métricas de avaliação por classe (k = {k})")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
               fontsize=8, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=140, bbox_inches="tight")
    return fig


def plotar_acuracias(resumo, arquivo=None):
    """Compara a acuracia das duas implementacoes para cada k."""
    ks = [r[0] for r in resumo]
    x = np.arange(len(ks))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b1 = ax.bar(x - largura / 2, [r[1] * 100 for r in resumo], largura,
                label="Implementação hardcore", color="#5B3E8E")
    b2 = ax.bar(x + largura / 2, [r[2] * 100 for r in resumo], largura,
                label="Scikit-learn", color="#7FA88C")
    ax.bar_label(b1, fmt="%.2f%%", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f%%", fontsize=8, padding=2)
    ax.set_xticks(x, [f"k = {k}" for k in ks])
    ax.set_ylim(90, 101)
    ax.set_ylabel("Taxa de reconhecimento (%)")
    ax.set_title("Acurácia por valor de k")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=140, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# 6. IMPRESSAO
# ----------------------------------------------------------------------
def imprimir_matriz(matriz, nomes):
    cabecalho = "".join(f"{n[:10]:>12}" for n in nomes)
    print(f"{'real/predito':>14}{cabecalho}")
    for i, nome in enumerate(nomes):
        linha = "".join(f"{v:>12}" for v in matriz[i])
        print(f"{nome:>14}{linha}")


def imprimir_resultado(titulo, matriz, precisao, revocacao, acuracia, nomes, media, desvio):
    print(f"\n--- {titulo} ---")
    imprimir_matriz(matriz, nomes)
    print(f"\n{'classe':>14}{'precisao':>12}{'revocacao':>12}")
    for i, nome in enumerate(nomes):
        print(f"{nome:>14}{precisao[i]:>12.4f}{revocacao[i]:>12.4f}")
    print(f"{'MEDIA':>14}{precisao.mean():>12.4f}{revocacao.mean():>12.4f}")
    print(f"Acuracia (taxa de reconhecimento): {acuracia:.4f}  ({acuracia*100:.2f}%)")
    print(f"Tempo: {media:.2f} +/- {desvio:.2f} ms (media de {REPETICOES} execucoes)")


# ----------------------------------------------------------------------
# 7. EXECUCAO PRINCIPAL
# ----------------------------------------------------------------------
def main():
    print("=" * 62)
    print("GCC128 - TRABALHO PRATICO 01 - CLASSIFICACAO KNN")
    print("=" * 62)
    print(f"Semente utilizada: {SEMENTE} (garante que ambas as implementacoes")
    print("recebam exatamente os mesmos conjuntos de treino e teste)")

    X, y, nomes = carregar_dados()
    X_tr, y_tr, X_te, y_te = dividir_treino_teste(X, y, PROP_TREINO, SEMENTE)
    n_classes = len(nomes)

    print(f"Base Iris: {len(X)} amostras, {X.shape[1]} atributos, {n_classes} classes")
    print(f"Treino: {len(X_tr)} amostras ({PROP_TREINO*100:.0f}%) | "
          f"Teste: {len(X_te)} amostras ({(1-PROP_TREINO)*100:.0f}%)")

    resumo = []

    for k in VALORES_K:
        print("\n" + "=" * 62)
        print(f"K = {k}")
        print("=" * 62)

        # --- hardcore ---
        pred_hc = knn_predizer(X_tr, y_tr, X_te, k)
        m_hc, d_hc = medir(lambda: knn_predizer(X_tr, y_tr, X_te, k))
        mc_hc = matriz_confusao(y_te, pred_hc, n_classes)
        prec_hc, rev_hc, acc_hc = calcular_metricas(mc_hc)
        imprimir_resultado("IMPLEMENTACAO HARDCORE", mc_hc, prec_hc, rev_hc,
                           acc_hc, nomes, m_hc, d_hc)

        # --- sklearn ---
        def rodar_sklearn():
            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_tr, y_tr)
            return modelo.predict(X_te)

        pred_sk = rodar_sklearn()
        m_sk, d_sk = medir(rodar_sklearn)
        mc_sk = confusion_matrix(y_te, pred_sk)
        prec_sk = precision_score(y_te, pred_sk, average=None, zero_division=0)
        rev_sk = recall_score(y_te, pred_sk, average=None, zero_division=0)
        acc_sk = accuracy_score(y_te, pred_sk)
        imprimir_resultado("IMPLEMENTACAO SKLEARN", mc_sk, prec_sk, rev_sk,
                           acc_sk, nomes, m_sk, d_sk)

        divergencias = int(np.sum(pred_hc != pred_sk))
        print(f"\nPredicoes divergentes entre as duas implementacoes: "
              f"{divergencias}/{len(y_te)}")

        plotar_avaliacao(mc_hc, prec_hc, rev_hc, acc_hc, nomes, k,
                         arquivo=f"avaliacao_k{k}.png")
        plt.close("all")

        resumo.append((k, acc_hc, acc_sk, m_hc, d_hc, m_sk, d_sk, divergencias))

    # --- estrutura efetivamente escolhida pelo sklearn ---
    modelo = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    print("\n" + "=" * 62)
    print(f"Estrutura selecionada pelo Sklearn (algorithm='auto'): {modelo._fit_method}")

    # --- tabela comparativa ---
    print("=" * 62)
    print("RESUMO COMPARATIVO")
    print("=" * 62)
    print(f"{'k':>3}{'acc hardcore':>15}{'acc sklearn':>14}"
          f"{'t hardcore (ms)':>20}{'t sklearn (ms)':>18}{'razao':>8}{'difs':>7}")
    for k, a_hc, a_sk, m_hc, d_hc, m_sk, d_sk, dv in resumo:
        print(f"{k:>3}{a_hc:>15.4f}{a_sk:>14.4f}"
              f"{m_hc:>13.2f} ± {d_hc:<4.2f}{m_sk:>11.2f} ± {d_sk:<4.2f}"
              f"{m_hc/m_sk:>7.1f}x{dv:>7}")
    print(f"\nTempos: media +/- desvio padrao de {REPETICOES} execucoes, apos aquecimento.")

    plotar_acuracias(resumo, arquivo="acuracias.png")
    plt.close("all")

    # --- impacto do tamanho da base de treino ---
    print("\n" + "=" * 62)
    print("IMPACTO DO TAMANHO DA BASE DE APRENDIZAGEM (k=3)")
    print("=" * 62)
    print(f"{'% treino':>10}{'n treino':>10}{'acuracia':>12}")
    for prop in [0.2, 0.4, 0.6, 0.8]:
        Xa, ya, Xb, yb = dividir_treino_teste(X, y, prop, SEMENTE)
        pred = knn_predizer(Xa, ya, Xb, 3)
        _, _, acc = calcular_metricas(matriz_confusao(yb, pred, n_classes))
        print(f"{prop*100:>9.0f}%{len(Xa):>10}{acc:>12.4f}")


if __name__ == "__main__":
    main()
