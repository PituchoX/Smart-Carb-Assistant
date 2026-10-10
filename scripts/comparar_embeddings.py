"""
Compara duas formas de ligar a classe detetada pelo modelo (ex: 'rice')
à linha certa da tabela de alimentos:

1. Pesquisa automática por semelhança de texto (embeddings) — a primeira
   abordagem que experimentei.
2. Coluna classe_modelo no Excel, revista à mão — a abordagem atual.

A coluna classe_modelo serve de "resposta certa", e medimos quantas vezes
a pesquisa automática acerta na mesma linha.
"""
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_EXCEL = os.path.join(PASTA_PROJETO, 'dados', 'tabela_alimentos.xlsx')

df = pd.read_excel(CAMINHO_EXCEL)

# Pares (classe, linha certa) a partir da coluna classe_modelo
pares = []
for i, linha in df.dropna(subset=['classe_modelo']).iterrows():
    for classe in linha['classe_modelo'].split(','):
        pares.append((classe.strip(), i))

modelo = SentenceTransformer('sentence-transformers/LaBSE')
vetores_alimentos = modelo.encode(df['alimento'].tolist(), normalize_embeddings=True)
vetores_classes = modelo.encode([classe for classe, _ in pares], normalize_embeddings=True)

# Para cada classe, a linha da tabela com o texto mais parecido
escolhas = np.argmax(vetores_classes @ vetores_alimentos.T, axis=1)

resultados = []
for (classe, certa), escolhida in zip(pares, escolhas):
    hc_certo = df.at[certa, 'hc_por_100g']
    hc_escolhido = df.at[escolhida, 'hc_por_100g']
    resultados.append({
        'classe': classe,
        'linha_certa': df.at[certa, 'alimento'],
        'linha_escolhida': df.at[escolhida, 'alimento'],
        'acertou': certa == escolhida,
        'erro_hc_por_100g': abs(hc_escolhido - hc_certo),
    })
resultados = pd.DataFrame(resultados)

print(f"Classes avaliadas: {len(resultados)}")
print(f"Acertos da pesquisa automática: {resultados['acertou'].sum()} "
      f"({resultados['acertou'].mean():.0%})")
print(f"Erro médio de HC quando falha: "
      f"{resultados.loc[~resultados['acertou'], 'erro_hc_por_100g'].mean():.1f} g por 100 g")
print("\nPiores erros:")
piores = resultados[~resultados['acertou']].sort_values('erro_hc_por_100g', ascending=False)
print(piores.head(10).to_string(index=False))
