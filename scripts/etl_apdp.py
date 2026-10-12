"""Extrai a tabela de hidratos do PDF da APDP para dados/tabela_alimentos.xlsx."""
import os
import re

import pandas as pd
import pdfplumber

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_PDF = os.path.join(PASTA_PROJETO, 'dados', 'DI-29_00-CHC_100g-Alimento_.pdf')
CAMINHO_EXCEL = os.path.join(PASTA_PROJETO, 'dados', 'tabela_alimentos.xlsx')

# Onde começa e acaba cada coluna (em pontos do PDF), para cada página
LIMITES_COLUNAS = {
    0: [0, 185, 360, 540],
    1: [0, 183, 345, 540],
}

SECCOES = [
    'Cereais e pseudo-cereais', 'Batata', 'Leguminosas', 'Pão', 'Bolachas',
    'Farinhas', 'Sopas', 'Lacticínios e bebidas vegetais', 'Fruta fresca',
    'Frutos amiláceos e oleaginosos', 'Frutos secos', 'Sementes',
    'Pratos mistos, cozinhados e outros',
]

# Na fruta há 2 valores (peso bruto e peso edível) e usamos o edível (o último)
SECCOES_FRUTA = ['Fruta fresca', 'Frutos secos']

# Exemplos: "Arroz solto cozido 28"  |  "Banana 12,9 21,8"  |  "Amora - 4,5"
PADRAO_LINHA = re.compile(r'^(.+?)\s+(?:(-|\d+(?:,\d+)?)\s+)?(\d+(?:,\d+)?)$')


def extrair_alimentos(caminho_pdf):
    alimentos = []

    with pdfplumber.open(caminho_pdf) as pdf:
        for num_pagina, pagina in enumerate(pdf.pages):
            limites = LIMITES_COLUNAS[num_pagina]

            for i in range(3):
                coluna = pagina.crop((limites[i], 40, limites[i + 1], pagina.height))
                texto = coluna.extract_text(x_tolerance=1.5) or ''
                # Única entrada que o PDF parte em 3 linhas
                texto = texto.replace('Iogurte sólido natural / 0%\n5\naçúcar',
                                      'Iogurte sólido natural / 0% açúcar 5')

                seccao = None
                for linha in texto.split('\n'):
                    linha = linha.strip()

                    titulo = linha.replace('g HC', '').strip()
                    if titulo in SECCOES:
                        seccao = titulo
                        continue
                    if seccao is None:
                        continue  # texto antes da primeira secção
                    if linha.startswith('*'):
                        seccao = None  # nota de rodapé = fim da secção
                        continue

                    m = PADRAO_LINHA.match(linha)
                    if not m:
                        continue
                    nome, valor_bruto, valor = m.groups()

                    # "Farinha de trigo tipo 55 74,3": o 55 é parte do nome
                    if valor_bruto and seccao not in SECCOES_FRUTA:
                        nome = f'{nome} {valor_bruto}'

                    # O * no PDF significa "elevada quantidade de gordura"
                    nome = nome.replace('*', '').replace('“', '').replace('”', '').strip()

                    alimentos.append({
                        'categoria': seccao,
                        'alimento': nome,
                        'hc_por_100g': float(valor.replace(',', '.')),
                    })

    return pd.DataFrame(alimentos)


if __name__ == '__main__':
    if os.path.exists(CAMINHO_EXCEL):
        print(f'{CAMINHO_EXCEL} já existe — apaga-o primeiro se o quiseres gerar de novo.')
    else:
        df = extrair_alimentos(CAMINHO_PDF)
        df['fonte'] = 'APDP'
        df['classe_modelo'] = ''
        df.to_excel(CAMINHO_EXCEL, index=False)
        print(f'{len(df)} alimentos extraídos do PDF e guardados em {CAMINHO_EXCEL}')
