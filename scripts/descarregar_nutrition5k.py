"""Descarrega os metadados e as fotos de cima do Nutrition5k (pratos com hidratos e de um só ingrediente)."""
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pandas as pd

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PASTA_N5K = os.path.join(PASTA_PROJETO, 'dados', 'nutrition5k')
PASTA_FOTOS = os.path.join(PASTA_N5K, 'fotos')
URL_BASE = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset'

FICHEIROS_METADADOS = [
    'metadata/dish_metadata_cafe1.csv',
    'metadata/dish_metadata_cafe2.csv',
    'dish_ids/splits/rgb_train_ids.txt',
    'dish_ids/splits/rgb_test_ids.txt',
]

INGREDIENTES_ALVO = [
    'white rice', 'brown rice', 'fried rice', 'wild rice', 'country rice',
    'potatoes', 'roasted potatoes', 'sweet potato', 'hash browns',
    'bread', 'wheat bread', 'bagels',
    'pasta', 'noodles', 'rice noodles',
    'pizza', 'cheese pizza', 'pepperoni pizza',
]


def descarregar(url, destino):
    """Descarrega um ficheiro. Devolve False se não existir no servidor."""
    if os.path.exists(destino):
        return True
    try:
        urllib.request.urlretrieve(url, destino)
        return True
    except HTTPError as e:
        if e.code == 404:
            return False
        raise


def ler_metadados():
    """
    No CSV original cada linha é um prato e os ingredientes vêm todos seguidos
    na mesma linha, em grupos de 7 colunas:
    id, nome, gramas, calorias, gordura, hidratos, proteína
    """
    linhas = []
    for nome in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']:
        with open(os.path.join(PASTA_N5K, nome), encoding='utf-8') as f:
            for linha in f:
                partes = linha.strip().split(',')
                prato = partes[0]
                for i in range(6, len(partes), 7):
                    linhas.append({
                        'prato': prato,
                        'ingrediente': partes[i + 1],
                        'peso_g': float(partes[i + 2]),
                        'hc_g': float(partes[i + 5]),
                    })
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    os.makedirs(PASTA_FOTOS, exist_ok=True)

    print('1. A descarregar metadados...')
    for caminho in FICHEIROS_METADADOS:
        descarregar(f'{URL_BASE}/{caminho}', os.path.join(PASTA_N5K, os.path.basename(caminho)))

    df = ler_metadados()
    df.to_csv(os.path.join(PASTA_N5K, 'ingredientes.csv'), index=False)

    # Só os pratos que têm foto de cima (listados nos ficheiros de splits)
    com_foto = set()
    for nome in ['rgb_train_ids.txt', 'rgb_test_ids.txt']:
        with open(os.path.join(PASTA_N5K, nome)) as f:
            com_foto |= set(f.read().split())

    # Pratos com alimentos ricos em hidratos + todos os pratos de um só
    # ingrediente (nesses sabemos exatamente o que a foto mostra e quanto pesa)
    n_ingredientes = df.groupby('prato').size()
    um_ingrediente = n_ingredientes[n_ingredientes == 1].index
    escolhidos = df['ingrediente'].isin(INGREDIENTES_ALVO) | df['prato'].isin(um_ingrediente)
    pratos = df[escolhidos & df['prato'].isin(com_foto)]['prato'].unique()
    print(f'2. A descarregar fotos de {len(pratos)} pratos (pode demorar alguns minutos)...')

    def descarregar_foto(prato):
        url = f'{URL_BASE}/imagery/realsense_overhead/{prato}/rgb.png'
        return descarregar(url, os.path.join(PASTA_FOTOS, f'{prato}.png'))

    with ThreadPoolExecutor(max_workers=16) as executor:
        resultados = list(executor.map(descarregar_foto, pratos))

    print(f'Concluído: {sum(resultados)} fotos descarregadas, '
          f'{resultados.count(False)} pratos sem foto no servidor.')
