"""
Descarrega o FoodSeg103 do Hugging Face e converte-o para o formato do YOLO,
para avaliar (e, mais tarde, treinar) modelos de segmentação de alimentos.

FoodSeg103: https://huggingface.co/datasets/EduardoPacheco/FoodSeg103 (Apache 2.0)

Conjuntos criados:
- test:  as 2135 fotos de teste oficiais (só para a comparação final)
- train: 90% das fotos de treino oficiais
- val:   os outros 10%, que o YOLO usa durante o treino para escolher o melhor modelo
         (assim as fotos de teste nunca influenciam o treino)

No FoodSeg103 cada foto tem uma máscara em que cada píxel tem o número do
alimento (0 = fundo, 1 = candy, ..., 103 = other ingredients). O YOLO precisa
de contornos (polígonos), por isso cada zona contínua de um alimento passa a
ser um polígono. A classe v da máscara corresponde à classe v-1 do modelo.

Uso:  python scripts/preparar_foodseg103.py            (só o conjunto de teste)
      python scripts/preparar_foodseg103.py --treino   (também treino e validação)
"""
import io
import json
import os
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PASTA_SAIDA = os.path.join(PASTA_PROJETO, 'dados', 'foodseg103')
URL_BASE = 'https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/resolve/main'

FICHEIROS = {
    'test': ['data/validation-00000-of-00001.parquet'],
    'train': [f'data/train-0000{i}-of-00003.parquet' for i in range(3)],
}
FRACAO_VALIDACAO = 0.1
AREA_MINIMA_PX = 50  # ignora pedacinhos minúsculos (ruído da anotação)


def descarregar(caminho_remoto):
    destino = os.path.join(PASTA_SAIDA, 'parquet', os.path.basename(caminho_remoto))
    if not os.path.exists(destino):
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        print(f'  a descarregar {caminho_remoto}...')
        urllib.request.urlretrieve(f'{URL_BASE}/{caminho_remoto}', destino)
    return destino


def mascara_para_poligonos(mascara):
    """Devolve as linhas do ficheiro de anotação YOLO: 'classe x1 y1 x2 y2 ...' (normalizado 0-1)."""
    altura, largura = mascara.shape
    linhas = []
    for valor in np.unique(mascara):
        if valor == 0:
            continue  # fundo
        zona = (mascara == valor).astype(np.uint8)
        contornos, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            if cv2.contourArea(contorno) < AREA_MINIMA_PX or len(contorno) < 3:
                continue
            pontos = contorno[:, 0, :].astype(float) / [largura, altura]
            coords = ' '.join(f'{v:.5f}' for v in pontos.flatten())
            linhas.append(f'{int(valor) - 1} {coords}')
    return linhas


def guardar(linha, split):
    """Guarda a foto e o ficheiro de anotação YOLO de uma linha do dataset."""
    nome = f"{linha['id']:05d}"
    pasta_img = os.path.join(PASTA_SAIDA, 'images', split)
    pasta_lbl = os.path.join(PASTA_SAIDA, 'labels', split)
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(pasta_lbl, exist_ok=True)
    Image.open(io.BytesIO(linha['image']['bytes'])).convert('RGB').save(os.path.join(pasta_img, f'{nome}.jpg'))
    mascara = np.array(Image.open(io.BytesIO(linha['label']['bytes'])))
    with open(os.path.join(pasta_lbl, f'{nome}.txt'), 'w') as f:
        f.write('\n'.join(mascara_para_poligonos(mascara)))


def converter(origem):
    df = pd.concat([pd.read_parquet(descarregar(c)) for c in FICHEIROS[origem]], ignore_index=True)
    if origem == 'test':
        destinos = ['test'] * len(df)
    else:
        # Separação aleatória mas fixa (random_state), para ser sempre a mesma
        validacao = set(df.sample(frac=FRACAO_VALIDACAO, random_state=42).index)
        destinos = ['val' if i in validacao else 'train' for i in df.index]

    for (_, linha), destino in zip(df.iterrows(), destinos):
        guardar(linha, destino)
    for destino in sorted(set(destinos)):
        print(f'  {destino}: {destinos.count(destino)} fotos convertidas')


def escrever_data_yaml():
    with urllib.request.urlopen(f'{URL_BASE}/id2label.json') as r:
        id2label = {int(k): v.strip() for k, v in json.load(r).items()}
    nomes = [id2label[i] for i in range(1, 104)]  # sem o fundo (0)
    with open(os.path.join(PASTA_SAIDA, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'path: {PASTA_SAIDA}\ntrain: images/train\nval: images/val\ntest: images/test\n')
        f.write('names:\n' + ''.join(f'  {i}: {json.dumps(n)}\n' for i, n in enumerate(nomes)))


if __name__ == '__main__':
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    escrever_data_yaml()
    converter('test')
    if '--treino' in sys.argv:
        converter('train')
    print(f'Pronto: {PASTA_SAIDA}')
