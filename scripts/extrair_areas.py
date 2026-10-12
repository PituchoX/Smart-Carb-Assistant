"""Mede a área da comida (em píxeis) em cada foto do Nutrition5k -> dados/nutrition5k/areas.csv"""
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PASTA_FOTOS = os.path.join(PASTA_PROJETO, 'dados', 'nutrition5k', 'fotos')
CAMINHO_SAIDA = os.path.join(PASTA_PROJETO, 'dados', 'nutrition5k', 'areas.csv')
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, 'modelo_segmentacao_alimentos.pt')

modelo = YOLO(CAMINHO_MODELO)
fotos = sorted(os.listdir(PASTA_FOTOS))
linhas = []

for i, nome in enumerate(fotos, start=1):
    r = modelo(os.path.join(PASTA_FOTOS, nome), conf=0.2, verbose=False)[0]
    altura, largura = r.orig_shape
    mascara_total = np.zeros((altura, largura), dtype=np.uint8)
    classes = []

    if r.masks is not None:
        for poligono, classe in zip(r.masks.xy, r.boxes.cls):
            cv2.fillPoly(mascara_total, [poligono.astype(np.int32)], 1)
            classes.append(modelo.names[int(classe)])

    linhas.append({
        'prato': nome.removesuffix('.png'),
        'area_px': int(mascara_total.sum()),
        'n_detecoes': len(classes),
        'classes_detetadas': ', '.join(classes),
    })

    if i % 100 == 0:
        print(f'{i}/{len(fotos)} fotos processadas')

pd.DataFrame(linhas).to_csv(CAMINHO_SAIDA, index=False)
print(f'Guardado: {CAMINHO_SAIDA}')
