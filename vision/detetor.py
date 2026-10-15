"""Deteta os alimentos na foto e mede a área (cm²) de cada um."""
import os

import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import CalibradorPrato

# YOLOv8s-seg treinado no FoodSeg103 (notebooks/treinar_detetor_colab.ipynb)
CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), '..', 'modelos', 'yolov8s_foodseg103.pt')

# Cores dos alimentos na foto anotada (a app usa as mesmas na lista)
CORES = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
         '#42d4f4', '#f032e6', '#9a6324', '#469990', '#808000']


def hex_para_bgr(cor):
    r, g, b = (int(cor[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


class SmartCarbDetector:
    def __init__(self, caminho_modelo: str = CAMINHO_MODELO,
                 diametro_prato_cm: float = 26.0):
        self.modelo = YOLO(caminho_modelo)
        self.calibrador = CalibradorPrato(diametro_prato_cm)

    def process_meal(self, caminho_imagem: str, salvar_anotada: bool = True, conf: float = 0.20) -> list[dict]:
        imagem_bgr = cv2.imread(caminho_imagem)
        if imagem_bgr is None:
            raise FileNotFoundError(f"Não consegui abrir a imagem: {caminho_imagem}")

        escala = self.calibrador.calcular_escala(imagem_bgr)
        if escala is None:
            raise ValueError(
                "Não foi possível medir o prato nesta foto. Tenta uma foto com o prato inteiro visível."
            )

        r = self.modelo(caminho_imagem, conf=conf, verbose=False)[0]
        deteções = []
        if r.masks is not None:
            for poligono, classe, confianca in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
                area_pixeis = cv2.contourArea(poligono.astype(np.float32))
                deteções.append({
                    'detetado': self.modelo.names[int(classe)],
                    'confianca': float(confianca),
                    'area_cm2': self.calibrador.pixels_para_cm2(area_pixeis, escala),
                    'cor': CORES[len(deteções) % len(CORES)],
                    'poligono': poligono.astype(np.int32),
                })

        if salvar_anotada:
            cv2.imwrite("resultado_detecao.jpg", desenhar(imagem_bgr, deteções))

        for det in deteções:
            del det['poligono']
        return deteções


def desenhar(imagem_bgr, deteções):
    """Pinta cada alimento com a sua cor e escreve o seu número no centro."""
    pintada = imagem_bgr.copy()
    for det in deteções:
        cv2.fillPoly(pintada, [det['poligono']], hex_para_bgr(det['cor']))
    imagem = cv2.addWeighted(pintada, 0.3, imagem_bgr, 0.7, 0)

    tamanho = max(imagem.shape[:2]) / 1000  # para o desenho ter o mesmo aspeto em fotos grandes e pequenas
    for n, det in enumerate(deteções, start=1):
        cor = hex_para_bgr(det['cor'])
        cv2.polylines(imagem, [det['poligono']], True, cor, max(2, int(3 * tamanho)))

        m = cv2.moments(det['poligono'])
        if m['m00'] == 0:
            continue
        cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
        raio = int(22 * tamanho)
        cv2.circle(imagem, (cx, cy), raio, cor, -1)
        cv2.circle(imagem, (cx, cy), raio, (255, 255, 255), max(2, int(3 * tamanho)))
        texto = str(n)
        escala_txt = 0.9 * tamanho
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala_txt, max(1, int(2 * tamanho)))
        cv2.putText(imagem, texto, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    escala_txt, (255, 255, 255), max(1, int(2 * tamanho)), cv2.LINE_AA)
    return imagem
