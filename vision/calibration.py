"""Encontra o prato com o YOLO-World e usa o seu diâmetro para converter píxeis em cm²."""
import os

import numpy as np
from ultralytics import YOLOWorld

PASTA_PROJETO = os.path.join(os.path.dirname(__file__), '..')
CAMINHO_YOLO_WORLD = os.path.join(PASTA_PROJETO, 'modelos', 'yolov8s-worldv2.pt')
DIAMETRO_PRATO_CM_DEFAULT = 26.0
CONFIANCA_MINIMA = 0.1
MARGEM_BORDA_PX = 3  # caixa a menos de 3 px da borda = prato cortado pela foto


class CalibradorPrato:
    def __init__(self, diametro_prato_cm: float = DIAMETRO_PRATO_CM_DEFAULT):
        self.diametro_cm = diametro_prato_cm
        self.modelo = YOLOWorld(CAMINHO_YOLO_WORLD)
        self.modelo.set_classes(['plate', 'bowl'])

    def calcular_escala(self, imagem_bgr: np.ndarray) -> tuple[float, float] | None:
        """Devolve (píxeis/cm na horizontal, píxeis/cm na vertical), ou None se não encontrar o prato."""
        r = self.modelo.predict(imagem_bgr, conf=CONFIANCA_MINIMA, verbose=False)[0]
        pratos = [(float(conf), caixa.tolist()) for conf, caixa, classe
                  in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)
                  if self.modelo.names[int(classe)] == 'plate']
        if not pratos:
            return None

        _, (x1, y1, x2, y2) = max(pratos)
        altura_img, largura_img = imagem_bgr.shape[:2]
        largura_px, altura_px = x2 - x1, y2 - y1

        # Numa foto tirada de lado o prato fica achatado, por isso a escala
        # é diferente na horizontal e na vertical. Se o prato sair da foto
        # num dos lados, usa-se a medida do outro.
        cortado_h = x1 < MARGEM_BORDA_PX or x2 > largura_img - MARGEM_BORDA_PX
        cortado_v = y1 < MARGEM_BORDA_PX or y2 > altura_img - MARGEM_BORDA_PX

        if cortado_h and cortado_v:
            return None
        if cortado_h:
            largura_px = altura_px
        if cortado_v:
            altura_px = largura_px

        return largura_px / self.diametro_cm, altura_px / self.diametro_cm

    def pixels_para_cm2(self, area_pixeis: float, escala: tuple[float, float]) -> float:
        """Converte uma área em píxeis para cm², dada a escala (horizontal, vertical)."""
        px_por_cm_h, px_por_cm_v = escala
        return area_pixeis / (px_por_cm_h * px_por_cm_v)
