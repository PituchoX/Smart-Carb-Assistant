"""
Calibração de escala: converte áreas em píxeis para cm², usando o prato
como referência de tamanho (por omissão, um prato de 26 cm de diâmetro).

O prato é encontrado com o YOLO-World, um modelo que deteta objetos a partir
de texto: pedimos-lhe "plate" (e "bowl", para não confundir taças com o prato).

Visto de cima, o prato é um círculo e a caixa à volta dele é um quadrado.
Numa foto tirada de lado, o prato parece uma oval achatada: a caixa fica
mais baixa do que larga. Por isso calculamos a escala separadamente na
horizontal e na vertical, o que corrige o efeito da inclinação da câmara
(assumindo que a câmara está inclinada para a frente, não rodada).

Histórico: a primeira versão procurava um círculo branco com OpenCV
(cor + Hough Circles). Falhava com pratos de cor (ex: bege), com fotos
tiradas de lado e quando havia uma taça ao lado do prato.
"""
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
        """
        Devolve (píxeis por cm na horizontal, píxeis por cm na vertical), ou
        None se não encontrar o prato. O chamador deve tratar o None (ex:
        pedir outra foto), nunca assumir um valor às cegas.
        """
        r = self.modelo.predict(imagem_bgr, conf=CONFIANCA_MINIMA, verbose=False)[0]
        pratos = [(float(conf), caixa.tolist()) for conf, caixa, classe
                  in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)
                  if self.modelo.names[int(classe)] == 'plate']
        if not pratos:
            return None

        _, (x1, y1, x2, y2) = max(pratos)  # o prato com mais confiança
        altura_img, largura_img = imagem_bgr.shape[:2]
        largura_px, altura_px = x2 - x1, y2 - y1

        cortado_h = x1 < MARGEM_BORDA_PX or x2 > largura_img - MARGEM_BORDA_PX
        cortado_v = y1 < MARGEM_BORDA_PX or y2 > altura_img - MARGEM_BORDA_PX

        if cortado_h and cortado_v:
            return None  # não conseguimos medir o prato em nenhuma direção
        if cortado_h:
            largura_px = altura_px  # usa a medida que está inteira (assume círculo)
        if cortado_v:
            altura_px = largura_px

        return largura_px / self.diametro_cm, altura_px / self.diametro_cm

    def pixels_para_cm2(self, area_pixeis: float, escala: tuple[float, float]) -> float:
        """Converte uma área em píxeis para cm², dada a escala (horizontal, vertical)."""
        px_por_cm_h, px_por_cm_v = escala
        return area_pixeis / (px_por_cm_h * px_por_cm_v)
