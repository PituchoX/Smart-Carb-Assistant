"""
Deteção de alimentos a partir de uma foto do prato: o que é cada alimento
e quanta área (cm²) ocupa. O peso é estimado depois, em nutrition/calculator.py.
"""
import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import CalibradorPrato


class SmartCarbDetector:
    def __init__(self, caminho_modelo: str = 'modelo_segmentacao_alimentos.pt',
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
                "Não foi possível medir o prato nesta foto — tenta uma foto com o prato inteiro visível."
            )

        resultados = self.modelo(caminho_imagem, conf=conf)
        deteções = []
        imagem_anotada = imagem_bgr.copy()

        for r in resultados:
            if r.masks is None:
                continue
            for i, poligono in enumerate(r.masks.xy):
                nome_ingles = self.modelo.names[int(r.boxes.cls[i])]
                
                # Ignorar prato/utensílios ou fundos se existirem
                if nome_ingles in ['plate', 'tray']:
                    continue

                pts = poligono.astype(np.int32)
                area_pixeis = cv2.contourArea(poligono.astype(np.float32))
                area_cm2 = self.calibrador.pixels_para_cm2(area_pixeis, escala)

                deteções.append({
                    'detetado': nome_ingles,
                    'area_cm2': area_cm2,
                })

                if salvar_anotada:
                    cv2.polylines(imagem_anotada, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    momento = cv2.moments(pts)
                    if momento["m00"] != 0:
                        cx = int(momento["m10"] / momento["m00"])
                        cy = int(momento["m01"] / momento["m00"])
                    else:
                        cx, cy = pts[0][0], pts[0][1]

                    cv2.putText(
                        imagem_anotada,
                        nome_ingles,
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        if salvar_anotada:
            cv2.imwrite("resultado_detecao.jpg", imagem_anotada)

        return deteções