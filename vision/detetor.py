"""
Deteção de alimentos e estimativa de peso a partir de uma foto do prato.
Suporta todas as classes do modelo FoodSeg103 sem dicionários rígidos manuais.
"""
import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import CalibradorPrato

# Propriedades físicas por tipo morfológico de alimento
CATEGORIAS_FISICAS = {
    'grao_cozido':  {'altura_cm': 2.2, 'densidade': 1.10},  # arroz, massa
    'fritos_amido': {'altura_cm': 1.8, 'densidade': 0.65},  # batatas fritas
    'folhas':       {'altura_cm': 0.8, 'densidade': 0.30},  # salada, couve
    'legume_polpa': {'altura_cm': 1.2, 'densidade': 0.60},  # tomate, cebola
    'proteina':     {'altura_cm': 1.5, 'densidade': 1.00},  # carnes, peixes
    'proteina_fina':{'altura_cm': 0.8, 'densidade': 1.00},  # ovo
    'pao':          {'altura_cm': 3.0, 'densidade': 0.25},  # pão
    'padrao':       {'altura_cm': 1.5, 'densidade': 0.85},  # valor default para novas classes
}

# Mapeamento rápido de tipo físico para inferir volume
MAPA_CATEGORIAS = {
    'rice': 'grao_cozido', 'pasta': 'grao_cozido', 'noodles': 'grao_cozido', 'corn': 'grao_cozido',
    'french fries': 'fritos_amido', 'potato': 'fritos_amido',
    'lettuce': 'folhas', 'cabbage': 'folhas', 'spinach': 'folhas', 'salad': 'folhas',
    'tomato': 'legume_polpa', 'onion': 'legume_polpa', 'cucumber': 'legume_polpa', 'carrot': 'legume_polpa',
    'egg': 'proteina_fina',
    'bread': 'pao',
    'steak': 'proteina', 'pork': 'proteina', 'chicken duck': 'proteina', 'fish': 'proteina',
    'sausage': 'proteina', 'fried meat': 'proteina', 'shrimp': 'proteina'
}


class SmartCarbDetector:
    def __init__(self, caminho_modelo: str = 'modelo_segmentacao_alimentos.pt',
                 diametro_prato_cm: float = 26.0):
        self.modelo = YOLO(caminho_modelo)
        self.calibrador = CalibradorPrato(diametro_prato_cm)

    def process_meal(self, caminho_imagem: str, salvar_anotada: bool = True, conf: float = 0.20) -> list[dict]:
        imagem_bgr = cv2.imread(caminho_imagem)
        if imagem_bgr is None:
            raise FileNotFoundError(f"Não consegui abrir a imagem: {caminho_imagem}")

        pixels_por_cm = self.calibrador.calcular_pixels_por_cm(imagem_bgr)
        if pixels_por_cm is None:
            raise ValueError(
                "Não foi possível detetar o prato nesta foto — tenta uma foto com o prato mais visível e centrado."
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

                # Determinar propriedades morfológicas (ou padrão genérico)
                tipo_fisico = MAPA_CATEGORIAS.get(nome_ingles, 'padrao')
                propriedades = CATEGORIAS_FISICAS[tipo_fisico]

                pts = poligono.astype(np.int32)
                area_pixeis = cv2.contourArea(poligono.astype(np.float32))
                area_cm2 = self.calibrador.pixels_para_cm2(area_pixeis, pixels_por_cm)
                
                # Estimar peso
                volume_cm3 = area_cm2 * propriedades['altura_cm']
                peso_g = volume_cm3 * propriedades['densidade']

                deteções.append({
                    'detetado': nome_ingles,
                    'peso_g': peso_g,
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
                        f"{nome_ingles}: {peso_g:.0f}g",
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        if salvar_anotada:
            cv2.imwrite("resultado_detecao.jpg", imagem_anotada)

        return deteções