"""
Deteção de alimentos e estimativa de peso a partir de uma foto do prato.

A escala pixels/cm é calculada por foto a partir do diâmetro do prato
detetado (ver calibration.py) — já não usa nenhum fator fixo.
"""
import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import CalibradorPrato

# CATEGORIAS: altura/densidade típicas por tipo físico de alimento.
# Acrescentar um alimento novo normalmente só precisa de escolher uma
# categoria já existente — só crias uma categoria nova quando a forma ou
# densidade for claramente diferente de todas as que já existem.
CATEGORIAS = {
    'grao_cozido':  {'altura_cm': 1.5, 'densidade': 1.10},  # arroz, massa
    'fritos_amido': {'altura_cm': 1.5, 'densidade': 0.60},  # batata frita
    'folhas':       {'altura_cm': 0.5, 'densidade': 0.30},  # alface, couve
    'legume_polpa': {'altura_cm': 1.0, 'densidade': 0.55},  # tomate, cebola
    'proteina':       {'altura_cm': 1.5, 'densidade': 1.00},  # carne/peixe
    'proteina_fina':  {'altura_cm': 0.8, 'densidade': 1.00},  # ovo estrelado/mexido
    'pao':          {'altura_cm': 3.0, 'densidade': 0.25},  # estrutura porosa
}

# nome_insa tem de ser substring literal de um nome real na tabela
# (dados/smart_carb.db, coluna `nome`, tudo em minúsculas, sem vírgulas a
# separar termos). Testar sempre contra a base de dados real antes de
# adicionar — já apanhámos nomes inventados que nunca davam match.
#
# Para 'proteina', a precisão da altura/densidade não é prioritária: HC
# é ≈0g em quase toda a proteína pura, por isso o peso estimado não muda
# o resultado final de forma relevante.
ATRIBUTOS_COMIDA = {
    'rice':          {'categoria': 'grao_cozido',  'nome_insa': 'arroz cozido simples'},
    'pasta':         {'categoria': 'grao_cozido',  'nome_insa': 'esparguete cozido'},
    'french fries':  {'categoria': 'fritos_amido', 'nome_insa': 'batata frita caseira'},
    'lettuce':       {'categoria': 'folhas',       'nome_insa': 'alface'},
    'tomato':        {'categoria': 'legume_polpa', 'nome_insa': 'tomate cru'},
    'onion':         {'categoria': 'legume_polpa', 'nome_insa': 'cebola crua'},
    'egg':           {'categoria': 'proteina_fina', 'nome_insa': 'ovo de galinha inteiro'},
    'steak':         {'categoria': 'proteina',     'nome_insa': 'vaca, bife grelhado'},
    'bread':         {'categoria': 'pao',          'nome_insa': 'pão de trigo'},
}


class SmartCarbDetector:
    def __init__(self, caminho_modelo: str = 'modelo_segmentacao_alimentos.pt',
                 diametro_prato_cm: float = 26.0):
        self.modelo = YOLO(caminho_modelo)
        self.calibrador = CalibradorPrato(diametro_prato_cm)

    def process_meal(self, caminho_imagem: str) -> list[dict]:
        """
        Corre a deteção/segmentação na imagem e devolve uma lista de
        deteções — uma por alimento reconhecido. Alimentos detetados que
        não constam de ATRIBUTOS_COMIDA são ignorados (silenciosamente,
        por agora).

        Levanta ValueError se não conseguir detetar o prato na foto —
        sem escala fiável, não faz sentido devolver pesos calculados às
        cegas.
        """
        imagem_bgr = cv2.imread(caminho_imagem)
        if imagem_bgr is None:
            raise FileNotFoundError(f"Não consegui abrir a imagem: {caminho_imagem}")

        pixels_por_cm = self.calibrador.calcular_pixels_por_cm(imagem_bgr)
        if pixels_por_cm is None:
            raise ValueError(
                "Não foi possível detetar o prato nesta foto — tenta uma "
                "foto com o prato mais visível, centrado e bem iluminado."
            )

        resultados = self.modelo(caminho_imagem)
        deteções = []

        for r in resultados:
            if r.masks is None:
                continue
            for i, poligono in enumerate(r.masks.xy):
                nome_ingles = self.modelo.names[int(r.boxes.cls[i])]
                if nome_ingles not in ATRIBUTOS_COMIDA:
                    continue

                atributos = ATRIBUTOS_COMIDA[nome_ingles]
                # r.masks.xy já vem em coordenadas da imagem ORIGINAL (o
                # ultralytics trata da conversão internamente) — ao
                # contrário de r.masks.data, que vem na resolução interna
                # usada pelo modelo e não bate certo com a escala pixels/cm
                # calculada a partir da foto original.
                area_pixeis = cv2.contourArea(poligono.astype(np.float32))
                area_cm2 = self.calibrador.pixels_para_cm2(area_pixeis, pixels_por_cm)
                peso_g = self._estimar_peso(area_cm2, atributos)

                deteções.append({
                    'detetado': nome_ingles,
                    'nome_insa': atributos['nome_insa'],
                    'peso_g': peso_g,
                })

        return deteções

    def _estimar_peso(self, area_cm2: float, atributos: dict) -> float:
        propriedades = CATEGORIAS[atributos['categoria']]
        volume_cm3 = area_cm2 * propriedades['altura_cm']
        return volume_cm3 * propriedades['densidade']