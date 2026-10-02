"""
Calibração de escala para o Smart Carb Assistant — via deteção do prato.

O modelo_segmentacao_alimentos.pt não tem uma classe "prato"/"plate" entre
as suas 103 classes (confirmado a correr modelo.names) — por isso usamos
visão computacional clássica para encontrar o contorno do prato e medir o
seu diâmetro em píxeis, sem precisar de nenhum marcador externo.

Histórico de decisões (para quem for mexer nisto depois):
- 1ª tentativa: Hough Circles direto na imagem em cinzento. Falhou em
  ambas as fotos de teste reais — a textura da mesa em madeira gera
  demasiadas bordas falsas e o círculo ajustado ficava maior do que a
  própria foto.
- 2ª tentativa: segmentar o prato por brilho (branco vs. resto) antes de
  procurar círculos. Funciona muito melhor, mas só com um cuidado extra:
  um guardanapo colorido claro na mesa também passa no filtro de brilho
  e "cola-se" ao prato no contorno. Resolvido filtrando também por baixa
  saturação em HSV (o prato é branco = saturação quase 0; o guardanapo,
  mesmo que claro, tem cor = saturação alta).
- A validação de "o círculo não pode ser maior do que a foto" mostrou-se
  boa a mais: quando o prato ocupa quase o enquadramento todo, o ajuste
  pode legitimamente ultrapassar a foto por uma margem pequena. Trocada
  por uma validação mais direta: confirmar que o contorno do círculo
  encontrado cai mesmo em cima de pixels do prato (não só que "cabe" na
  foto). Testado nas duas fotos reais: 76% de cobertura na foto boa,
  32% na má — o threshold de 55% abaixo separa bem os dois casos.

Limitação conhecida: assume um prato branco/claro (o filtro de saturação
não vai funcionar bem com pratos coloridos ou padronizados).
"""
import cv2
import numpy as np

DIAMETRO_PRATO_CM_DEFAULT = 26.0  # prato "médio" comum, ajustável
COBERTURA_MINIMA = 0.55  # ver histórico de decisões acima


class CalibradorPrato:
    TENTATIVAS_PARAM2 = [50, 40, 30, 20, 15]

    def __init__(self, diametro_prato_cm: float = DIAMETRO_PRATO_CM_DEFAULT):
        self.diametro_cm = diametro_prato_cm

    def calcular_pixels_por_cm(self, imagem_bgr: np.ndarray) -> float | None:
        """
        Deteta o prato e devolve a razão pixels/cm. Devolve None se não
        conseguir uma deteção suficientemente confiante — o chamador deve
        tratar este caso (ex: pedir nova foto), nunca assumir um valor
        às cegas.
        """
        mascara = self._segmentar_prato(imagem_bgr)
        altura, largura = mascara.shape
        raio_min = int(min(altura, largura) * 0.2)
        raio_max = int(min(altura, largura) * 0.6)

        circulos = None
        for param2 in self.TENTATIVAS_PARAM2:
            circulos = cv2.HoughCircles(
                mascara, cv2.HOUGH_GRADIENT, dp=1.5, minDist=largura,
                param1=50, param2=param2, minRadius=raio_min, maxRadius=raio_max,
            )
            if circulos is not None:
                break

        if circulos is None:
            return None

        cx, cy, raio_px = (float(v) for v in circulos[0, 0])
        if self._cobertura_do_circulo(mascara, cx, cy, raio_px) < COBERTURA_MINIMA:
            return None

        return (raio_px * 2) / self.diametro_cm

    def _segmentar_prato(self, imagem_bgr: np.ndarray) -> np.ndarray:
        """Isola o prato (claro, baixa saturação) do resto da foto."""
        hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
        kernel = np.ones((9, 9), np.uint8)
        return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    def _cobertura_do_circulo(self, mascara: np.ndarray, cx: float, cy: float,
                               raio: float, n_pontos: int = 72) -> float:
        """
        Confirma que o círculo encontrado corresponde mesmo ao prato:
        testa vários raios entre 85% e 100% do raio ajustado (tolerante a
        comida que toque no rebordo) e devolve a melhor cobertura obtida.
        """
        altura, largura = mascara.shape
        melhor_cobertura = 0.0
        for fator in (0.85, 0.90, 0.95, 1.0):
            r = raio * fator
            acertos = 0
            for i in range(n_pontos):
                ang = 2 * np.pi * i / n_pontos
                x, y = int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))
                if 0 <= x < largura and 0 <= y < altura and mascara[y, x] > 0:
                    acertos += 1
            melhor_cobertura = max(melhor_cobertura, acertos / n_pontos)
        return melhor_cobertura

    def pixels_para_cm2(self, area_pixeis: int, pixels_por_cm: float) -> float:
        """Converte uma área em píxeis para cm², dada a razão de calibração."""
        return area_pixeis / (pixels_por_cm ** 2)