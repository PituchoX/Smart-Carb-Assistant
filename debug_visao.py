"""
Script de diagnóstico: desenha as máscaras detetadas por cima da foto e
imprime a área de cada deteção. Usar sempre que um peso estimado parecer
implausível, para perceber se o problema é na deteção (área errada) ou
noutro sítio (altura/densidade assumidas, tabela INSA, etc.).
"""
import cv2
from ultralytics import YOLO

CAMINHO_MODELO = 'modelo_segmentacao_alimentos.pt'
CAMINHO_IMAGEM = 'prato.jpg'

modelo = YOLO(CAMINHO_MODELO)
resultados = modelo(CAMINHO_IMAGEM)

for r in resultados:
    if r.masks is None:
        print("Nenhuma deteção nesta imagem.")
        continue

    # Desenha as máscaras/caixas detetadas por cima da foto original
    imagem_anotada = r.plot()
    cv2.imwrite('debug_deteccoes.jpg', imagem_anotada)
    print("Guardado: debug_deteccoes.jpg — abre e confirma visualmente "
          "se a máscara de cada alimento cobre a área que esperas.")
    print()

    print(f"{'Alimento':<15} | {'Área (píxeis²)':>15}")
    print("-" * 35)
    for i, poligono in enumerate(r.masks.xy):
        nome = modelo.names[int(r.boxes.cls[i])]
        area = cv2.contourArea(poligono.astype('float32'))
        print(f"{nome:<15} | {area:>15.0f}")