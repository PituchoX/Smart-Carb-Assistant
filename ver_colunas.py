from ultralytics import YOLO
modelo = YOLO('modelo_segmentacao_alimentos.pt')
print(modelo.names)