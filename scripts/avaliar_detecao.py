
import os
import sys

from ultralytics import YOLO, YOLOE

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_YAML = os.path.join(PASTA_PROJETO, 'dados', 'foodseg103', 'data.yaml')

MODELOS = {
    'original': os.path.join(PASTA_PROJETO, 'modelo_segmentacao_alimentos.pt'),  # YOLOv8n de gorkatorres
    'yolov8s': os.path.join(PASTA_PROJETO, 'modelos', 'yolov8s_foodseg103.pt'),   # treinado por nós
    'yoloe-11s': os.path.join(PASTA_PROJETO, 'modelos', 'yoloe-11s-seg.pt'),
    'yoloe-11l': os.path.join(PASTA_PROJETO, 'modelos', 'yoloe-11l-seg.pt'),
}


def carregar(nome):
    if nome.endswith('.pt'):
        return YOLO(nome)  # um ficheiro de modelo qualquer (ex: um modelo treinado por nós)
    if not nome.startswith('yoloe'):
        return YOLO(MODELOS[nome])
    # O YOLOE recebe os nomes dos alimentos em texto: usamos as mesmas 103 classes
    nomes = list(YOLO(MODELOS['original']).names.values())
    modelo = YOLOE(MODELOS[nome])
    modelo.set_classes(nomes, modelo.get_text_pe(nomes))
    return modelo


if __name__ == '__main__':
    resultados = {}
    for nome in sys.argv[1:]:
        print(f'=== {nome}')
        metricas = carregar(nome).val(data=DATA_YAML, split='test', imgsz=640, batch=8,
                                      conf=0.001, plots=False, verbose=False)
        resultados[nome] = (metricas.seg.map50, metricas.seg.map)

    print('\nModelo        mAP50 (máscaras)   mAP50-95')
    for nome, (map50, map_) in resultados.items():
        print(f'{nome:<13} {map50:>10.3f}        {map_:.3f}')
