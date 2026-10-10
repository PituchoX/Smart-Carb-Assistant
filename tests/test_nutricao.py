import os

from nutrition.calculator import CalculadoraNutricional
from scripts.etl_apdp import CAMINHO_PDF, extrair_alimentos

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, 'modelo_segmentacao_alimentos.pt')


def test_extracao_do_pdf():
    df = extrair_alimentos(CAMINHO_PDF)
    assert len(df) == 164

    hc = dict(zip(df['alimento'], df['hc_por_100g']))
    assert hc['Arroz solto cozido'] == 28.0
    assert hc['Farinha de trigo tipo 55'] == 74.3   # o 55 é parte do nome
    assert hc['Banana'] == 21.8                     # fruta: valor edível


def test_classes_do_excel_existem_no_modelo():
    from ultralytics import YOLO
    classes_modelo = set(YOLO(CAMINHO_MODELO).names.values())
    classes_excel = set(CalculadoraNutricional().tabela)

    # Nenhuma classe no Excel com erro de escrita (ex: 'ryce')
    assert classes_excel <= classes_modelo


def test_calcular_prato():
    calc = CalculadoraNutricional()
    resultado = calc.calcular_prato([
        {'detetado': 'rice', 'peso_g': 150},   # 150g x 28g/100g = 42g
        {'detetado': 'steak', 'peso_g': 200},  # carne: 0g
        {'detetado': 'cake', 'peso_g': 80},    # sem linha no Excel
    ])
    assert resultado['total_g'] == 42.0
    assert resultado['sem_correspondencia'] == ['cake']
