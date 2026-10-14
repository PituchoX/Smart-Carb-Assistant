import os

from nutrition.calculator import CalculadoraNutricional
from scripts.etl_apdp import CAMINHO_PDF, extrair_alimentos

PASTA_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, 'modelos', 'yolov8s_foodseg103.pt')


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
        {'detetado': 'rice', 'area_cm2': 120},
        {'detetado': 'steak', 'area_cm2': 100},  # carne: 0g de HC
        {'detetado': 'cake', 'area_cm2': 50},    # sem linha no Excel
    ])
    arroz, bife, bolo = resultado['itens']

    assert arroz['peso_g'] > 0
    assert arroz['hc_g'] == arroz['peso_g'] * 28 / 100   # Arroz solto cozido: 28g/100g
    assert bife['hc_g'] == 0
    assert bolo['hc_g'] is None                           # fica fora do total
    assert resultado['total_g'] == round(arroz['hc_g'], 1)
    assert resultado['sem_correspondencia'] == ['cake']


def test_mais_area_da_mais_peso():
    calc = CalculadoraNutricional()
    assert calc.estimar_peso(200, 'Batata') > calc.estimar_peso(50, 'Batata')
