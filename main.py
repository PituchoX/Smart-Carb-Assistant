from src.vision.detector import DetetorAlimentos
from src.nutrition.calculator import CalculadoraNutricional


def main():
    imagem = "maça.jpg"
    print("=" * 50)
    print("SMART CARB ASSISTANT: A PROCESSAR REFEIÇÃO")
    print("=" * 50)

    detetor = DetetorAlimentos()
    detecoes = detetor.detetar(imagem)

    if not detecoes:
        print("Nenhum alimento identificado na imagem.")
        return

    print(f"\nAlimentos detetados na imagem ({len(detecoes)} instâncias):")
    for item in detecoes:
        print(f" - {item['classe']} (Confiança: {item['confianca'] * 100:.1f}%)")

    classes = [item["classe"] for item in detecoes]

    calculadora = CalculadoraNutricional(caminho_parquet="data/processed/tca_clean.parquet")
    resultado = calculadora.calcular_prato(classes)

    print("\n" + "=" * 50)
    print("RESUMO NUTRICIONAL (BASE INSA)")
    print("=" * 50)
    print(resultado)

if __name__ == "__main__":
    main()