"""
Ponto de entrada: Smart Carb Assistant.
Executa deteção especializada em comida (YOLO-Seg + Food101) e mapeamento vetorial INSA.
"""
from src.vision.detector import SmartCarbDetector
from src.nutrition.calculator import CalculadoraNutricional

def main():
    imagem = "prato.jpg"

    print("=" * 80)
    print("   SMART CARB ASSISTANT | COMPUTER VISION + INSA SEMANTIC MATCHING")
    print("=" * 80)

    # 1. Instanciar o novo detetor de 2 Fases (YOLO-Seg + ResNet)
    detetor = SmartCarbDetector()
    
    # 2. Processar a imagem com o novo método
    detecoes = detetor.process_meal(imagem)

    if not detecoes:
        print("Nenhum alimento identificado no prato.")
        return

    # 3. Calcular os valores nutricionais
    calculadora = CalculadoraNutricional()
    resultado = calculadora.calcular_prato(detecoes)

    print("\n--- ESTIMATIVA POR ALIMENTO (ÁREA VISUAL & TCA INSA) ---")
    cabecalho = f"{'Detetado':<16} | {'Registo INSA':<28} | {'Peso':<6} | {'HC (g)':<7} | {'Margem Segura'}"
    print(cabecalho)
    print("-" * 85)

    for item in resultado["itens"]:
        det = item["detetado"][:14]
        match = item["nome_insa"][:26]
        peso = f"{item['peso_g']:.0f}g"
        hc = f"{item['hc_g']:.1f}g"
        margem = f"[{item['margem'][0]:.1f}g - {item['margem'][1]:.1f}g]"
        print(f"{det:<16} | {match:<28} | {peso:<6} | {hc:<7} | {margem}")

    print("-" * 85)
    print(f"Total Estimado: {resultado['total_g']} g de Hidratos de Carbono")
    print(f"Intervalo Clínico: {resultado['intervalo_clinico']} g")
    print("=" * 85)

if __name__ == "__main__":
    main()