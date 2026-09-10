from src.nutrition.calculator import CalculadoraNutricional

calc = CalculadoraNutricional("data/processed/tca_clean.parquet")
resultado = calc.calcular_prato(["rice", "apple"])

print("--- RESULTADO DO PRATO (Dicionário Simples) ---")
for item in resultado["itens"]:
    print(f"- {item['alimento']} (~{item['peso_estimado_g']}g):")
    print(f"  HC: {item['hc_esperado_g']}g [{item['hc_min_g']}g a {item['hc_max_g']}g]")

print(f"\nTotal Esperado : {resultado['total_esperado_g']}g")
print(f"Intervalo Seguro : {resultado['intervalo_seguro_g'][0]}g a {resultado['intervalo_seguro_g'][1]}g")