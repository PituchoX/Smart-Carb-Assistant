import sqlite3
import os
import numpy as np
from ultralytics import YOLO

# 1. Fator de calibração e Dicionário Físico/Semântico
# Baixámos o fator de escala porque a foto foi tirada bastante perto do prato
FATOR_ESCALA = 0.003 

# Ajustámos as alturas para valores mais realistas e as strings do INSA para pesquisa exata
ATRIBUTOS_COMIDA = {
    'rice': {
        'nome_insa': 'Arroz, branqueado, cozido', 
        'altura_cm': 1.5, 
        'densidade': 1.12
    },
    'egg': {
        'nome_insa': 'Ovo, de galinha, inteiro', 
        'altura_cm': 0.8, 
        'densidade': 1.03
    },
    'french fries': {
        'nome_insa': 'Batata, frita', 
        'altura_cm': 1.5, 
        'densidade': 0.60
    },
    'steak': {
        'nome_insa': 'Vaca, febra, grelhada', 
        'altura_cm': 1.0, 
        'densidade': 1.05
    },
    'cilantro mint': {
        'nome_insa': 'Alface', 
        'altura_cm': 0.5, 
        'densidade': 0.30
    }
}

# 2. Carregar o Modelo e Processar a Imagem
modelo = YOLO('modelo_segmentacao_alimentos.pt')
resultados = modelo('teste.jpg')

# 3. Ligar à Base de Dados SQLite Local
caminho_db = os.path.join('dados', 'smart_carb.db')
conn = sqlite3.connect(caminho_db)
cursor = conn.cursor()

print("\n--- Resultados Nutricionais Estimados ---")

for r in resultados:
    if r.masks is not None:
        for i, mascara in enumerate(r.masks.data):
            nome_ingles = modelo.names[int(r.boxes.cls[i])]
            area_pixeis = np.count_nonzero(mascara.cpu().numpy())
            
            # Se o alimento detetado estiver no nosso dicionário
            if nome_ingles in ATRIBUTOS_COMIDA:
                alimento = ATRIBUTOS_COMIDA[nome_ingles]
                
                # Matemática para estimativa de peso
                area_cm2 = area_pixeis * FATOR_ESCALA
                volume_cm3 = area_cm2 * alimento['altura_cm']
                peso_estimado_g = volume_cm3 * alimento['densidade']
                
                try:
                    # Query SQL atualizada com os nomes exatos das colunas
                    cursor.execute(
                        "SELECT hidratos_carbono_g FROM alimentos WHERE nome LIKE ? LIMIT 1", 
                        ('%' + alimento['nome_insa'] + '%',)
                    )
                    row = cursor.fetchone()
                    
                    if row and row[0]:
                        # Converte a vírgula do INSA para ponto, caso exista
                        hc_por_100g = float(str(row[0]).replace(',', '.'))
                        hc_total = (peso_estimado_g / 100) * hc_por_100g
                        
                        print(f"🍽️ {nome_ingles.capitalize():<15} | Peso: {peso_estimado_g:>6.1f}g | Hidratos: {hc_total:>5.1f}g")
                    else:
                        print(f"🍽️ {nome_ingles.capitalize():<15} | Peso: {peso_estimado_g:>6.1f}g | Hidratos: N/A no INSA")
                except Exception as e:
                    print(f"Erro na query SQL: {e}")
                    
conn.close()