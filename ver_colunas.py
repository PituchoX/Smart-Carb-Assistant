import sqlite3
import os

caminho_db = os.path.join('dados', 'smart_carb.db')
conn = sqlite3.connect(caminho_db)
cursor = conn.cursor()

# Extrair os nomes reais das colunas da tua tabela
cursor.execute("SELECT * FROM alimentos LIMIT 0")
colunas = [descricao[0] for descricao in cursor.description]

print("\nAs colunas exatas na tua base de dados são:")
for col in colunas:
    print(f"- {col}")

conn.close()