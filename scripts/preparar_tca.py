import pandas as pd
import sqlite3

print("A ler o ficheiro Excel do INSA...")
# O header=1 diz ao Pandas para ignorar a primeira linha e usar a segunda como nome das colunas
df = pd.read_excel('insa_tca.xlsx', sheet_name='INSA - BDCA_v 7.1 - 2026', header=1) 

# Os nomes EXATOS das colunas no ficheiro, incluindo espaços e quebras de linha (\n)
colunas_interesse = [
    'Cod', 
    'Nome do alimento', 
    'Energia\n[kcal] ', 
    'Hidratos de carbono \n[g]'
]

print("A limpar e organizar os dados...")
df_clean = df[colunas_interesse].copy()

# Renomear as colunas para nomes simples de usar na base de dados
df_clean.rename(columns={
    'Cod': 'codigo',
    'Nome do alimento': 'nome',
    'Energia\n[kcal] ': 'energia_kcal',
    'Hidratos de carbono \n[g]': 'hidratos_carbono_g'
}, inplace=True)

# Remover linhas vazias e uniformizar o texto (tudo em minúsculas)
df_clean = df_clean.dropna(subset=['nome'])
df_clean['nome'] = df_clean['nome'].str.strip().str.lower()

# Garantir que os hidratos de carbono são números
df_clean['hidratos_carbono_g'] = pd.to_numeric(df_clean['hidratos_carbono_g'], errors='coerce').fillna(0)

print("A guardar na base de dados SQLite...")
conn = sqlite3.connect('smart_carb.db')
df_clean.to_sql('alimentos', conn, if_exists='replace', index=False)
conn.close()

print("Sucesso! O ficheiro 'smart_carb.db' foi criado e está pronto.")