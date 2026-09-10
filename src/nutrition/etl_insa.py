from pathlib import Path
import pandas as pd


def executar_etl(caminho_excel: str, caminho_saida_parquet: str):
    print(f"1. [Extract] A ler ficheiro Excel: {caminho_excel}...")
    # O cabeçalho útil do INSA está na linha de índice 1
    df_raw = pd.read_excel(caminho_excel, sheet_name=0, header=1)

    print("2. [Transform] A limpar colunas e tipos de dados...")
    # Limpar quebras de linha '\n' nos cabeçalhos
    df_raw.columns = [str(col).replace("\n", " ").strip() for col in df_raw.columns]

    # Localizar colunas
    col_nome = [c for c in df_raw.columns if "nome do alimento" in c.lower()][0]
    col_hidratos = [c for c in df_raw.columns if "hidratos de carbono" in c.lower()][0]
    col_fibra = [c for c in df_raw.columns if "fibra" in c.lower()][0]
    col_energia = [c for c in df_raw.columns if "energia" in c.lower() and "kcal" in c.lower()][0]

    # Manter apenas o essencial
    df_clean = df_raw[["Cod", col_nome, col_hidratos, col_fibra, col_energia]].copy()
    df_clean.columns = ["id", "nome", "hidratos_100g", "fibra_100g", "kcal_100g"]

    # Converter texto, vírgulas e 'Tr' (traços) em números decimais reais
    def sanitizar_numeros(coluna):
        return (
            coluna.astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
            .replace({"Tr": "0.0", "nan": "0.0", "": "0.0", "None": "0.0"})
            .astype(float)
        )

    for col in ["hidratos_100g", "fibra_100g", "kcal_100g"]:
        df_clean[col] = sanitizar_numeros(df_clean[col])

    df_clean = df_clean.dropna(subset=["nome"])

    print(f"3. [Load] A gravar {len(df_clean)} alimentos em Parquet...")
    Path(caminho_saida_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(caminho_saida_parquet, index=False)
    print(f"✓ Base tratada com sucesso em: {caminho_saida_parquet}")


if __name__ == "__main__":
    executar_etl(
        caminho_excel="data/raw/tca_insa.xlsx",
        caminho_saida_parquet="data/processed/tca_clean.parquet",
    )