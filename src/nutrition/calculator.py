"""
Módulo de Domínio Nutricional:
Lê o Parquet limpo e calcula a estimativa de hidratos de carbono 
com margem de incerteza (mínimo, esperado, máximo) usando dicionários simples.
"""
import pandas as pd

# Mapeamento simples: Label da Visão -> Registo no INSA
MAPEAMENTO_ALIMENTOS = {
    "rice": {
        "id_insa": 403,  # Arroz cozido simples
        "peso_medio_g": 150.0,
        "margem_erro": 0.15,  # +-15%
    },
    "apple": {
        "id_insa": 40200010,  # Maçã (média com e sem casca)
        "peso_medio_g": 120.0,
        "margem_erro": 0.10,  # +-10%
    },
    "potato": {
        "id_insa": 583,  # Batata cozida
        "peso_medio_g": 130.0,
        "margem_erro": 0.15,
    },
}


class CalculadoraNutricional:
    def __init__(self, caminho_parquet: str):
        # Carrega a base tratada e indexa pelo ID para pesquisa direta
        self.df = pd.read_parquet(caminho_parquet)
        self.df_por_id = self.df.set_index("id")

    def calcular_prato(self, classes_detetadas: list) -> dict:
        """
        Recebe uma lista de classes (ex: ['rice', 'apple'])
        e devolve um dicionário simples com os cálculos.
        """
        itens_calculados = []
        total_esperado = 0.0
        total_min = 0.0
        total_max = 0.0

        for classe in classes_detetadas:
            if classe not in MAPEAMENTO_ALIMENTOS:
                continue

            config = MAPEAMENTO_ALIMENTOS[classe]
            id_alimento = config["id_insa"]

            if id_alimento not in self.df_por_id.index:
                continue

            linha = self.df_por_id.loc[id_alimento]
            if isinstance(linha, pd.DataFrame):
                linha = linha.iloc[0]

            peso = config["peso_medio_g"]
            margem = config["margem_erro"]
            hc_100g = float(linha["hidratos_100g"])

            # Cálculo de hidratos
            hc_esp = (peso * hc_100g) / 100.0
            hc_min = hc_esp * (1.0 - margem)
            hc_max = hc_esp * (1.0 + margem)

            itens_calculados.append({
                "alimento": str(linha["nome"]),
                "peso_estimado_g": peso,
                "hc_esperado_g": round(hc_esp, 1),
                "hc_min_g": round(hc_min, 1),
                "hc_max_g": round(hc_max, 1),
            })

            total_esperado += hc_esp
            total_min += hc_min
            total_max += hc_max

        return {
            "itens": itens_calculados,
            "total_esperado_g": round(total_esperado, 1),
            "intervalo_seguro_g": [round(total_min, 1), round(total_max, 1)],
        }