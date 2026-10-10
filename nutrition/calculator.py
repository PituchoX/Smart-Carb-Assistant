import json
import os

import pandas as pd

PASTA_PROJETO = os.path.join(os.path.dirname(__file__), '..')
CAMINHO_EXCEL = os.path.join(PASTA_PROJETO, 'dados', 'tabela_alimentos.xlsx')
# Gramas por cm² de cada categoria de alimento, aprendidos com dados reais
# em notebooks/modelo_peso.ipynb
CAMINHO_MODELO_PESO = os.path.join(PASTA_PROJETO, 'modelos', 'gramas_por_cm2.json')
MARGEM_SEGURANCA = 0.15


class CalculadoraNutricional:
    def __init__(self, caminho_excel: str = CAMINHO_EXCEL):
        df = pd.read_excel(caminho_excel)
        self.tabela_completa = df  # usada na app para o utilizador escolher outro alimento
        with open(CAMINHO_MODELO_PESO, encoding='utf-8') as f:
            self.modelo_peso = json.load(f)

        # A coluna classe_modelo diz a que linha corresponde cada classe do
        # modelo de visão (ex: 'rice' -> "Arroz solto cozido"). Uma célula
        # pode ter várias classes separadas por vírgula (ex: "pasta, noodles").
        self.tabela = {}
        for _, linha in df.dropna(subset=['classe_modelo']).iterrows():
            for classe in str(linha['classe_modelo']).split(','):
                classe = classe.strip()
                if classe in self.tabela:
                    raise ValueError(f"A classe '{classe}' aparece em mais do que uma linha do Excel")
                self.tabela[classe] = linha

    def estimar_peso(self, area_cm2: float, categoria: str | None) -> float:
        # Categorias sem dados de treino (ou alimentos fora do Excel) usam o valor geral
        g_por_cm2 = self.modelo_peso['por_categoria'].get(categoria, self.modelo_peso['geral'])
        return area_cm2 * g_por_cm2

    def calcular_prato(self, deteções: list[dict]) -> dict:
        itens = []
        total_g = 0.0
        sem_correspondencia = []

        for det in deteções:
            classe = det['detetado']
            linha = self.tabela.get(classe)
            categoria = linha['categoria'] if linha is not None else None
            peso_g = self.estimar_peso(det['area_cm2'], categoria)

            if linha is None:
                # Não inventamos um valor: o alimento fica fora do total e o
                # utilizador é avisado para o contar à parte.
                sem_correspondencia.append(classe)
                itens.append({'detetado': classe, 'alimento': None, 'fonte': None,
                              'peso_g': peso_g, 'hc_g': None, 'margem': None})
                continue

            hc_g = (peso_g / 100.0) * linha['hc_por_100g']
            margem = (hc_g * (1 - MARGEM_SEGURANCA), hc_g * (1 + MARGEM_SEGURANCA))
            total_g += hc_g

            itens.append({
                'detetado': classe,
                'alimento': linha['alimento'],
                'fonte': linha['fonte'],
                'peso_g': peso_g,
                'hc_g': hc_g,
                'margem': margem,
            })

        intervalo_clinico = (
            round(total_g * (1 - MARGEM_SEGURANCA), 1),
            round(total_g * (1 + MARGEM_SEGURANCA), 1),
        )

        return {
            'itens': itens,
            'total_g': round(total_g, 1),
            'intervalo_clinico': intervalo_clinico,
            'sem_correspondencia': sem_correspondencia,
        }
