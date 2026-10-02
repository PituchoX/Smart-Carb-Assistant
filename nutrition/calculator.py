import os
import sqlite3

MARGEM_SEGURANCA = 0.15  # ±15%, margem de segurança clínica


class CalculadoraNutricional:
    def __init__(self, caminho_db: str = os.path.join('dados', 'smart_carb.db')):
        self.caminho_db = caminho_db

    def calcular_prato(self, deteções: list[dict]) -> dict:
        conn = sqlite3.connect(self.caminho_db)
        cursor = conn.cursor()

        itens = []
        total_g = 0.0

        for deteção in deteções:
            hc_por_100g = self._procurar_hidratos(cursor, deteção['nome_insa'])
            if hc_por_100g is None:
                continue

            hc_g = (deteção['peso_g'] / 100) * hc_por_100g
            margem = (hc_g * (1 - MARGEM_SEGURANCA), hc_g * (1 + MARGEM_SEGURANCA))

            itens.append({
                'detetado': deteção['detetado'],
                'nome_insa': deteção['nome_insa'],
                'peso_g': deteção['peso_g'],
                'hc_g': hc_g,
                'margem': margem,
            })
            total_g += hc_g

        conn.close()

        intervalo_clinico = (
            round(total_g * (1 - MARGEM_SEGURANCA), 1),
            round(total_g * (1 + MARGEM_SEGURANCA), 1),
        )

        return {
            'itens': itens,
            'total_g': round(total_g, 1),
            'intervalo_clinico': intervalo_clinico,
        }

    def _procurar_hidratos(self, cursor: sqlite3.Cursor, nome_insa: str) -> float | None:
        cursor.execute(
            "SELECT hidratos_carbono_g FROM alimentos WHERE nome LIKE ? LIMIT 1",
            ('%' + nome_insa + '%',)
        )
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(str(row[0]).replace(',', '.'))
        return None