import os
import tempfile

import cv2
import streamlit as st

from nutrition.calculator import MARGEM_SEGURANCA, CalculadoraNutricional
from vision.detetor import SmartCarbDetector

st.set_page_config(page_title="Smart Carb Assistant", page_icon="🍽️", layout="wide")


# cache para não repetir a deteção sempre que o utilizador muda um campo
@st.cache_resource
def carregar_modelos():
    return SmartCarbDetector(), CalculadoraNutricional()


@st.cache_data(show_spinner=False)
def detetar(imagem_bytes: bytes, diametro_prato: float):
    detetor, _ = carregar_modelos()
    detetor.calibrador.diametro_cm = diametro_prato

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(imagem_bytes)
        caminho_tmp = tmp.name
    try:
        deteções = detetor.process_meal(caminho_tmp, salvar_anotada=True)
    finally:
        os.remove(caminho_tmp)

    imagem_anotada = cv2.cvtColor(cv2.imread("resultado_detecao.jpg"), cv2.COLOR_BGR2RGB)
    return deteções, imagem_anotada


def bolinha(numero, cor):
    """Círculo com o número do alimento, da mesma cor que na foto."""
    return f'<div class="bolinha" style="background:{cor}">{numero}</div>'


def passo(numero, icone, titulo, texto):
    return (f'<div class="passo"><div class="icone">{icone}</div><div class="numero">PASSO {numero}</div>'
            f'<h4>{titulo}</h4><p>{texto}</p></div>')


with open(os.path.join(os.path.dirname(__file__), "estilo.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Cabeçalho e upload ----------
st.markdown("""
<div class="hero">
  <h1>🍽️ Smart Carb Assistant</h1>
  <p>Tira uma foto ao prato e fica a saber quantos hidratos de carbono tem.</p>
  <span class="chip">📷 Deteção dos alimentos</span>
  <span class="chip">⚖️ Peso estimado</span>
  <span class="chip">📋 Tabela da APDP</span>
</div>
""", unsafe_allow_html=True)

col_titulo, col_def = st.columns([4, 1], vertical_alignment="bottom")
col_titulo.markdown("#### 📷 Carrega a foto do teu prato")
with col_def.popover("⚙️ Definições", width="stretch"):
    diametro_prato = st.slider("Diâmetro do prato (cm)", 20.0, 32.0, 26.0, 0.5,
                               help="O prato serve de régua para medir a comida.")
foto = st.file_uploader("Foto do prato", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if foto is None:
    st.markdown("#### Como funciona")
    c1, c2, c3 = st.columns(3)
    c1.markdown(passo(1, "📸", "Tira a foto", "De cima ou pouco inclinada, com o prato inteiro e boa luz."),
                unsafe_allow_html=True)
    c2.markdown(passo(2, "🔍", "A app analisa", "Deteta cada alimento, mede a área e estima o peso."),
                unsafe_allow_html=True)
    c3.markdown(passo(3, "✅", "Confirma e conta", "Corrige o que estiver errado e vê o total de hidratos."),
                unsafe_allow_html=True)
    st.stop()

with st.spinner("A analisar o prato..."):
    try:
        deteções, imagem_anotada = detetar(foto.getvalue(), diametro_prato)
    except ValueError as e:
        st.error(str(e))
        st.stop()

if not deteções:
    st.warning("Não foi detetado nenhum alimento na foto.")
    st.stop()

# ---------- Resultado ----------
_, calculadora = carregar_modelos()
tabela = calculadora.tabela_completa
NAO_CONTAR = -1
opcoes = [NAO_CONTAR] + list(tabela.index)


def nome_opcao(i):
    return "— não contar —" if i == NAO_CONTAR else tabela.at[i, 'alimento']


col_foto, col_lista = st.columns([1, 1.1], gap="large")
col_foto.image(imagem_anotada, width='stretch')

with col_lista:
    caixa_total = st.empty()  # preenchida no fim, depois de somar os alimentos

    # Alimentos com hidratos primeiro; os que contam 0 g ficam numa secção recolhida
    com_hc, sem_hc = [], []
    for n, det in enumerate(deteções, start=1):
        linha = calculadora.tabela.get(det['detetado'])
        sugestao = NAO_CONTAR if linha is None else int(linha.name)
        if linha is not None and linha['hc_por_100g'] == 0:
            sem_hc.append((n, det, sugestao))
        else:
            com_hc.append((n, det, sugestao))

    total_hc = 0.0
    por_escolher = []

    LARGURAS = [0.3, 3, 0.9, 0.7]

    def cabecalho():
        _, c_alimento, c_peso, c_hc = st.columns(LARGURAS)
        c_alimento.caption("Alimento")
        c_peso.caption("Peso (g)")
        c_hc.caption("Hidratos")

    def mostrar_alimento(n, det, sugestao):
        global total_hc
        c_num, c_alimento, c_peso, c_hc = st.columns(LARGURAS, vertical_alignment="center")
        c_num.markdown(bolinha(n, det['cor']), unsafe_allow_html=True)
        escolha = c_alimento.selectbox(
            "Alimento", opcoes, index=opcoes.index(sugestao), format_func=nome_opcao,
            key=f"alimento_{n}", label_visibility="collapsed",
            help=f"O modelo detetou: {det['detetado']} ({det['confianca']:.0%} de confiança)",
        )
        if escolha == NAO_CONTAR:
            c_hc.markdown("—")
            if sugestao == NAO_CONTAR:
                por_escolher.append(n)
            return

        linha = tabela.loc[escolha]
        peso_estimado = calculadora.estimar_peso(det['area_cm2'], linha['categoria'])
        # a key muda com o alimento, para o peso ser estimado de novo
        peso = c_peso.number_input("Peso (g)", min_value=0, value=int(round(peso_estimado)), step=5,
                                   key=f"peso_{n}_{escolha}", label_visibility="collapsed")
        hc = peso * linha['hc_por_100g'] / 100
        total_hc += hc
        c_hc.markdown(f"**{hc:.0f} g**")

    st.markdown("##### Alimentos com hidratos")
    st.caption("Corrige o alimento ou o peso se não estiverem certos.")
    cabecalho()
    for item in com_hc:
        mostrar_alimento(*item)

    if sem_hc:
        with st.expander(f"Sem hidratos ({len(sem_hc)}) — carne, peixe, ovo, legumes"):
            cabecalho()
            for item in sem_hc:
                mostrar_alimento(*item)

    if por_escolher:
        st.warning("Escolhe o alimento nº " + ", ".join(map(str, por_escolher))
                   + " na lista, ou conta-o à parte: não está incluído no total.")

    caixa_total.markdown(f"""
    <div class="total">
      <div class="rotulo">Hidratos de carbono no prato</div>
      <div class="valor">{total_hc:.0f} g</div>
      <div class="intervalo">Intervalo provável: {total_hc * (1 - MARGEM_SEGURANCA):.0f} – {total_hc * (1 + MARGEM_SEGURANCA):.0f} g</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("⚠️ Estimativa a partir de uma foto: o peso pode ter erros de dezenas de gramas. "
           "Não substitui a pesagem nem a orientação da tua equipa de saúde.")
