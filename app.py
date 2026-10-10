import os
import tempfile

import cv2
import streamlit as st

from nutrition.calculator import MARGEM_SEGURANCA, CalculadoraNutricional
from vision.detetor import SmartCarbDetector

st.set_page_config(
    page_title="Smart Carb Assistant",
    page_icon="🍽️",
    layout="wide"
)


# O Streamlit volta a correr este script sempre que o utilizador mexe num
# campo. Os modelos e a deteção ficam em cache para não repetir trabalho.
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


st.title("🍽️ Smart Carb Assistant")
st.markdown(
    "Tira uma foto ao prato: a app deteta os alimentos, estima o peso e calcula os hidratos de carbono "
    "com a tabela da APDP. **Confirma sempre os alimentos e os pesos antes de usar o valor.**"
)

with st.sidebar:
    st.header("⚙️ Configurações")
    diametro_prato = st.slider("Diâmetro do prato (cm):", min_value=20.0, max_value=32.0, value=26.0, step=0.5)
    st.info(
        "O prato é detetado automaticamente (YOLO-World) e o seu diâmetro serve de régua para "
        "converter píxeis em cm². Também corrige fotos tiradas de lado."
    )

uploaded_file = st.file_uploader("Carrega uma foto do teu prato (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("A detetar o prato e os alimentos..."):
        try:
            deteções, imagem_anotada = detetar(uploaded_file.getvalue(), diametro_prato)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Foto original")
        st.image(uploaded_file, width='stretch')
    with col2:
        st.subheader("🎯 Alimentos detetados")
        st.image(imagem_anotada, width='stretch')

    if not deteções:
        st.warning("Não foi detetado nenhum alimento na foto.")
        st.stop()

    _, calculadora = carregar_modelos()
    tabela = calculadora.tabela_completa

    # Opções da lista: "não contar" + todas as linhas do Excel (pelo número da linha)
    NAO_CONTAR = -1
    opcoes = [NAO_CONTAR] + list(tabela.index)

    def nome_opcao(i):
        if i == NAO_CONTAR:
            return "— não contar —"
        return f"{tabela.at[i, 'alimento']} ({tabela.at[i, 'categoria']})"

    st.markdown("---")
    st.subheader("✏️ Confirma os alimentos")
    st.caption(
        "Corrige o alimento se a deteção estiver errada (ex: escolher \"Batata assada\" em vez de "
        "\"Batata cozida\") e ajusta o peso se te parecer diferente. Escolhe \"não contar\" para ignorar uma deteção."
    )

    total_hc = 0.0
    por_escolher = []

    for n, det in enumerate(deteções):
        classe = det['detetado']
        linha_sugerida = calculadora.tabela.get(classe)
        sugestao = NAO_CONTAR if linha_sugerida is None else int(linha_sugerida.name)

        c_det, c_alimento, c_peso, c_hc = st.columns([1.2, 3, 1.2, 1])
        c_det.markdown(f"**{classe}**")

        escolha = c_alimento.selectbox(
            "Alimento", opcoes, index=opcoes.index(sugestao), format_func=nome_opcao,
            key=f"alimento_{n}", label_visibility="collapsed",
        )
        if escolha == NAO_CONTAR:
            c_peso.markdown("—")
            c_hc.markdown("—")
            if linha_sugerida is None:
                por_escolher.append(classe)
            continue

        linha = tabela.loc[escolha]
        peso_estimado = calculadora.estimar_peso(det['area_cm2'], linha['categoria'])
        # A chave inclui o alimento escolhido: ao mudar de alimento, o peso volta a ser estimado
        peso = c_peso.number_input(
            "Peso (g)", min_value=0, value=int(round(peso_estimado)), step=5,
            key=f"peso_{n}_{escolha}", label_visibility="collapsed",
        )
        hc = peso * linha['hc_por_100g'] / 100
        total_hc += hc
        c_hc.markdown(f"**{hc:.1f} g** HC")

    if por_escolher:
        st.warning(
            "Estes alimentos não têm correspondência automática e **não estão no total**. "
            "Escolhe o alimento na lista ou conta-os à parte: " + ", ".join(por_escolher)
        )

    st.markdown("---")
    met1, met2 = st.columns(2)
    met1.metric("Total de hidratos de carbono", f"{total_hc:.1f} g")
    met2.metric(
        f"Intervalo (±{MARGEM_SEGURANCA:.0%})",
        f"{total_hc * (1 - MARGEM_SEGURANCA):.1f} – {total_hc * (1 + MARGEM_SEGURANCA):.1f} g",
    )
    st.caption(
        "Estimativa a partir de uma foto: o peso pode ter erros de dezenas de gramas. "
        "Não substitui a pesagem nem a orientação da tua equipa de saúde."
    )
