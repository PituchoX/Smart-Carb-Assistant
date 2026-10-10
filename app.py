import os
import tempfile
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from nutrition.calculator import CalculadoraNutricional
from vision.detetor import SmartCarbDetector

st.set_page_config(
    page_title="Smart Carb Assistant",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Smart Carb Assistant")
st.markdown(
    "**Computer Vision & Mapeamento Nutricional (APDP / INSA)** para contagem precisa de hidratos de carbono em pratos portugueses."
)

# Barra lateral para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    diametro_prato = st.slider(
        "Diâmetro de referência do prato (cm):",
        min_value=20.0,
        max_value=32.0,
        value=26.0,
        step=0.5
    )
    st.info(
        "A calibração de escala estima o diâmetro do prato branco via OpenCV para converter píxeis em cm² reais sem marcadores externos."
    )

# Upload da imagem
uploaded_file = st.file_uploader(
    "Carrega uma foto do teu prato (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Guardar ficheiro temporário para o OpenCV processar
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        caminho_tmp = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Imagem Original")
        st.image(uploaded_file, use_container_width=True)

    with st.spinner("A processar imagem (calibração de prato e segmentação alimentar)..."):
        try:
            detetor = SmartCarbDetector(diametro_prato_cm=diametro_prato)
            detecoes = detetor.process_meal(caminho_tmp, salvar_anotada=True)

            with col2:
                st.subheader("🎯 Segmentação Detetada")
                if os.path.exists("resultado_detecao.jpg"):
                    img_result = Image.open("resultado_detecao.jpg")
                    st.image(img_result, use_container_width=True)

            if not detecoes:
                st.warning("Nenhum alimento catalogado foi identificado na foto.")
            else:
                calculadora = CalculadoraNutricional()
                resultado = calculadora.calcular_prato(detecoes)

                st.markdown("---")
                st.subheader("📊 Estimativa Nutricional (Tabela APDP / INSA)")

                # Preparar tabela para visualização
                dados_tabela = []
                for item in resultado["itens"]:
                    sem_valor = item["hc_g"] is None
                    dados_tabela.append({
                        "Alimento Detetado": item["detetado"],
                        "Correspondência": "⚠️ sem correspondência" if sem_valor else item["alimento"],
                        "Fonte": item["fonte"] or "-",
                        "Peso Estimado (g)": f"{item['peso_g']:.0f} g",
                        "Hidratos de Carbono (g)": "?" if sem_valor else f"{item['hc_g']:.1f} g",
                        "Margem Clínica (±15%)": "-" if sem_valor else f"[{item['margem'][0]:.1f}g - {item['margem'][1]:.1f}g]"
                    })

                df_resultado = pd.DataFrame(dados_tabela)
                st.dataframe(df_resultado, use_container_width=True)

                if resultado["sem_correspondencia"]:
                    st.warning(
                        "Estes alimentos foram detetados mas não têm um valor de HC fiável e "
                        "**não estão incluídos no total** — conta-os à parte: "
                        + ", ".join(resultado["sem_correspondencia"])
                    )

                # Destaque com Métricas
                met1, met2 = st.columns(2)
                with met1:
                    st.metric(
                        label="Total Estimado de Hidratos de Carbono",
                        value=f"{resultado['total_g']} g"
                    )
                with met2:
                    st.metric(
                        label="Intervalo Clínico Seguro (±15%)",
                        value=f"{resultado['intervalo_clinico'][0]}g - {resultado['intervalo_clinico'][1]}g"
                    )

        except ValueError as e:
            st.error(f"Erro na calibração: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro no processamento: {e}")
        finally:
            if os.path.exists(caminho_tmp):
                os.remove(caminho_tmp)