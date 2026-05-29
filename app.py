import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sqlalchemy import create_engine
from datetime import datetime
import os
import plotly.express as px

# =========================
# CONFIGURAÇÃO
# =========================

st.set_page_config(
    page_title="Monitoramento de Feridas",
    layout="wide"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# BANCO DE DADOS
# =========================

engine = create_engine("sqlite:///database.db")

# =========================
# CRIAR TABELA
# =========================

query = """
CREATE TABLE IF NOT EXISTS feridas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente TEXT,
    data TEXT,
    imagem TEXT,
    area_cm REAL,
    perimetro REAL,
    observacoes TEXT
)
"""

with engine.begin() as conn:
    conn.exec_driver_sql(query)

# =========================
# FUNÇÕES
# =========================

def segmentar_ferida(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Intervalo de cor para detectar áreas avermelhadas
    lower = np.array([0, 30, 30])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return mask, contours


def calcular_medidas(contour, pixels_por_cm=37):

    area_pixels = cv2.contourArea(contour)

    perimetro_pixels = cv2.arcLength(
        contour,
        True
    )

    area_cm = area_pixels / (pixels_por_cm ** 2)

    perimetro_cm = perimetro_pixels / pixels_por_cm

    return area_cm, perimetro_cm


def desenhar_contorno(img, contour):

    output = img.copy()

    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        4
    )

    return output


def salvar_dados(
    paciente,
    data,
    imagem,
    area_cm,
    perimetro_cm,
    observacoes
):

    df = pd.DataFrame({
        "paciente": [paciente],
        "data": [data],
        "imagem": [imagem],
        "area_cm": [area_cm],
        "perimetro": [perimetro_cm],
        "observacoes": [observacoes]
    })

    df.to_sql(
        "feridas",
        engine,
        if_exists="append",
        index=False
    )


def carregar_dados():

    query = "SELECT * FROM feridas"

    return pd.read_sql(query, engine)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Monitoramento")

pagina = st.sidebar.radio(
    "Menu",
    [
        "Nova Avaliação",
        "Histórico",
        "Dashboard"
    ]
)

# =========================
# NOVA AVALIAÇÃO
# =========================

if pagina == "Nova Avaliação":

    st.title("Nova Avaliação de Ferida")

    paciente = st.text_input(
        "Nome do paciente"
    )

    observacoes = st.text_area(
        "Observações"
    )

    st.subheader(
        "Captura da imagem"
    )

    foto = st.camera_input(
        "Fotografe a ferida"
    )

    upload = st.file_uploader(
        "Ou envie uma imagem",
        type=["png", "jpg", "jpeg"]
    )

    imagem_final = None

    if foto:
        imagem_final = foto

    elif upload:
        imagem_final = upload

    if imagem_final:

        try:

            # =========================
            # LEITURA SEGURA DA IMAGEM
            # =========================

            image = Image.open(
                imagem_final
            ).convert("RGB")

            img = np.array(image)

            st.image(
                img,
                caption="Imagem Original",
                use_container_width=True
            )

            # =========================
            # SEGMENTAÇÃO
            # =========================

            mask, contours = segmentar_ferida(img)

            st.subheader("Segmentação")

            st.image(
                mask,
                caption="Máscara da Ferida",
                use_container_width=True
            )

            if len(contours) > 0:

                maior_contorno = max(
                    contours,
                    key=cv2.contourArea
                )

                area_cm, perimetro_cm = calcular_medidas(
                    maior_contorno
                )

                imagem_contorno = desenhar_contorno(
                    img,
                    maior_contorno
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        imagem_contorno,
                        caption="Ferida Detectada",
                        use_container_width=True
                    )

                with col2:

                    st.metric(
                        "Área estimada (cm²)",
                        round(area_cm, 2)
                    )

                    st.metric(
                        "Perímetro estimado (cm)",
                        round(perimetro_cm, 2)
                    )

                # =========================
                # SALVAR
                # =========================

                if st.button(
                    "Salvar Avaliação"
                ):

                    nome_arquivo = (
                        f"{datetime.now().timestamp()}.png"
                    )

                    caminho = os.path.join(
                        UPLOAD_FOLDER,
                        nome_arquivo
                    )

                    cv2.imwrite(
                        caminho,
                        cv2.cvtColor(
                            img,
                            cv2.COLOR_RGB2BGR
                        )
                    )

                    salvar_dados(
                        paciente,
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        caminho,
                        area_cm,
                        perimetro_cm,
                        observacoes
                    )

                    st.success(
                        "Avaliação salva com sucesso"
                    )

            else:

                st.error(
                    "Nenhuma ferida detectada"
                )

        except Exception as e:

            st.error(
                f"Erro ao processar imagem: {e}"
            )

# =========================
# HISTÓRICO
# =========================

elif pagina == "Histórico":

    st.title(
        "Histórico de Avaliações"
    )

    dados = carregar_dados()

    if len(dados) == 0:

        st.warning(
            "Nenhuma avaliação encontrada"
        )

    else:

        pacientes = dados[
            "paciente"
        ].unique()

        paciente_escolhido = st.selectbox(
            "Selecione o paciente",
            pacientes
        )

        dados_paciente = dados[
            dados["paciente"] == paciente_escolhido
        ]

        dados_paciente = dados_paciente.sort_values(
            by="data"
        )

        for _, row in dados_paciente.iterrows():

            st.divider()

            col1, col2 = st.columns([1, 1])

            with col1:

                if os.path.exists(row["imagem"]):

                    st.image(
                        row["imagem"],
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Imagem não encontrada"
                    )

            with col2:

                st.write(
                    f"Data: {row['data']}"
                )

                st.write(
                    f"Área: {round(row['area_cm'], 2)} cm²"
                )

                st.write(
                    f"Perímetro: {round(row['perimetro'], 2)} cm"
                )

                st.write(
                    f"Observações: {row['observacoes']}"
                )

# =========================
# DASHBOARD
# =========================

elif pagina == "Dashboard":

    st.title(
        "Dashboard de Evolução"
    )

    dados = carregar_dados()

    if len(dados) == 0:

        st.warning(
            "Nenhum dado encontrado"
        )

    else:

        pacientes = dados[
            "paciente"
        ].unique()

        paciente = st.selectbox(
            "Paciente",
            pacientes
        )

        df = dados[
            dados["paciente"] == paciente
        ].copy()

        df["data"] = pd.to_datetime(
            df["data"]
        )

        df = df.sort_values(
            by="data"
        )

        # =========================
        # GRÁFICO ÁREA
        # =========================

        fig = px.line(
            df,
            x="data",
            y="area_cm",
            markers=True,
            title="Evolução da Área da Ferida"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================
        # GRÁFICO PERÍMETRO
        # =========================

        fig2 = px.line(
            df,
            x="data",
            y="perimetro",
            markers=True,
            title="Evolução do Perímetro"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =========================
        # ANÁLISE
        # =========================

        if len(df) >= 2:

            area_inicial = df.iloc[0][
                "area_cm"
            ]

            area_final = df.iloc[-1][
                "area_cm"
            ]

            variacao = (
                (
                    area_inicial - area_final
                )
                / area_inicial
            ) * 100

            st.subheader(
                "Evolução"
            )

            if variacao > 0:

                st.success(
                    f"Redução de {round(variacao, 2)}%"
                )

            elif variacao < 0:

                st.error(
                    f"Aumento de {round(abs(variacao), 2)}%"
                )

            else:

                st.info(
                    "Sem alteração"
                )

        st.subheader(
            "Tabela de Dados"
        )

        st.dataframe(
            df,
            use_container_width=True
        )
