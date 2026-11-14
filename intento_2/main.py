import time
import streamlit as st

from analizador_datos import AnalizadorServicio
from camara import capturar_frame_vivo, seleccionar_region_pantalla
from mineria_visual import renderizar_capa_mineria
from seguimiento_yolo import DetectorSimulador, cargar_lector_ocr

st.set_page_config(page_title="Simulador - Data Warehouse & Minería", page_icon="🎥", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #28a745 !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        border: 2px solid #1e7e34 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "region_live" not in st.session_state:
    st.session_state.region_live = None
if "grabando" not in st.session_state:
    st.session_state.grabando = False
if "analizador" not in st.session_state:
    st.session_state.analizador = AnalizadorServicio()
if "detector" not in st.session_state:
    st.session_state.detector = DetectorSimulador()

ocr_reader = cargar_lector_ocr()

# ==========================================
# CAPA 1: CONTROL DE CÁMARA / CAPTURA EN VIVO
# ==========================================
st.title("🎥 Sistema Inteligente de Monitoreo y Minería de Datos")
st.markdown("### ¿No tienes cámara? ¿O no sabes programar una para que se conecte a esta web?")
st.caption("Toma, te prestamos esta **versión para pobres** para que uses XD")

col_btn1, col_btn2 = st.columns([1.2, 3])

with col_btn1:
    if st.button("!!aqui pobre!!"):
        region = seleccionar_region_pantalla()
        if region:
            st.session_state.region_live = region
            st.session_state.grabando = True
            st.rerun()

with col_btn2:
    if st.session_state.region_live:
        if st.session_state.grabando:
            if st.button("⏹️ Detener Grabación y Ver Minería Completa"):
                st.session_state.grabando = False
                st.rerun()
        else:
            if st.button("▶️ Reanudar Grabación"):
                st.session_state.grabando = True
                st.rerun()

st.divider()

# ==========================================
# CAPA 2: VISIÓN EN VIVO + ANALIZADOR DE CONSULTAS
# ==========================================
col_video, col_analitica = st.columns([1.5, 1.5])

with col_video:
    st.subheader("📡 1. Cámara / Seguimiento en Vivo")
    visor_video = st.empty()

with col_analitica:
    st.subheader("🧠 2. Analizador de Consultas (Data Warehouse)")
    pregunta_usuario = st.text_input(
        "¿Qué quieres analizar?",
        value="cuantos clientes vinieron hoy, cuantas veces ocuparon cada mesa y que mesero trabajo mas?",
        placeholder="Ej: cuantos clientes vinieron hoy / cuantas veces ocuparon B1 / que mesero trabajo mas",
    )
    respuesta_placeholder = st.empty()
    tabla_placeholder = st.empty()

st.divider()

# Contenedor para la Capa 3 (Minería de Datos, KDD, CRISP-DM, Modelos Híbridos y Gráficas)
contenedor_mineria = st.empty()

if st.session_state.region_live and st.session_state.grabando:
    ciclo = 0
    while st.session_state.grabando:
        ciclo += 1
        frame_bgr = capturar_frame_vivo(st.session_state.region_live)

        # Capa 1: Visión y Seguimiento
        frame_procesado, datos_frame = st.session_state.detector.procesar_frame(frame_bgr, ocr_reader)

        # Capa 2: Almacenamiento en Data Warehouse (DuckDB) y respuesta a consulta
        st.session_state.analizador.registrar_frame(datos_frame)
        texto_respuesta, df_detalle, tema = st.session_state.analizador.responder_pregunta_usuario(pregunta_usuario)

        visor_video.image(frame_procesado, caption="Capturando y analizando en tiempo real...", width="stretch")

        with respuesta_placeholder.container():
            st.success(texto_respuesta)

        with tabla_placeholder.container():
            if not df_detalle.empty:
                st.dataframe(df_detalle, width="stretch", hide_index=True)

        # Capa 3: Actualizar Minería de Datos, K-Means, Modelo Híbrido y Gráficas cada 12 frames
        if ciclo == 1 or ciclo % 12 == 0:
            df_hechos, df_mesas, df_meseros = st.session_state.analizador.obtener_datasets_dw()
            with contenedor_mineria.container():
                renderizar_capa_mineria(df_hechos, df_mesas, df_meseros, pregunta_usuario, tema, ciclo=ciclo)

        time.sleep(0.12)
else:
    visor_video.info("Presiona **!!aqui pobre!!** para encuadrar el simulador y comenzar a alimentar el Data Warehouse.")
    df_hechos, df_mesas, df_meseros = st.session_state.analizador.obtener_datasets_dw()
    if not df_hechos.empty:
        texto_respuesta, df_detalle, tema = st.session_state.analizador.responder_pregunta_usuario(pregunta_usuario)
        respuesta_placeholder.success(texto_respuesta)
        if not df_detalle.empty:
            tabla_placeholder.dataframe(df_detalle, width="stretch", hide_index=True)
        with contenedor_mineria.container():
            renderizar_capa_mineria(df_hechos, df_mesas, df_meseros, pregunta_usuario, tema)