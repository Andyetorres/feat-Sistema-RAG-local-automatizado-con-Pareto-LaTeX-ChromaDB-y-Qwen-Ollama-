import os
import time
import streamlit as st

from analizador_datos import AnalizadorServicio
from camara import capturar_frame_vivo, seleccionar_region_pantalla
from generador_latex import renderizar_seccion_latex
from mineria_visual import renderizar_capa_mineria
from seguimiento_yolo import DetectorSimulador, cargar_lector_ocr

st.set_page_config(
    page_title="Analítica Inteligente de Restaurante — Visión, DW & Minería",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Sistema Inteligente de Monitoreo y Minería de Datos en Restaurante")
st.caption("Arquitectura Integral: Visión Artificial (Cap. 1) • Data Warehouse DuckDB (Cap. 2) • Minería KDD/CRISP-DM (Cap. 3) • Reporte LaTeX/PDF (Cap. 4)")

with st.expander("📖 Guía de Funcionamiento y Arquitectura del Sistema", expanded=False):
    if os.path.exists("intento_2/123.jpeg"):
        st.image("intento_2/123.jpeg", width="stretch")
        st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Visión Artificial (Cap. 1)**")
        st.caption("Captura en tiempo real con `mss`, detección de entidades y lectura de zonas con EasyOCR.")
    with col2:
        st.markdown("**2. Data Warehouse (Cap. 2)**")
        st.caption("Almacenamiento dimensional de eventos, tiempos, mesas (`A1`–`B2`) y meseros en DuckDB.")
    with col3:
        st.markdown("**3. Minería de Datos (Cap. 3)**")
        st.caption("Procesamiento bajo metodología KDD/CRISP-DM con Clustering K-Means y Modelo Híbrido (Árbol + MLP).")
    with col4:
        st.markdown("**4. Reporte LaTeX (Cap. 4)**")
        st.caption("Generación automatizada de código `.tex` y compilación a documento ejecutivo PDF.")
    st.info("💡 **Flujo de uso:** 1) Selecciona la región del simulador → 2) Inicia en vivo → 3) Detén la captura para consultar y generar el informe LaTeX.")

# =========================================================================
# INICIALIZACIÓN DE ESTADO EN SESIÓN
# =========================================================================
if "region_live" not in st.session_state:
    st.session_state.region_live = None

if "grabando" not in st.session_state:
    st.session_state.grabando = False

if "detector" not in st.session_state:
    st.session_state.detector = DetectorSimulador()

if "analizador" not in st.session_state:
    st.session_state.analizador = AnalizadorServicio()

ocr_reader = cargar_lector_ocr()

# =========================================================================
# BARRA LATERAL: CONTROLES DE CAPTURA EN VIVO
# =========================================================================
with st.sidebar:
    st.header("🎛️ Control de Transmisión")

    if st.button("🎯 Seleccionar Región del Simulador", width="stretch"):
        st.session_state.grabando = False
        region = seleccionar_region_pantalla()
        if region:
            st.session_state.region_live = region
            st.success(f"Región encuadrada: {region}")
        else:
            st.warning("Selección cancelada.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Iniciar en Vivo", width="stretch", disabled=st.session_state.region_live is None):
            st.session_state.grabando = True
    with col_btn2:
        if st.button("⏸️ Detener Captura", width="stretch"):
            st.session_state.grabando = False

    if st.button("🔄 Reiniciar Data Warehouse", width="stretch"):
        st.session_state.grabando = False
        st.session_state.detector = DetectorSimulador()
        st.session_state.analizador = AnalizadorServicio()
        if "ultimo_reporte_latex" in st.session_state:
            del st.session_state.ultimo_reporte_latex
        st.rerun()

    st.divider()
    st.markdown("**Estado del Sistema:**")
    if st.session_state.grabando:
        st.success("🟢 Capturando y analizando en vivo")
    elif st.session_state.region_live:
        st.warning("🟡 En pausa (datos guardados en DW)")
    else:
        st.info("⚪ Esperando encuadre de pantalla")

# =========================================================================
# CAPA 1 Y CAPA 2: VISIÓN EN VIVO + ANALIZADOR DE CONSULTAS (DATA WAREHOUSE)
# =========================================================================
col_camara, col_analizador = st.columns([1.15, 1])

with col_camara:
    st.subheader("📡 1. Cámara / Seguimiento en Vivo")
    visor_video = st.empty()

with col_analizador:
    st.subheader("🧠 2. Analizador de Consultas (Data Warehouse)")
    pregunta_usuario = st.text_input(
        "¿Qué quieres analizar?",
        value="cuantos clientes vinieron hoy, cuantas veces ocuparon cada mesa y que mesero trabajo mas",
        key="input_consulta_dw",
    )
    respuesta_placeholder = st.empty()
    tabla_placeholder = st.empty()

st.divider()

# =========================================================================
# CONTENEDOR PARA CAPA 3: MINERÍA DE DATOS, K-MEANS & MODELO HÍBRIDO
# =========================================================================
contenedor_mineria = st.empty()

# =========================================================================
# CAPA 4: GENERADOR DE INFORME LATEX / PDF (FUERA DEL BUCLE EN VIVO)
# =========================================================================
renderizar_seccion_latex(st.session_state.analizador)

# =========================================================================
# BUCLE DE EJECUCIÓN EN TIEMPO REAL vs MODO PAUSA / CONSULTA
# =========================================================================
if st.session_state.region_live and st.session_state.grabando:
    ciclo = 0
    while st.session_state.grabando:
        ciclo += 1
        frame_bgr = capturar_frame_vivo(st.session_state.region_live)

        # Capa 1: Visión Artificial y Rastreo Determinista
        frame_procesado, datos_frame = st.session_state.detector.procesar_frame(frame_bgr, ocr_reader)

        # Capa 2: Persistencia en DuckDB y respuesta dinámica
        st.session_state.analizador.registrar_frame(datos_frame)
        texto_respuesta, df_detalle, tema = st.session_state.analizador.responder_pregunta_usuario(pregunta_usuario)

        visor_video.image(frame_procesado, caption="Capturando y analizando en tiempo real...", width="stretch")

        with respuesta_placeholder.container():
            st.success(texto_respuesta)

        with tabla_placeholder.container():
            if not df_detalle.empty:
                st.dataframe(df_detalle, width="stretch", hide_index=True)

        # Capa 3: Actualizar Minería de Datos, Frecuencias, K-Means y Modelo Híbrido cada 12 frames
        if ciclo == 1 or ciclo % 12 == 0:
            df_hechos, df_mesas, df_meseros = st.session_state.analizador.obtener_datasets_dw()
            contenedor_mineria.empty()
            with contenedor_mineria.container():
                renderizar_capa_mineria(df_hechos, df_mesas, df_meseros, pregunta_usuario, tema, ciclo=ciclo)

        time.sleep(0.12)
else:
    visor_video.info("Presiona **'Seleccionar Región del Simulador'** y luego **'Iniciar en Vivo'** en el panel izquierdo.")

    df_hechos, df_mesas, df_meseros = st.session_state.analizador.obtener_datasets_dw()
    if not df_hechos.empty:
        texto_respuesta, df_detalle, tema = st.session_state.analizador.responder_pregunta_usuario(pregunta_usuario)
        respuesta_placeholder.success(texto_respuesta)
        if not df_detalle.empty:
            tabla_placeholder.dataframe(df_detalle, width="stretch", hide_index=True)
        contenedor_mineria.empty()
        with contenedor_mineria.container():
            renderizar_capa_mineria(df_hechos, df_mesas, df_meseros, pregunta_usuario, tema, ciclo=0)