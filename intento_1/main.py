import streamlit as st
import pandas as pd
import duckdb
import os

# --- 1. CONFIGURACIÓN GLOBAL DE LA PÁGINA ---
st.set_page_config(
    page_title="Laboratorio de Minería de Datos Inteligente",
    page_icon="🚀",
    layout="wide"
)

# --- 2. ESTILOS VISUALES MODERNOS (CSS) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(51, 65, 85, 0.7) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. INICIALIZACIÓN DEL ESTADO DE SESIÓN GLOBAL ---
if "archivo_cargado" not in st.session_state:
    st.session_state["archivo_cargado"] = False

if "res_prep" not in st.session_state:
    st.session_state["res_prep"] = None

if "res_modelos" not in st.session_state:
    st.session_state["res_modelos"] = {}

if "pregunta_negocio" not in st.session_state:
    st.session_state["pregunta_negocio"] = "¿Cuántas personas llegaron al restaurante, cuál fue su distribución espacial y qué zonas registran mayor afluencia?"


# --- 4. FUNCIÓN GLOBAL DE CARGA DESDE DUCKDB ---
@st.cache_data
def cargar_datos_duckdb():
    db_path = "data/restaurante_analytics.duckdb"
    if os.path.exists(db_path):
        con = duckdb.connect(db_path, read_only=True)
        try:
            df = con.execute("SELECT * FROM hechos_video").fetchdf()
            con.close()
            return df
        except Exception:
            con.close()
    return pd.DataFrame(columns=['timestamp_segundo', 'track_id', 'clase', 'posicion_x', 'posicion_y', 'zona_detectada'])

df_global = cargar_datos_duckdb()

# --- 5. INTERFAZ DE LA PÁGINA PRINCIPAL (LANDING PAGE) ---
st.title("🚀 Laboratorio de Minería de Datos Inteligente")
st.subheader("Consultoría Ejecutiva • Analítica de Flujos y Video (Restaurante)")

if not df_global.empty:
    st.info(f"🗄️ Almacén DuckDB activo con **{len(df_global):,} registros** disponibles para análisis.")
else:
    st.warning("⚠️ No se detectaron registros en DuckDB. Dirígete a la Fase 1 para cargar datos o generar simulación.")

st.markdown("---")

st.markdown("""
### 🧭 Guía de Navegación CRISP-DM y RAG
Utiliza el menú lateral izquierdo para desplazarte por las diferentes etapas del proyecto:
1. **01_Negocio_e_Ingesta**: Carga de archivos, configuración de la pregunta de negocio y filtrado de literatura (Pareto).
2. **02_Datos_y_Preparacion**: Perfilado estadístico avanzado y limpieza de datos mediante el núcleo analítico.
3. **03_Modelado_Analitico**: Ejecución de series de tiempo (Holt) y segmentación por clústeres (K-Medias / Buyer Personas).
4. **04_Informe_LaTeX**: Consolidación técnica con datos reales, previsualización en PDF y vectorización final en ChromaDB.
""")

st.markdown("---")
st.caption("Universidad del Tolima · IDEAD • Arquitectura Modular de Minería de Datos")