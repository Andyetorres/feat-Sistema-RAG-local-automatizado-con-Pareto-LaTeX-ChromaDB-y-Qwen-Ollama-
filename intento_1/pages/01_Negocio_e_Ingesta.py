import streamlit as st
import pandas as pd
import os
import io

from src.simulador_datos import generar_datos_simulados
from src.buscador import filtrar_documentos_pareto

duckdb_ok = True
try:
    import duckdb
except ImportError:
    duckdb_ok = False

st.set_page_config(page_title="Fase 1: Negocio & Ingesta", page_icon="📁", layout="wide")

st.header("1. Ingesta de Datos, Pregunta de Negocio y Literatura")

col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📁 Ingesta de Archivos (Datos o Contexto de Texto)")
    archivo = st.file_uploader(
        "Sube un archivo estructurado (CSV, Excel) o de texto/literatura",
        type=["xlsx", "txt", "csv", "md"]
    )
    
    if archivo is not None:
        try:
            if archivo.name.endswith(('.csv', '.txt', '.md')):
                contenido_archivo = archivo.read()
                try:
                    texto_dec = contenido_archivo.decode("utf-8")
                except UnicodeDecodeError:
                    texto_dec = contenido_archivo.decode("latin-1")
                
                st.session_state["texto_usuario"] = texto_dec
                st.success("📄 Archivo de texto/contexto cargado con éxito.")
                
                try:
                    df_subido = pd.read_csv(io.StringIO(texto_dec), sep=None, engine='python')
                    if len(df_subido.columns) > 1:
                        st.session_state["df_tabular"] = df_subido
                except Exception:
                    pass
            elif archivo.name.endswith('.xlsx'):
                df_subido = pd.read_excel(archivo)
                st.session_state["df_tabular"] = df_subido
                st.success(f"¡Archivo Excel '{archivo.name}' cargado con éxito!")
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

with col_der:
    st.subheader("⚡ Generador de Respaldo (DuckDB)")
    st.caption("Genera 500 registros sintéticos de flujos de personas en el restaurante.")
    if st.button("🚀 Generar Datos Sintéticos"):
        try:
            generar_datos_simulados()
            st.success("¡500 registros sintéticos generados en DuckDB!")
            st.rerun()
        except Exception as e:
            st.error(f"Error generando datos: {e}")

db_path = "data/restaurante_analytics.duckdb"
df = pd.DataFrame()
if os.path.exists(db_path) and duckdb_ok:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM hechos_video").fetchdf()
    except Exception:
        pass
    con.close()
elif "df_tabular" in st.session_state:
    df = st.session_state["df_tabular"]

if not df.empty:
    st.info(f"Fuente de datos activa con {len(df):,} registros listos.")
    st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")
st.subheader("❓ Pregunta de Negocio")
pregunta = st.text_area(
    "Define la decisión o pregunta que respaldarán los datos:",
    value=st.session_state.get("pregunta_negocio", "¿Cuántas personas llegaron al restaurante y qué zonas registran mayor afluencia?"),
    height=80
)
st.session_state["pregunta_negocio"] = pregunta

st.markdown("---")
st.subheader("🔎 Filtrado de Literatura (Principio de Pareto 80/20)")
if st.button("Aplicar Pareto a la Literatura Base"):
    docs_pareto = filtrar_documentos_pareto(umbral_objetivo=80)
    st.session_state["docs_pareto"] = docs_pareto
    st.success(f"¡Se seleccionaron {len(docs_pareto)} documentos clave bajo el criterio de Pareto!")
    for d in docs_pareto:
        st.markdown(f"- **{d['titulo']}** (Aporte: {d['aporte_porcentual']}%)")

st.markdown("---")
if st.button("Siguiente: Datos & Preparación ➡️"):
    st.switch_page("pages/02_Datos_y_Preparacion.py")