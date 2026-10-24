import streamlit as st
import duckdb
import pandas as pd

st.title("📊 Fase 2 y 3: Perfilamiento y Preparación de Datos")

db_path = "data/restaurante_analytics.duckdb"

try:
    con = duckdb.connect(db_path, read_only=True)
    df_eventos = con.execute("SELECT * FROM hechos_video").fetchdf()
    con.close()
    
    if not df_eventos.empty:
        st.success(f"Datos cargados exitosamente desde DuckDB ({len(df_eventos)} registros de video analizados).")
        
        st.subheader("Vista Previa de Eventos Detectados (YOLO)")
        st.dataframe(df_eventos.head(10))
        
        st.subheader("Métricas de Calidad de Datos")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Detecciones", len(df_eventos))
        c2.metric("IDs Únicos (Personas Rastreadas)", df_eventos['track_id'].nunique())
        c3.metric("Valores Nulos", int(df_eventos.isna().sum().sum()))
    else:
        st.warning("La base de datos DuckDB está vacía. Sube y procesa un video primero desde la página principal (`app.py`).")

except Exception as e:
    st.error(f"No se pudo conectar a DuckDB: {e}. Asegúrate de procesar un video previamente.")