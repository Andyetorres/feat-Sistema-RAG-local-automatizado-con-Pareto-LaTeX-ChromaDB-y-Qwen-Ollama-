import streamlit as st
import pandas as pd
import duckdb
import os
from src.nucleo_mineria import pronosticar, segmentar

st.set_page_config(page_title="Fase 3: Modelado Analítico", page_icon="🤖", layout="wide")

st.header("🤖 Fase 4: Modelado Analítico (Series de Tiempo y Clústeres)")

@st.cache_data
def cargar_datos_pagina3():
    db_path = "data/restaurante_analytics.duckdb"
    if os.path.exists(db_path):
        con = duckdb.connect(db_path, read_only=True)
        try:
            df = con.execute("SELECT * FROM hechos_video").fetchdf()
            con.close()
            return df
        except Exception:
            con.close()
    return pd.DataFrame()

df = cargar_datos_pagina3()
if df.empty and "df_tabular" in st.session_state:
    df = st.session_state["df_tabular"]

if not df.empty:
    st.subheader("📈 Pronóstico de Flujo (Series de Tiempo / Holt)")
    try:
        if 'timestamp_segundo' in df.columns:
            df['minuto'] = (df['timestamp_segundo'] / 60).astype(int)
            res_serie = pronosticar(df, periodo='minuto', medida='track_id', agregacion='conteo', horizonte=5)
            st.session_state["res_serie"] = res_serie
            st.line_chart(res_serie["tabla"][["observado", "pronóstico"]])
            for line in res_serie["lineas"]:
                st.caption(f"• {line}")
        else:
            st.info("Columna 'timestamp_segundo' no disponible para serie temporal.")
    except Exception as e:
        st.warning(f"Error calculando serie temporal: {e}")

    st.markdown("---")
    st.subheader("🎯 Segmentación por Clústeres (K-Medias / Buyer Personas)")
    try:
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != 'track_id']
        if len(num_cols) >= 2:
            res_seg = segmentar(df, variables=num_cols[:3], k=3)
            st.session_state["res_seg"] = res_seg
            st.dataframe(res_seg["perfil"], use_container_width=True)
            st.markdown("**Gráfica del Codo (Inercia):**")
            st.line_chart(res_seg["codo"])
        else:
            st.info("Insuficientes columnas numéricas para segmentar.")
    except Exception as e:
        st.warning(f"Error al segmentar: {e}")

    st.markdown("---")
    if st.button("Siguiente: Informe LaTeX & IA ➡️"):
        st.switch_page("pages/04_Informe_LaTeX.py")
else:
    st.warning("⚠️ No hay datos cargados. Ve a la página '01_Negocio_e_Ingesta'.")