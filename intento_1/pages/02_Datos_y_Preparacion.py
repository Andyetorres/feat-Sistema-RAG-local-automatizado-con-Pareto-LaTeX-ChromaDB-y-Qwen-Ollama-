import streamlit as st
import pandas as pd
import duckdb
import os
from src.nucleo_mineria import perfil, preparar, generar_smart_insights

st.set_page_config(page_title="Fase 2: Datos & Preparación", page_icon="📊", layout="wide")

st.header("2. Comprensión y Preparación de los Datos (CRISP-DM)")

@st.cache_data
def cargar_datos_pagina2():
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

df = cargar_datos_pagina2()
if df.empty and "df_tabular" in st.session_state:
    df = st.session_state["df_tabular"]

if not df.empty:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Registros / Filas", f"{len(df):,}")
    m2.metric("Variables", df.shape[1])
    m3.metric("Numéricas", len(df.select_dtypes(include=['number']).columns))
    m4.metric("Categóricas", len(df.select_dtypes(include=['object']).columns))
    m5.metric("Celdas vacías", int(df.isna().sum().sum()))

    st.subheader("📊 Perfil Estadístico Avanzado (Núcleo)")
    df_perfil = perfil(df)
    st.dataframe(df_perfil, use_container_width=True)

    st.subheader("💡 Smart Insights Automáticos")
    insights = generar_smart_insights(df, st.session_state.get("res_prep"), {})
    for ins in insights:
        st.info(ins)

    st.markdown("---")
    st.subheader("🧹 Preparación y Limpieza de Datos")
    col1, col2 = st.columns(2)
    with col1:
        tratar_vacios = st.selectbox("Estrategia para valores faltantes:", ["imputar", "eliminar", "ninguno"])
    with col2:
        tratar_atipicos = st.selectbox("Estrategia para atípicos (1.5xIQR):", ["recortar", "eliminar", "ninguno"])

    if st.button("⚙️ Ejecutar Preparación con el Núcleo"):
        res_prep = preparar(df, quitar_duplicados=True, vacios=tratar_vacios, atipicos=tratar_atipicos)
        st.session_state["res_prep"] = res_prep
        st.success(f"Preparación completada. Filas resultantes: {res_prep['info']['filas_despues']}")
        for log_msg in res_prep["log"]:
            st.caption(f"• {log_msg}")

    st.markdown("---")
    if st.button("Siguiente: Modelado Analítico ➡️"):
        st.switch_page("pages/03_Modelado_Analitico.py")
else:
    st.warning("⚠️ No hay datos cargados. Ve a la página '01_Negocio_e_Ingesta'.")