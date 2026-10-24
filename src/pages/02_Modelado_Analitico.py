import streamlit as st
import duckdb
import pandas as pd
import altair as alt

st.title("🤖 Fase 4: Modelado Analítico (Segmentación y Series de Tiempo)")

db_path = "data/restaurante_analytics.duckdb"

try:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM hechos_video").fetchdf()
    con.close()

    if not df.empty:
        st.subheader("📈 Serie de Tiempo: Flujo de Ingresos por Minuto")
        df['minuto'] = (df['timestamp_segundo'] / 60).astype(int)
        flujo_tiempo = df.groupby('minuto').size().reset_index(name='conteo_personas')
        
        chart = alt.Chart(flujo_tiempo).mark_line(point=True).encode(
            x='minuto:Q',
            y='conteo_personas:Q',
            tooltip=['minuto', 'conteo_personas']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.subheader("🎯 Segmentación por Zonas del Restaurante")
        segmentos = df.groupby('zona_detectada').size().reset_index(name='total')
        st.bar_chart(segmentos.set_index('zona_detectada'))

    else:
        st.info("Procesa un video en la página principal para visualizar los modelos analíticos.")
except Exception as e:
    st.warning("Base de datos no encontrada o sin registros estructurados todavía.")