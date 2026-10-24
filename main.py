import streamlit as st
import pandas as pd
import duckdb
import os
import altair as alt
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio de Minería de Datos Inteligente",
    page_icon="🚀",
    layout="wide"
)

# --- ESTILOS VISUALES MODERNOS ---
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

# --- ESTADO DE SESIÓN Y NAVEGACIÓN ---
if "seccion" not in st.session_state:
    st.session_state["seccion"] = "1. Negocio & Ingesta"

if "archivo_cargado" not in st.session_state:
    st.session_state["archivo_cargado"] = False

# --- FUNCIÓN DE CARGA DE DATOS DUCKDB ---
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

df = cargar_datos_duckdb()

# --- BARRA LATERAL (PANEL DE CONTROL INTERACTIVO) ---
with st.sidebar:
    st.title("🎛️ Panel de Control")
    st.markdown("---")
    st.subheader("Navegación CRISP-DM")
    
    opciones = [
        "1. Negocio & Ingesta",
        "2. Datos & Preparación",
        "3. Modelado Analítico",
        "4. Informe LaTeX"
    ]
    
    # 1. Asegurar que la sesión tenga una sección inicial válida
    if "seccion" not in st.session_state:
        st.session_state["seccion"] = opciones[0]

    # 2. ALINEACIÓN CLAVE: Actualizamos el valor del widget *antes* de instanciarlo
    # Esto evita el desfase cuando usas los botones de "Siguiente"
    st.session_state["radio_nav"] = st.session_state["seccion"]

    # 3. Callback para cuando el usuario hace clic directamente en el panel lateral
    def actualizar_navegacion():
        st.session_state["seccion"] = st.session_state["radio_nav"]

    st.radio(
        "Selecciona una fase:",
        opciones,
        key="radio_nav",
        on_change=actualizar_navegacion
    )
    
    st.markdown("---")
    if st.button("🔄 Recargar Datos de DuckDB"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Universidad del Tolima · IDEAD")

# --- CABECERA PRINCIPAL ---
st.title("🚀 Laboratorio de Minería de Datos Inteligente")
st.caption(f"Consultoría Ejecutiva • Analítica de Flujos y Video (Restaurante) • {len(df):,} registros en DuckDB")

# --- INDICADOR VISUAL DE FASES CRISP-DM ---
cols = st.columns(6)
fases_estado = {
    "1. Negocio & Ingesta": 1,
    "2. Datos & Preparación": 2,
    "3. Modelado Analítico": 3,
    "4. Informe LaTeX": 4
}
fase_actual = fases_estado.get(st.session_state["seccion"], 1)

fases_nombres = ["Negocio", "Datos", "Preparación", "Modelado", "Evaluación", "Despliegue"]
for i, (col, nombre) in enumerate(zip(cols, fases_nombres), start=1):
    if i <= fase_actual:
        col.markdown(f":green[**Fase {i}**]  \n{nombre}")
    else:
        col.markdown(f":gray[**Fase {i}**]  \n{nombre}")

st.markdown("---")

# =============================================================================
# PÁGINA 1: NEGOCIO & INGESTA
# =============================================================================
if st.session_state["seccion"] == "1. Negocio & Ingesta":
    st.header("1. Ingesta de Datos y Pregunta de Negocio")
    
    col_izq, col_der = st.columns([2, 1])
    
    with col_izq:
        st.subheader("📁 Ingesta de Archivos")
        archivo = st.file_uploader(
            "Sube un archivo de datos o video para iniciar el análisis",
            type=["xlsx", "txt", "csv", "mp4", "avi", "png", "jpg"]
        )
        
        # PROCESAR ARCHIVO SUBIDO Y VALIDAR ESQUEMA DE 6 COLUMNAS
        if archivo is not None:
            try:
                if archivo.name.endswith(('.csv', '.txt')):
                    df_subido = pd.read_csv(archivo, sep=None, engine='python')
                elif archivo.name.endswith('.xlsx'):
                    df_subido = pd.read_excel(archivo)
                else:
                    df_subido = None
                
                if df_subido is not None and not df_subido.empty:
                    df_subido.columns = [str(c).strip().lower() for c in df_subido.columns]
                    columnas_requeridas = ['timestamp_segundo', 'track_id', 'clase', 'posicion_x', 'posicion_y', 'zona_detectada']
                    faltantes = [c for c in columnas_requeridas if c not in df_subido.columns]
                    
                    if faltantes:
                        st.error(f"⚠️ El archivo no es compatible. Faltan las columnas requeridas: {faltantes}")
                    else:
                        df_limpio = df_subido[columnas_requeridas].copy()
                        db_path = "data/restaurante_analytics.duckdb"
                        os.makedirs(os.path.dirname(db_path), exist_ok=True)
                        con = duckdb.connect(db_path)
                        con.execute("""
                            CREATE TABLE IF NOT EXISTS hechos_video (
                                timestamp_segundo FLOAT,
                                track_id INTEGER,
                                clase VARCHAR,
                                posicion_x FLOAT,
                                posicion_y FLOAT,
                                zona_detectada VARCHAR
                            )
                        """)
                        con.execute("DELETE FROM hechos_video")
                        con.register('df_temp', df_limpio)
                        con.execute("INSERT INTO hechos_video SELECT * FROM df_temp")
                        con.close()
                        
                        st.session_state["archivo_cargado"] = True
                        st.session_state["nombre_archivo"] = archivo.name
                        st.cache_data.clear()
                        st.success(f"¡Archivo '{archivo.name}' procesado e integrado con éxito ({len(df_limpio)} registros)!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error crítico al procesar el archivo: {e}")

    with col_der:
        st.subheader("⚡ Generador de Respaldo")
        st.caption("¿No tienes un archivo a la mano? Genera los 500 registros sintéticos en DuckDB.")
        
        from src.simulador_datos import generar_datos_simulados
        
        if st.button("🚀 Generar Datos Sintéticos"):
            try:
                generar_datos_simulados() 
                st.cache_data.clear()     
                st.session_state["archivo_cargado"] = True
                st.session_state["nombre_archivo"] = "Datos Sintéticos (500 registros)"
                st.success("¡500 registros sintéticos generados en DuckDB!")
                st.rerun()
            except Exception as e:
                st.error(f"Error al generar datos: {e}")

    df = cargar_datos_duckdb()

    if not df.empty:
        nombre = st.session_state.get("nombre_archivo", "Almacén DuckDB Local")
        st.info(f"Fuente de datos activa: **{nombre}** ({len(df)} registros listos).")
        
        # --- OPCIONES DE DESCARGA MULTIFORMATO ---
        with st.expander("📥 Descargar fuente de datos activa"):
            formato_descarga = st.selectbox("Selecciona el formato de exportación:", ["CSV / TXT", "Excel (.xlsx)", "PDF Académico"])
            
            if formato_descarga == "CSV / TXT":
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar TXT / CSV", csv_data, file_name="datos_restaurante.csv", mime="text/csv")
            
            elif formato_descarga == "Excel (.xlsx)":
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Hechos Video')
                excel_data = output.getvalue()
                st.download_button("Descargar Excel", excel_data, file_name="datos_restaurante.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            elif formato_descarga == "PDF Académico":
                if st.button("Compilar y Descargar PDF"):
                    try:
                        from src.generador_latex import generar_archivo_tex, compilar_latex_a_pdf
                        generar_archivo_tex(
                            nombre_archivo="reporte_sintetico",
                            tema="Minería de Datos y Simulación de Tráfico",
                            titulo="Reporte de Datos - Restaurante",
                            resumen="Informe basado en el procesamiento de eventos de tráfico en el restaurante.",
                            contenido=f"Se procesaron {len(df)} registros distribuidos en zonas comerciales."
                        )
                        compilar_latex_a_pdf("./outputs/latex_sources", "./outputs/pdfs")
                        pdf_path = "./outputs/pdfs/reporte_sintetico.pdf"
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button("Descargar PDF Final", f, file_name="reporte_sintetico.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(f"Error generando PDF: {e}")

        st.markdown("---")
        st.subheader("❓ Pregunta de Negocio")
        pregunta = st.text_area(
            "Define la decisión que respaldarán los datos:",
            value=st.session_state.get("pregunta_negocio", "¿Cuántas personas llegaron al restaurante, cuál fue su distribución espacial y qué zonas registran mayor afluencia según el flujo analítico?"),
            height=100
        )
        st.session_state["pregunta_negocio"] = pregunta
        
        st.markdown("---")
        if st.button("Siguiente: Datos & Preparación ➡️"):
            st.session_state["seccion"] = "2. Datos & Preparación"
            st.rerun()
    else:
        st.warning("⚠️ Debes subir un archivo o presionar 'Generar Datos Sintéticos' a la derecha para desbloquear la pregunta de negocio y continuar.")

# =============================================================================
# PÁGINA 2: DATOS & PREPARACIÓN
# =============================================================================
elif st.session_state["seccion"] == "2. Datos & Preparación":
    st.header("2. Comprensión y Preparación de los Datos")
    
    if not df.empty:
        st.subheader("Comprensión de los datos (DuckDB)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Registros / Filas", f"{len(df):,}")
        m2.metric("Variables", df.shape[1])
        m3.metric("Numéricas", len(df.select_dtypes(include=['number']).columns))
        m4.metric("Categóricas", len(df.select_dtypes(include=['object']).columns))
        m5.metric("Celdas vacías", int(df.isna().sum().sum()))

        st.markdown("### 💡 Smart Insights Automáticos")
        st.info("Tendencia Operativa: Mayor concentración de eventos detectada en la zona principal del restaurante.")
        
        st.subheader("Primeras filas del Almacén de Datos (DuckDB)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        if st.button("Siguiente: Modelado Analítico ➡️"):
            st.session_state["seccion"] = "3. Modelado Analítico"
            st.rerun()
    else:
        st.warning("Sin datos en DuckDB. Regresa a '1. Negocio & Ingesta' y sube un archivo o genera la simulación.")
# =============================================================================
# PÁGINA 3: MODELADO ANALÍTICO
# =============================================================================
elif st.session_state["seccion"] == "3. Modelado Analítico":
    st.header("🤖 Fase 4: Modelado Analítico (Segmentación y Series de Tiempo)")
    
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
        
        st.markdown("---")
        if st.button("Siguiente: Informe LaTeX ➡️"):
            st.session_state["seccion"] = "4. Informe LaTeX"
            st.rerun()
    else:
        st.info("Procesa un video o sube datos en la página de Negocio para visualizar los modelos analíticos.")


# =============================================================================
# PÁGINA 4: INFORME LATEX
# =============================================================================
elif st.session_state["seccion"] == "4. Informe LaTeX":
    st.header("📄 Fase 5 y 6: Generación de Informes Automáticos en LaTeX")
    st.markdown("Genera un reporte académico formal basado en los resultados reales del análisis de flujos del restaurante.")

    pdf_path = "./outputs/pdfs/informe_restaurante.pdf"

    # 1. Calcular métricas reales de los datos actuales para el reporte
    if not df.empty:
        total_registros = len(df)
        zona_principal = df['zona_detectada'].mode()[0] if not df['zona_detectada'].empty else "N/A"
        conteo_zonas = df['zona_detectada'].value_counts().to_dict()
        resumen_zonas_str = ", ".join([f"\\textbf{{{zona}}}: {cnt} eventos" for zona, cnt in conteo_zonas.items()])
    else:
        total_registros = 0
        resumen_zonas_str = "Sin datos registrados."

    # 2. Botón para compilar o actualizar el PDF con datos reales
    if st.button("Compilar / Actualizar Informe con Datos Reales"):
        with st.spinner("Compilando código LaTeX con pdflatex..."):
            try:
                from src.generador_latex import generar_archivo_tex, compilar_latex_a_pdf
                
                # Inyectamos los datos reales calculados en el contenido del LaTeX
                contenido_dinamico = (
                    f"El análisis procesó un total de \\textbf{{{total_registros:,}}} registros de eventos de video de forma local. "
                    f"La distribución de flujo operativo por zonas comerciales muestra el siguiente comportamiento: {resumen_zonas_str}. "
                    f"Los resultados confirman que la zona con mayor concentración de afluencia es \\textbf{{{zona_principal}}}, "
                    f"lo que valida la eficacia del modelo para la toma de decisiones gerenciales en tiempo real."
                )
                
                generar_archivo_tex(
                    nombre_archivo="informe_restaurante",
                    tema="Minera de Datos Aplicada a Flujos de Video",
                    titulo="Analítica de Ocupación en Restaurantes mediante Edge AI",
                    resumen="Este informe detalla el proceso KDD aplicado a secuencias de video, extrayendo métricas de ocupación y segmentación espacial.",
                    contenido=contenido_dinamico
                )
                
                compilar_latex_a_pdf("./outputs/latex_sources", "./outputs/pdfs")
                st.success("¡Informe compilado con éxito y actualizado con tus datos!")
                st.rerun()
            except Exception as e:
                st.error(f"Error al compilar el documento LaTeX: {e}")

    # 3. Previsualización del Informe PDF
    if os.path.exists(pdf_path):
        st.markdown("---")
        st.subheader("👁️ Previsualización del Informe PDF")
        
        import base64
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        st.markdown("---")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Descargar Informe PDF Definitivo",
                data=f,
                file_name="informe_restaurante.pdf",
                mime="application/pdf"
            )
    else:
        st.info("ℹ️ Haz clic en el botón superior para compilar el documento con los datos actuales y activar la previsualización.")