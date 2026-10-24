import streamlit as st
import os

st.title("📄 Fase 5 y 6: Generación de Informes Automáticos en LaTeX")

st.markdown("Genera un reporte académico formal basado en los resultados del análisis de flujos del restaurante.")

if st.button("Compilar Informe en LaTeX y Generar PDF"):
    with st.spinner("Compilando código LaTeX con pdflatex..."):
        try:
            # Llamar a tu función existente de generación LaTeX
            from src.generador_latex import generar_archivo_tex, compilar_latex_a_pdf
            
            ruta_tex = generar_archivo_tex(
                nombre_archivo="informe_restaurante",
                tema="Minera de Datos Aplicada a Flujos de Video",
                titulo="Analítica de Ocupación en Restaurantes mediante Edge AI",
                resumen="Este informe detalla el proceso KDD aplicado a secuencias de video mediante YOLOv8 y DuckDB.",
                contenido="El análisis demuestra la viabilidad de extraer métricas de ocupación y segmentación de zonas de manera local."
            )
            
            compilar_latex_a_pdf("./outputs/latex_sources", "./outputs/pdfs")
            st.success("¡Informe compilado con éxito!")
            
            pdf_path = "./outputs/pdfs/informe_restaurante.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Descargar Informe PDF", f, file_name="informe_restaurante.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Error al compilar el documento LaTeX: {e}")