import base64
from datetime import datetime
import os
import shutil
import subprocess
import pandas as pd
import streamlit as st

PLANTILLA_LATEX_RESTAURANTE = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage{geometry}
\geometry{margin=2.2cm}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}

\title{\textbf{Informe Técnico de Analítica y Minería de Datos:\\ <<TITULO>>}}
\author{Sistema Inteligente de Monitoreo en Tiempo Real (Visión + Data Warehouse)}
\date{<<FECHA>>}

\begin{document}

\maketitle

\begin{abstract}
<<RESUMEN>>
\end{abstract}

\section{Enfoque Principal del Análisis}
<<SECCION_ENFOQUE>>

\section{Arquitectura del Sistema (Visión Artificial y Data Warehouse)}
El monitoreo se ejecutó mediante detección por visión artificial y rastreo determinista de entidades en salón, alimentando un \textbf{Data Warehouse dimensional en DuckDB} compuesto por la tabla de hechos \texttt{fact\_servicio\_live} (<<TOTAL_EVENTOS>> registros capturados) y las dimensiones \texttt{dim\_mesas} y \texttt{fact\_visitas\_meseros}.

\begin{table}[H]
\centering
\caption{Indicadores Globales Consolidados en el Data Warehouse}
\begin{tabular}{lr}
\toprule
\textbf{Indicador Operativo} & \textbf{Valor Registrado} \\
\midrule
Total de clientes atendidos hoy (con plato servido) & <<CLIENTES_HOY>> \\
Clientes activos en el último corte & <<CLIENTES_SALON>> \\
Mesas ocupadas en el último corte & <<MESAS_OCUPADAS>> de <<TOTAL_MESAS>> \\
Total de platos servidos detectados & <<PLATOS_ACTUALES>> \\
Personal activo (Chefs / Meseros) & <<CHEFS>> Chefs / <<MESEROS>> Meseros \\
\bottomrule
\end{tabular}
\end{table}

\section{Desglose de Ocupación por Mesa (A1 -- B2)}
<<TABLA_MESAS>>

\section{Productividad y Tiempos de Atención por Mesero}
<<TABLA_MESEROS>>

\section{Minería de Datos (Metodologías KDD, CRISP-DM y Modelo Híbrido)}
Siguiendo las fases de \textbf{KDD} (Selección, Preprocesamiento, Transformación, Minería y Evaluación) y \textbf{CRISP-DM}, se aplicó segmentación no supervisada mediante \textit{K-Means} y un \textbf{Modelo Híbrido Predictivo} (Árbol de Decisión + Red Neuronal Perceptrón Multicapa MLP) para anticipar saturación en el comedor:
<<SECCION_MINERIA>>

\section{Conclusiones y Recomendaciones Operativas}
<<CONCLUSIONES>>

\end{document}
"""


def escapar_latex(texto: str) -> str:
    """Escapa caracteres especiales de LaTeX para evitar errores de compilación."""
    if texto is None:
        return ""
    s = str(texto)
    reemplazos = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for viejo, nuevo in reemplazos:
        s = s.replace(viejo, nuevo)
    return s


def _construir_tabla_mesas_latex(df_mesas: pd.DataFrame) -> str:
    if df_mesas.empty:
        return "No se registraron datos de mesas en la sesión."

    filas = []
    for _, r in df_mesas.iterrows():
        mesa = escapar_latex(r["mesa_id"])
        sillas = int(r["cantidad_sillas"])
        veces = int(r["veces_ocupada"])
        cli = int(r["clientes_acumulados"])
        visitas = int(r["visitas_mesero"])
        t_prom = float(r["tiempo_promedio_atencion_seg"])
        filas.append(f"{mesa} & {sillas} & {veces} & {cli} & {visitas} & {t_prom:.2f} s \\\\")

    cuerpo = "\n".join(filas)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Rotación, Clientes Atendidos y Tiempo de Servicio por Mesa}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "\\textbf{Mesa} & \\textbf{Sillas} & \\textbf{Veces Ocupada} & \\textbf{Clientes} & \\textbf{Atenciones} & \\textbf{Tiempo Prom.} \\\\\n"
        "\\midrule\n"
        f"{cuerpo}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


def _construir_tabla_meseros_latex(df_meseros: pd.DataFrame) -> str:
    if df_meseros.empty:
        return "Aún no se han cerrado ciclos completos de atención detenidos en mesa ($\\ge 1.2$ s)."

    filas = []
    for _, r in df_meseros.iterrows():
        mesero = escapar_latex(r["Mesero"])
        atendidas = int(r["Mesas_Atendidas"])
        distintas = int(r["Mesas_Distintas"])
        t_prom = float(r["Tiempo_Promedio_Seg"])
        t_tot = float(r["Tiempo_Total_Seg"])
        filas.append(f"{mesero} & {atendidas} & {distintas} & {t_prom:.2f} s & {t_tot:.1f} s \\\\")

    cuerpo = "\n".join(filas)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Rendimiento y Tiempos Promedio del Personal de Salón}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "\\textbf{Mesero} & \\textbf{Atenciones} & \\textbf{Mesas Distintas} & \\textbf{Tiempo Prom.} & \\textbf{Tiempo Total} \\\\\n"
        "\\midrule\n"
        f"{cuerpo}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


def generar_informe_latex(
    df_hechos: pd.DataFrame,
    df_mesas: pd.DataFrame,
    df_meseros: pd.DataFrame,
    enfoque_usuario: str,
    directorio_tex: str = "intento_2/outputs/latex_sources",
    directorio_pdf: str = "intento_2/outputs/pdfs",
) -> tuple[str, str | None, str]:
    """
    Construye el documento .tex adaptado al enfoque pedido por el usuario,
    compila a PDF usando pdflatex y limpia archivos auxiliares (.aux, .log, .out).
    Retorna: (ruta_tex, ruta_pdf_o_None, contenido_tex)
    """
    os.makedirs(directorio_tex, exist_ok=True)
    os.makedirs(directorio_pdf, exist_ok=True)

    row_ult = df_hechos.iloc[-1]
    clientes_hoy = int(row_ult["clientes_acumulados_dia"])
    clientes_salon = int(row_ult["clientes_en_salon"])
    mesas_oc = int(row_ult["mesas_ocupadas"])
    total_mesas = int(row_ult["total_mesas"])
    platos_act = int(row_ult["platos_comida"])
    chefs = int(row_ult["chefs"])
    meseros = int(row_ult["meseros"])
    tasa_oc_prom = round(float(df_hechos["tasa_ocupacion"].mean()), 1)

    mesa_lider = "N/A"
    if not df_mesas.empty:
        top_m = df_mesas.sort_values(by=["clientes_acumulados", "veces_ocupada"], ascending=False).iloc[0]
        mesa_lider = f"{top_m['mesa_id']} ({int(top_m['clientes_acumulados'])} clientes, {int(top_m['veces_ocupada'])} ocupaciones)"

    mesero_lider = "N/A"
    if not df_meseros.empty:
        top_w = df_meseros.iloc[0]
        mesero_lider = f"{top_w['Mesero']} ({int(top_w['Mesas_Atendidas'])} atenciones, {top_w['Tiempo_Promedio_Seg']}s promedio)"

    enfoque_limpio = enfoque_usuario.strip()
    enf_lower = enfoque_limpio.lower()

    # Personalizar el título, resumen y sección destacada según lo que pida el usuario
    if enfoque_limpio and any(k in enf_lower for k in ["cliente", "dia", "día", "afluencia", "flujo"]):
        titulo = "Análisis de Afluencia y Flujo de Clientes por Jornada"
        seccion_enfoque = (
            f"De acuerdo con la solicitud específica del usuario (\\textit{{{escapar_latex(enfoque_limpio)}}}), "
            f"el sistema contabilizó un acumulado de \\textbf{{{clientes_hoy} clientes atendidos}} con servicio de plato confirmado en mesa. "
            f"La tasa promedio de ocupación durante la ventana de observación fue del \\textbf{{{tasa_oc_prom}\\%}}, "
            f"registrando un pico simultáneo de \\textbf{{{int(df_hechos['clientes_en_salon'].max())} clientes}} en salón."
        )
    elif enfoque_limpio and any(k in enf_lower for k in ["mesa", "ocup", "silla", "a1", "a2", "a3", "b1", "b2"]):
        titulo = "Estudio de Rotación y Ocupación por Mesa"
        seccion_enfoque = (
            f"En respuesta al enfoque solicitado (\\textit{{{escapar_latex(enfoque_limpio)}}}), "
            f"la mesa con mayor demanda y rotación fue \\textbf{{{escapar_latex(mesa_lider)}}}. "
            f"Las mesas de mayor capacidad (B1 y B2 de 6 puestos) concentran el mayor volumen de comensales por ciclo de servicio."
        )
    elif enfoque_limpio and any(k in enf_lower for k in ["mesero", "personal", "tiempo", "rapidez", "atencion", "atención"]):
        titulo = "Evaluación de Productividad y Tiempos de Atención del Personal"
        seccion_enfoque = (
            f"En respuesta al enfoque solicitado (\\textit{{{escapar_latex(enfoque_limpio)}}}), "
            f"el integrante con mayor carga operativa fue \\textbf{{{escapar_latex(mesero_lider)}}}. "
            f"Se filtraron únicamente paradas efectivas de servicio ($\\ge 1.2$ segundos junto a mesa) para garantizar precisión."
        )
    else:
        titulo = "Reporte Ejecutivo Integral de Operación y Minería de Datos"
        seccion_enfoque = (
            "Al no restringirse a una única variable, este informe sintetiza los hallazgos principales del sistema: "
            f"se atendieron \\textbf{{{clientes_hoy} clientes}}, la mesa de mayor rotación fue \\textbf{{{escapar_latex(mesa_lider)}}}, "
            f"y el desempeño destacado del personal correspondió a \\textbf{{{escapar_latex(mesero_lider)}}}."
        )

    resumen = (
        f"Este informe presenta los resultados obtenidos tras analizar {len(df_hechos)} eventos en tiempo real "
        f"mediante visión artificial, almacenamiento en Data Warehouse (DuckDB) y modelos de Minería de Datos. "
        f"Se consolidó un total de {clientes_hoy} clientes atendidos y una ocupación media del {tasa_oc_prom}\\%."
    )

    seccion_mineria = (
        f"\\begin{{itemize}}\n"
        f"\\item \\textbf{{Ocupación Media del Salón:}} {tasa_oc_prom}\\% de capacidad instalada.\n"
        f"\\item \\textbf{{Mesa Líder en Demanda:}} {escapar_latex(mesa_lider)}.\n"
        f"\\item \\textbf{{Mesero Líder en Servicio:}} {escapar_latex(mesero_lider)}.\n"
        f"\\item \\textbf{{Modelo Híbrido (Árbol + MLP):}} Evalúa la carga por mesero y variación de flujo para emitir alertas tempranas de saturación.\n"
        f"\\end{{itemize}}"
    )

    conclusiones = (
        f"El pipeline automatizado permitió transformar capturas visuales en inteligencia de negocio estructurada. "
        f"Se recomienda priorizar la reposición rápida en \\textbf{{{escapar_latex(mesa_lider.split(' ')[0])}}} "
        f"y reforzar el salón cuando la ocupación supere el 60\\% para mantener tiempos de atención óptimos."
    )

    contenido_tex = PLANTILLA_LATEX_RESTAURANTE
    reemplazos_plantilla = {
        "<<TITULO>>": titulo,
        "<<FECHA>>": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "<<RESUMEN>>": resumen,
        "<<SECCION_ENFOQUE>>": seccion_enfoque,
        "<<TOTAL_EVENTOS>>": str(len(df_hechos)),
        "<<CLIENTES_HOY>>": str(clientes_hoy),
        "<<CLIENTES_SALON>>": str(clientes_salon),
        "<<MESAS_OCUPADAS>>": str(mesas_oc),
        "<<TOTAL_MESAS>>": str(total_mesas),
        "<<PLATOS_ACTUALES>>": str(platos_act),
        "<<CHEFS>>": str(chefs),
        "<<MESEROS>>": str(meseros),
        "<<TABLA_MESAS>>": _construir_tabla_mesas_latex(df_mesas),
        "<<TABLA_MESEROS>>": _construir_tabla_meseros_latex(df_meseros),
        "<<SECCION_MINERIA>>": seccion_mineria,
        "<<CONCLUSIONES>>": conclusiones,
    }
    for marcador, valor in reemplazos_plantilla.items():
        contenido_tex = contenido_tex.replace(marcador, valor)

    nombre_base = f"informe_restaurante_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ruta_tex = os.path.join(directorio_tex, f"{nombre_base}.tex")
    with open(ruta_tex, "w", encoding="utf-8") as f:
        f.write(contenido_tex)

    ruta_pdf = None
    if shutil.which("pdflatex"):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", f"-output-directory={directorio_pdf}", ruta_tex],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            candidato_pdf = os.path.join(directorio_pdf, f"{nombre_base}.pdf")
            if os.path.exists(candidato_pdf):
                ruta_pdf = candidato_pdf

            # Limpieza automática de archivos auxiliares de LaTeX
            for ext in [".aux", ".log", ".out", ".toc", ".synctex.gz"]:
                for d in [directorio_pdf, directorio_tex]:
                    aux_file = os.path.join(d, nombre_base + ext)
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
        except subprocess.CalledProcessError:
            ruta_pdf = None

    return ruta_tex, ruta_pdf, contenido_tex


def renderizar_seccion_latex(analizador):
    """Interfaz siempre visible para escribir el enfoque, generar, visualizar y descargar el informe LaTeX."""
    st.divider()
    st.header("📄 Capa 4: Documentación Académica Automatizada (LaTeX / PDF)")
    st.caption("Pausa la captura en vivo (⏸️ Detener Captura) cuando quieras redactar tu enfoque y generar el documento.")

    with st.form(key="form_generador_latex"):
        col_enf, col_btn = st.columns([2.2, 1])
        with col_enf:
            enfoque_latex = st.text_input(
                "¿Qué tema quieres que sobresalga en el informe LaTeX? (Déjalo vacío para resumen general):",
                placeholder="Ej: clientes por día / qué mesas se ocuparon más / desempeño de los meseros",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generar_click = st.form_submit_button("📄 Realizar informe tipo LaTeX")

    if generar_click:
        df_hechos, df_mesas, df_meseros = analizador.obtener_datasets_dw()
        if df_hechos.empty:
            st.warning("⚠️ Aún no hay datos capturados en el Data Warehouse. Inicia la captura en vivo por unos segundos primero.")
        else:
            with st.spinner("Generando código LaTeX, tabulando Data Warehouse y compilando PDF..."):
                ruta_tex, ruta_pdf, codigo_tex = generar_informe_latex(
                    df_hechos, df_mesas, df_meseros, enfoque_latex
                )
                st.session_state.ultimo_reporte_latex = {
                    "ruta_tex": ruta_tex,
                    "ruta_pdf": ruta_pdf,
                    "codigo_tex": codigo_tex,
                }

    if "ultimo_reporte_latex" in st.session_state:
        rep = st.session_state.ultimo_reporte_latex
        ruta_pdf = rep["ruta_pdf"]
        ruta_tex = rep["ruta_tex"]
        codigo_tex = rep["codigo_tex"]

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            if ruta_pdf and os.path.exists(ruta_pdf):
                with open(ruta_pdf, "rb") as f_pdf:
                    st.download_button(
                        label="⬇️ Descargar Informe Compilado (.PDF)",
                        data=f_pdf.read(),
                        file_name=os.path.basename(ruta_pdf),
                        mime="application/pdf",
                        key="dl_pdf_latex",
                    )
            else:
                st.warning("⚠️ `pdflatex` no está instalado en el sistema; puedes descargar el `.tex` compilable.")

        with c_dl2:
            st.download_button(
                label="⬇️ Descargar Código Fuente (.TEX)",
                data=codigo_tex.encode("utf-8"),
                file_name=os.path.basename(ruta_tex),
                mime="application/x-tex",
                key="dl_tex_source",
            )

        tab_vista_pdf, tab_codigo_tex = st.tabs(["👁️ Visualizar Documento PDF", "💻 Ver Código LaTeX (.tex)"])
        with tab_vista_pdf:
            if ruta_pdf and os.path.exists(ruta_pdf):
                with open(ruta_pdf, "rb") as f_pdf:
                    base64_pdf = base64.b64encode(f_pdf.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.info("Vista previa PDF no disponible sin `pdflatex`. Revisa la pestaña **Ver Código LaTeX (.tex)** o instala `texlive-latex-base`.")

        with tab_codigo_tex:
            st.code(codigo_tex, language="latex")