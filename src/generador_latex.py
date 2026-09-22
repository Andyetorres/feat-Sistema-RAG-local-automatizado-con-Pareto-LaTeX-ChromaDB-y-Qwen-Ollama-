"""
Módulo de Generación y Compilación LaTeX
Crea plantillas académicas en .tex y automatiza su compilación a PDF usando pdflatex.
"""

import os
import subprocess

PLANTILLA_LATEX_BASE = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}

\title{\textbf{Informe Académico Automatizado: %(titulo)s}}
\author{Sistema Local RAG - IDEAD}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
%(resumen)s
\end{abstract}

\section{Introducción y Contexto}
Este documento ha sido generado automáticamente como parte de la recopilación bibliográfica sobre \textbf{%(tema)s}, integrando motores de búsqueda académica y bases de datos vectoriales locales.

\section{Desarrollo Teórico}
%(contenido)s

\section{Referencias}
\begin{enumerate}
    \item Recopilado mediante cadenas booleanas optimizadas para investigación en ingeniería.
\end{enumerate}

\end{document}
"""

def generar_archivo_tex(nombre_archivo, tema, titulo, resumen, contenido, directorio_salida="./outputs/latex_sources"):
    """Genera un archivo .tex con estructura formal académica."""
    os.makedirs(directorio_salida, exist_ok=True)
    ruta_completa = os.path.join(directorio_salida, f"{nombre_archivo}.tex")
    
    contenido_tex = PLANTILLA_LATEX_BASE % {
        "titulo": titulo,
        "resumen": resumen,
        "tema": tema,
        "contenido": contenido
    }
    
    with open(ruta_completa, "w", encoding="utf-8") as f:
        f.write(contenido_tex)
    
    print(f"[LaTeX] Archivo generado: {ruta_completa}")
    return ruta_completa

def compilar_latex_a_pdf(directorio_tex, directorio_pdf="./outputs/pdfs"):
    """Compila todos los archivos .tex de una carpeta a formato PDF usando pdflatex."""
    os.makedirs(directorio_pdf, exist_ok=True)
    
    if not os.path.exists(directorio_tex):
        print(f"El directorio {directorio_tex} no existe.")
        return

    for archivo in os.listdir(directorio_tex):
        if archivo.endswith(".tex"):
            ruta_completa = os.path.join(directorio_tex, archivo)
            try:
                # Ejecuta pdflatex indicando el directorio de salida de los PDFs (opcional vía argumentos o moviendo el archivo)
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", f"-output-directory={directorio_pdf}", ruta_completa],
                    check=True
                )
                print(f"[PDF] Compilado exitosamente: {archivo}")
            except subprocess.CalledProcessError as e:
                print(f"[Error] Falló la compilación de {archivo}: {e}")