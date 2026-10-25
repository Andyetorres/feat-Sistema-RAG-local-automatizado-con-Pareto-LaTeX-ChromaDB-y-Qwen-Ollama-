"""
Módulo de Generación y Compilación LaTeX
Crea plantillas académicas en .tex y automatiza su compilación a PDF usando pdflatex 
de forma específica y limpia (sin saturar el almacenamiento local).
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

def compilar_latex_a_pdf(directorio_tex, directorio_pdf="./outputs/pdfs", nombre_archivo=None):
    """
    Compila un archivo .tex específico (o todos si no se especifica) a formato PDF 
    y limpia automáticamente los archivos auxiliares generados (logs, aux, out).
    """
    os.makedirs(directorio_pdf, exist_ok=True)
    
    if not os.path.exists(directorio_tex):
        print(f"El directorio {directorio_tex} no existe.")
        return

    # Si se pasa un nombre de archivo específico, solo compilamos ese. Si no, filtramos los .tex
    if nombre_archivo:
        archivos_a_compilar = [f"{nombre_archivo}.tex" if not nombre_archivo.endswith(".tex") else nombre_archivo]
    else:
        archivos_a_compilar = [f for f in os.listdir(directorio_tex) if f.endswith(".tex")]

    for archivo in archivos_a_compilar:
        ruta_completa = os.path.join(directorio_tex, archivo)
        
        if not os.path.exists(ruta_completa):
            print(f"El archivo {archivo} no se encuentra en {directorio_tex}.")
            continue
            
        try:
            # Ejecuta pdflatex dirigiendo la salida PDF al directorio correspondiente
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", f"-output-directory={directorio_pdf}", ruta_completa],
                check=True
            )
            print(f"[PDF] Compilado exitosamente: {archivo}")
            
            # --- LIMPIEZA AUTOMÁTICA DE ARCHIVOS AUXILIARES ---
            # pdflatex genera archivos como .aux, .log, .out que ensucian las carpetas. Los borramos.
            nombre_base = os.path.splitext(archivo)[0]
            extensiones_basura = [".aux", ".log", ".out", ".toc", ".synctex.gz"]
            
            for ext in extensiones_basura:
                for directorio in [directorio_pdf, directorio_tex]:
                    ruta_aux = os.path.join(directorio, nombre_base + ext)
                    if os.path.exists(ruta_aux):
                        os.remove(ruta_aux)
            
            print(f"[Limpieza] Archivos temporales de '{nombre_base}' eliminados correctamente.")

        except subprocess.CalledProcessError as e:
            print(f"[Error] Falló la compilación de {archivo}: {e}")