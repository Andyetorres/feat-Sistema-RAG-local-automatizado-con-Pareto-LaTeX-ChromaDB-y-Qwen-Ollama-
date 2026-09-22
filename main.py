"""
Orquestador Principal del Sistema Local Automatizado (RAG + LaTeX + ChromaDB + Pareto + Ollama Qwen)
"""

from src.buscador import filtrar_documentos_pareto
from src.generador_latex import generar_archivo_tex, compilar_latex_a_pdf
from src.procesador import cargar_y_fragmentar_pdfs
from src.vectorizador import inicializar_o_cargar_vector_db
from src.generador_ia import consultar_rag_con_ia

def ejecutar_pipeline():
    print("--- INICIANDO PIPELINE AUTOMATIZADO ---")
    
    # 1 y 2. Búsqueda y Filtrado estricto por Principio de Pareto (80/20)
    documentos_clave = filtrar_documentos_pareto(umbral_objetivo=80)
    print(f"\nTotal documentos seleccionados por Pareto: {len(documentos_clave)}")
    
    # 3. Generación de documentos LaTeX exclusivamente para los seleccionados
    print("\n[Paso 3] Generando documentos fuente en LaTeX (Top Pareto)...")
    for doc in documentos_clave:
        generar_archivo_tex(
            nombre_archivo=doc["archivo_tex"],
            tema=doc["tema"],
            titulo=doc["titulo"],
            resumen=doc["resumen"],
            contenido=doc["contenido"]
        )

    # 4. Compilar archivos .tex a PDF
    print("\n[Paso 4] Compilando LaTeX a PDF...")
    compilar_latex_a_pdf(
        directorio_tex="./outputs/latex_sources", 
        directorio_pdf="./outputs/pdfs"
    )
    
    # 5. Procesar y cargar en ChromaDB (RAG) solo la literatura de alto impacto
    print("\n[Paso 5] Indexando PDFs seleccionados en ChromaDB...")
    chunks = cargar_y_fragmentar_pdfs("./outputs/pdfs")
    
    if chunks:
        retriever = inicializar_o_cargar_vector_db(chunks=chunks, persist_directory="./data/chroma_db")
        
        # 6. Prueba de consulta conversacional con IA Local (Ollama + Qwen)
        print("\n[Paso 6] Consultando al modelo local Qwen con RAG...")
        pregunta = "¿Cómo funciona el proceso KDD en minería de datos?"
        print(f"Consulta del usuario: '{pregunta}'")
        
        respuesta_ia = consultar_rag_con_ia(retriever, pregunta)
        print(f"\n--- RESPUESTA ACADÉMICA DE QWEN --- \n{respuesta_ia}")
    else:
        print("No se encontraron chunks para indexar.")

    print("\n--- PIPELINE FINALIZADO CON ÉXITO ---")

if __name__ == "__main__":
    ejecutar_pipeline()