"""
Módulo de Procesamiento de Textos y Chunking
Utiliza LangChain para fragmentar la literatura recopilada en chunks óptimos para IA.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

def cargar_y_fragmentar_pdfs(carpeta_pdfs):
    """Carga los PDFs de un directorio y los divide en chunks estructurados."""
    if not os.path.exists(carpeta_pdfs):
        print(f"Carpeta de PDFs no encontrada: {carpeta_pdfs}")
        return []

    loader = PyPDFDirectoryLoader(carpeta_pdfs)
    documentos = loader.load()
    
    # Fragmentación de texto optimizada para modelos de lenguaje
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"[Procesador] Se generaron {len(chunks)} fragmentos (chunks) listos para vectorizar.")
    return chunks