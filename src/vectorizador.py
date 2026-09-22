"""
Módulo de Vectorización e Indexación en ChromaDB
Almacena los chunks de texto de forma local para consultas semánticas mediante RAG.
"""

from langchain_community.vectorstores import Chroma
# Usamos embeddings de HuggingFace de forma local y gratuita, ideal para entornos sin API Key
from langchain_huggingface import HuggingFaceEmbeddings

def inicializar_o_cargar_vector_db(chunks=None, persist_directory="./data/chroma_db"):
    """
    Inicializa o carga la base de datos vectorial ChromaDB.
    Si se pasan nuevos chunks, los indexa.
    """
    # Modelo ligero y eficiente para ejecución local
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if chunks:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"[ChromaDB] Nuevos documentos indexados y persistidos en '{persist_directory}'.")
    else:
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_embeddings=embeddings if 'embeddings' in locals() else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        print(f"[ChromaDB] Base de datos cargada desde '{persist_directory}'.")
        
    return vector_db.as_retriever(search_kwargs={"k": 3})