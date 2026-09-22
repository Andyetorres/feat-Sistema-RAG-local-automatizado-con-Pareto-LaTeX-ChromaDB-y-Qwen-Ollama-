"""
Módulo de Generación de Respuesta con IA Local (Ollama + Qwen)
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def consultar_rag_con_ia(retriever, pregunta):
    """
    Usa Qwen via Ollama para generar una respuesta basada estrictamente 
    en los documentos recuperados de ChromaDB.
    """
    print(f"\n[IA Local] Consultando al modelo Qwen (qwen2.5:3b)...")
    
    # Inicializar el modelo local ligero (ideal para 8GB de RAM)
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
    
    # Prompt estructurado para guiar al modelo
    template = """Eres un asistente de investigación académico. Utiliza el siguiente contexto recuperado 
    para responder la pregunta del usuario de manera clara, profesional y basada en los documentos.
    Si no sabes la respuesta o no está en el contexto, dilo honestamente.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta académica:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Función auxiliar para formatear los documentos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construir la cadena RAG con LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ejecutar la cadena
    respuesta = rag_chain.invoke(pregunta)
    return respuesta