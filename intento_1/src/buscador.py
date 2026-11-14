"""
Módulo de Búsqueda y Filtrado (Fases 1 y 2)
Define la literatura y aplica el Principio de Pareto (80/20).
"""

# Literatura recopilada con su respectivo aporte estimado (Principio de Pareto)
LITERATURA_RECOPILADA = [
    {
        "id": "Doc_1",
        "tema": "Minería de Datos y KDD",
        "titulo": "Proceso KDD y Aprendizaje Estadístico en R",
        "resumen": "Este reporte analiza la integración del proceso KDD utilizando lenguajes estadísticos como R.",
        "contenido": "El proceso de Knowledge Discovery in Databases (KDD) consta de fases iterativas que van desde la selección de datos hasta la interpretación de patrones...",
        "tipo_fuente": "Paper académico (R)",
        "aporte_porcentual": 35,
        "archivo_tex": "mineria_kdd_reporte"
    },
    {
        "id": "Doc_2",
        "tema": "CRISP-DM y Modelos Híbridos",
        "titulo": "Metodología CRISP-DM en Arquitecturas Predictivas",
        "resumen": "Estudio sobre la aplicación de CRISP-DM combinada con modelos híbridos de predicción.",
        "contenido": "La metodología estándar interindustrial para minería de datos (CRISP-DM) provee un marco robusto estructurado en seis fases principales...",
        "tipo_fuente": "Guía metodológica",
        "aporte_porcentual": 30,
        "archivo_tex": "crisp_dm_reporte"
    },
    {
        "id": "Doc_3",
        "tema": "Predicción y Data Warehouse",
        "titulo": "Arquitecturas Analíticas y Modelado Predictivo en DW",
        "resumen": "Evaluación de estructuras OLAP y modelos de predicción basados en almacenes de datos.",
        "contenido": "El uso de Data Warehouses combinados con algoritmos de forecasting permite potenciar la toma de decisiones estratégicas...",
        "tipo_fuente": "Artículo técnico",
        "aporte_porcentual": 18,
        "archivo_tex": "dw_prediccion_reporte"
    },
    {
        "id": "Doc_4",
        "tema": "Modelos Híbridos Avanzados",
        "titulo": "Optimización mediante Modelos Híbridos",
        "resumen": "Investigación sobre arquitecturas híbridas avanzadas.",
        "contenido": "Los modelos híbridos combinan técnicas de machine learning y estadística tradicional...",
        "tipo_fuente": "Paper de investigación",
        "aporte_porcentual": 12,
        "archivo_tex": "modelos_hibridos_reporte"
    },
    {
        "id": "Doc_5",
        "tema": "Casos de Estudio Teóricos",
        "titulo": "Fundamentos Teóricos Generales",
        "resumen": "Revisión teórica general.",
        "contenido": "Conceptos introductorios sobre gestión de datos...",
        "tipo_fuente": "Libro general",
        "aporte_porcentual": 5,
        "archivo_tex": "teoria_general_reporte"
    }
]

def filtrar_documentos_pareto(umbral_objetivo=80):
    """
    Filtra los documentos aplicando el Principio de Pareto (80/20) 
    hasta alcanzar el umbral de aporte acumulado deseado.
    """
    documentos_ordenados = sorted(LITERATURA_RECOPILADA, key=lambda x: x["aporte_porcentual"], reverse=True)
    
    acumulado = 0
    seleccionados = []
    
    print("\n[Fase 2] Aplicando Principio de Pareto (80/20)...")
    for doc in documentos_ordenados:
        if acumulado < umbral_objetivo:
            seleccionados.append(doc)
            acumulado += doc["aporte_porcentual"]
            print(f" -> [SELECCIONADO] {doc['id']} ({doc['tema']}) - Aporte: {doc['aporte_porcentual']}% (Acumulado: {acumulado}%)")
        else:
            print(f" -> [DESCARTADO/OPCIONAL] {doc['id']} ({doc['tema']}) - Aporte: {doc['aporte_porcentual']}% (Fuera del 80%)")
            
    return seleccionados