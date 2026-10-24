import duckdb
import os
import pandas as pd
import numpy as np

def generar_datos_simulados(db_path="data/restaurante_analytics.duckdb"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    
    # Crear la tabla si no existe
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
    
    # Generar 500 registros simulados de clientes en el restaurante
    # np.random.seed(42)
    timestamps = np.linspace(0, 600, 500) # 10 minutos de video simulado
    track_ids = np.random.randint(1, 15, 500)
    pos_x = np.random.uniform(50, 600, 500)
    pos_y = np.random.uniform(50, 400, 500)
    zonas = ["Zona Mesas Principal" if x > 300 else "Zona Ingreso/Barra" for x in pos_x]
    
    df_simulado = pd.DataFrame({
        'timestamp_segundo': timestamps,
        'track_id': track_ids,
        'clase': 'persona',
        'posicion_x': pos_x,
        'posicion_y': pos_y,
        'zona_detectada': zonas
    })
    
    # Insertar en DuckDB
    con.execute("DELETE FROM hechos_video") # Limpiar datos previos si los hay
    con.register('df_temp', df_simulado)
    con.execute("INSERT INTO hechos_video SELECT * FROM df_temp")
    con.close()
    
    print("[Simulador] Datos analíticos simulados cargados exitosamente en DuckDB.")

if __name__ == "__main__":
    generar_datos_simulados()