import cv2
import duckdb
import os
from ultralytics import YOLO

def procesar_video_restaurante(video_path, db_path="data/restaurante_analytics.duckdb"):
    """
    Procesa un video frame por frame usando YOLOv8, extrae detecciones de personas,
    asigna una zona analítica y las almacena en DuckDB para minería de datos.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    
    # Tabla estructurada OLAP con la columna de zona integrada
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
    
    print("[Vision AI] Cargando modelo YOLOv8 local...")
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] No se pudo abrir el archivo de video: {video_path}")
        con.close()
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Valor por defecto seguro
        
    frame_count = 0
    skip_frames = 5  # Procesar 1 de cada 5 frames para optimizar tu Lenovo IdeaPad Slim 3
    
    print("[Vision AI] Iniciando procesamiento de fotogramas...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Inferencia con tracking de objetos (clase 0 = persona)
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        timestamp = frame_count / fps

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                x_centro = float((x1 + x2) / 2)
                y_centro = float((y1 + y2) / 2)
                
                # Asignación de zona lógica para minería de datos
                zona = "Zona Mesas Principal" if x_centro > 300 else "Zona Ingreso/Barra"

                con.execute(
                    "INSERT INTO hechos_video VALUES (?, ?, ?, ?, ?, ?)",
                    (float(timestamp), int(track_id), "persona", x_centro, y_centro, zona)
                )

    cap.release()
    con.close()
    print("[Vision AI] Procesamiento de video finalizado y almacenado en DuckDB con éxito.")
    return True