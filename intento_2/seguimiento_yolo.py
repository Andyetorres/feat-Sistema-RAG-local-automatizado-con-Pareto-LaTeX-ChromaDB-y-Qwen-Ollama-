from datetime import datetime
import cv2
import easyocr
import numpy as np
import streamlit as st


@st.cache_resource
def cargar_lector_ocr():
    return easyocr.Reader(["es", "en"], gpu=False)


def mascara_color_rgb(img_rgb: np.ndarray, color_rgb: tuple[int, int, int], tolerancia: int = 16) -> np.ndarray:
    c = np.array(color_rgb, dtype=np.int16)
    lower = np.clip(c - tolerancia, 0, 255).astype(np.uint8)
    upper = np.clip(c + tolerancia, 0, 255).astype(np.uint8)
    return cv2.inRange(img_rgb, lower, upper)


class DetectorSimulador:
    COLOR_MESA_LIBRE = (139, 69, 19)
    COLOR_MESA_OCUPADA = (180, 80, 50)
    COLOR_CAJA = (46, 204, 113)
    COLOR_COCINA = (214, 162, 108)

    TONOS_PIEL = [
        (255, 224, 189),
        (241, 194, 125),
        (224, 172, 105),
        (198, 134, 66),
        (141, 85, 36),
    ]

    ROPA_CLIENTES = [
        (235, 77, 75),
        (9, 132, 227),
        (0, 184, 148),
        (253, 203, 110),
        (108, 92, 231),
    ]

    TRAJE_MESERO = (30, 30, 30)
    TRAJE_CAJERO = (45, 52, 54)

    def __init__(self):
        self.frame_count = 0
        self.cache_ocr = []
        self.zonas_estaticas = {}
        self.meseros_pool = {}
        self.max_meseros_reales = 0

    def _asignar_ids_meseros(self, posiciones_actuales: list[tuple[int, int]], max_dist: float) -> dict[str, tuple[int, int]]:
        """
        Mantiene estrictamente los 2 meseros reales del simulador (Mesero_1 y Mesero_2).
        """
        MAX_MESEROS_SIMULADOR = 2
        if not hasattr(self, "meseros_pool"):
            self.meseros_pool = {}

        # Quedarse como máximo con las 2 detecciones más separadas entre sí
        posiciones_filtradas = []
        for pos in posiciones_actuales:
            if all(np.hypot(pos[0] - p[0], pos[1] - p[1]) > 14 for p in posiciones_filtradas):
                posiciones_filtradas.append(pos)
            if len(posiciones_filtradas) == MAX_MESEROS_SIMULADOR:
                break

        asignados = {}
        usados = set()

        # 1. Emparejar por cercanía con Mesero_1 / Mesero_2
        for pos in posiciones_filtradas:
            mejor_id = None
            mejor_dist = max_dist * 2.5
            for mid, info in self.meseros_pool.items():
                if mid in usados:
                    continue
                d = np.hypot(pos[0] - info["pos"][0], pos[1] - info["pos"][1])
                if d < mejor_dist:
                    mejor_dist = d
                    mejor_id = mid

            if mejor_id is not None:
                self.meseros_pool[mejor_id] = {"pos": pos, "frames_perdido": 0}
                asignados[mejor_id] = pos
                usados.add(mejor_id)

        # 2. Reutilizar slot libre o crear hasta máximo Mesero_2
        for pos in posiciones_filtradas:
            if pos in asignados.values():
                continue
            libres = [mid for mid in self.meseros_pool if mid not in usados]
            if libres:
                reutilizado = max(libres, key=lambda m: self.meseros_pool[m]["frames_perdido"])
                self.meseros_pool[reutilizado] = {"pos": pos, "frames_perdido": 0}
                asignados[reutilizado] = pos
                usados.add(reutilizado)
            elif len(self.meseros_pool) < MAX_MESEROS_SIMULADOR:
                nuevo_id = f"Mesero_{len(self.meseros_pool) + 1}"
                self.meseros_pool[nuevo_id] = {"pos": pos, "frames_perdido": 0}
                asignados[nuevo_id] = pos
                usados.add(nuevo_id)

        for mid in self.meseros_pool:
            if mid not in usados:
                self.meseros_pool[mid]["frames_perdido"] += 1

        return asignados

    def procesar_frame(self, frame_bgr: np.ndarray, ocr_reader) -> tuple[np.ndarray, dict]:
        self.frame_count += 1
        frame_anotado = frame_bgr.copy()
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]
        area_total = float(h_img * w_img)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.frame_count == 1 or self.frame_count % 35 == 0:
            self.cache_ocr = ocr_reader.readtext(frame_bgr, detail=1)
            self._actualizar_zonas_estaticas(self.cache_ocr)

        for nombre_zona, (zx1, zy1, zx2, zy2) in self.zonas_estaticas.items():
            cv2.rectangle(frame_anotado, (zx1 - 3, zy1 - 3), (zx2 + 3, zy2 + 3), (255, 200, 0), 2)
            cv2.putText(frame_anotado, nombre_zona, (zx1, max(15, zy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 2)

        # 1. DETECCIÓN DETERMINISTA DE LAS 5 MESAS (A1..A3 arriba, B1..B2 abajo)
        mask_m_libre = mascara_color_rgb(img_rgb, self.COLOR_MESA_LIBRE, tolerancia=14)
        mask_m_ocupada = mascara_color_rgb(img_rgb, self.COLOR_MESA_OCUPADA, tolerancia=14)
        mask_mesas = cv2.bitwise_or(mask_m_libre, mask_m_ocupada)

        kernel_mesa = np.ones((7, 7), np.uint8)
        mask_mesas_solida = cv2.morphologyEx(mask_mesas, cv2.MORPH_CLOSE, kernel_mesa)
        contornos_mesas, _ = cv2.findContours(mask_mesas_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mesas_raw = []
        for cnt in contornos_mesas:
            area = cv2.contourArea(cnt)
            ratio_area = area / area_total
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            if 0.0025 <= ratio_area <= 0.045 and 0.85 <= aspect_ratio <= 2.6:
                pix_ocupada = cv2.countNonZero(mask_m_ocupada[y:y + h, x:x + w])
                esta_ocupada = pix_ocupada > (area * 0.25)
                mesas_raw.append({
                    "bbox": (x, y, x + w, y + h),
                    "centro": (x + w // 2, y + h // 2),
                    "ocupada": esta_ocupada,
                })

        # Ordenar por fila (arriba vs abajo) y luego de izquierda a derecha
        mesas_raw.sort(key=lambda m: (m["centro"][1] // max(1, int(h_img * 0.18)), m["centro"][0]))

        fila_superior = [m for m in mesas_raw if m["centro"][1] < (h_img * 0.48)]
        fila_inferior = [m for m in mesas_raw if m["centro"][1] >= (h_img * 0.48)]
        fila_superior.sort(key=lambda m: m["centro"][0])
        fila_inferior.sort(key=lambda m: m["centro"][0])

        mesas_finales = []
        for idx, m in enumerate(fila_superior, start=1):
            mx1, my1, mx2, my2 = m["bbox"]
            etiqueta = f"A{idx} (4p)"
            mesas_finales.append({
                "id": etiqueta,
                "bbox": (mx1, my1, mx2, my2),
                "centro": m["centro"],
                "sillas": 4,
                "ocupada": m["ocupada"],
                "platos": 0,
            })

        for idx, m in enumerate(fila_inferior, start=1):
            mx1, my1, mx2, my2 = m["bbox"]
            etiqueta = f"B{idx} (6p)"
            mesas_finales.append({
                "id": etiqueta,
                "bbox": (mx1, my1, mx2, my2),
                "centro": m["centro"],
                "sillas": 6,
                "ocupada": m["ocupada"],
                "platos": 0,
            })

        for mesa in mesas_finales:
            mx1, my1, mx2, my2 = mesa["bbox"]
            estado_txt = "OCUPADA" if mesa["ocupada"] else "LIBRE"
            color_borde = (0, 140, 255) if mesa["ocupada"] else (0, 255, 0)
            cv2.rectangle(frame_anotado, (mx1, my1), (mx2, my2), color_borde, 2)
            cv2.putText(
                frame_anotado,
                f"{mesa['id']} [{estado_txt}]",
                (mx1, my1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color_borde,
                2,
            )

        # Máscaras de color
        mask_piel_total = np.zeros((h_img, w_img), dtype=np.uint8)
        for tono in self.TONOS_PIEL:
            mask_piel_total = cv2.bitwise_or(mask_piel_total, mascara_color_rgb(img_rgb, tono, tolerancia=10))

        mask_caja = mascara_color_rgb(img_rgb, self.COLOR_CAJA, tolerancia=15)
        mask_cocina = mascara_color_rgb(img_rgb, self.COLOR_COCINA, tolerancia=15)
        mask_traje_mesero = mascara_color_rgb(img_rgb, self.TRAJE_MESERO, tolerancia=12)
        mask_traje_cajero = mascara_color_rgb(img_rgb, self.TRAJE_CAJERO, tolerancia=12)
        mask_blanco = mascara_color_rgb(img_rgb, (250, 250, 250), tolerancia=18)

        mask_ropa_clientes = np.zeros((h_img, w_img), dtype=np.uint8)
        for col_ropa in self.ROPA_CLIENTES:
            mask_ropa_clientes = cv2.bitwise_or(mask_ropa_clientes, mascara_color_rgb(img_rgb, col_ropa, tolerancia=16))

        # 2. CHEFS EN COCINA
        chefs = []
        kernel_cocina = np.ones((11, 11), np.uint8)
        mask_cocina_solida = cv2.morphologyEx(mask_cocina, cv2.MORPH_CLOSE, kernel_cocina)
        contornos_cocina, _ = cv2.findContours(mask_cocina_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox_cocina = None
        if contornos_cocina:
            cnt_max_cocina = max(contornos_cocina, key=cv2.contourArea)
            if cv2.contourArea(cnt_max_cocina) > (area_total * 0.02):
                kx, ky, kw, kh = cv2.boundingRect(cnt_max_cocina)
                bbox_cocina = (kx, ky, kx + kw, ky + kh)

                roi_blanco_cocina = mask_blanco[ky:ky + kh, kx:kx + kw]
                kernel_chef = np.ones((14, 14), np.uint8)
                roi_chefs_unidos = cv2.morphologyEx(roi_blanco_cocina, cv2.MORPH_CLOSE, kernel_chef)
                contornos_chefs, _ = cv2.findContours(roi_chefs_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                candidatos_chefs = []
                for c_chef in contornos_chefs:
                    area_c = cv2.contourArea(c_chef)
                    cx_rel, cy_rel, cw, ch = cv2.boundingRect(c_chef)
                    if area_c >= 45 and cw >= 6 and ch >= 10:
                        abs_x = kx + cx_rel
                        abs_y = ky + cy_rel
                        centro_c = (abs_x + cw // 2, abs_y + ch // 2)
                        candidatos_chefs.append((centro_c, (abs_x, abs_y, cw, ch)))

                distancia_min_chef = max(18, int(w_img * 0.025))
                for centro_c, (abs_x, abs_y, cw, ch) in candidatos_chefs:
                    if not any(np.hypot(centro_c[0] - ex, centro_c[1] - ey) < distancia_min_chef for (ex, ey) in chefs):
                        chefs.append(centro_c)
                        cv2.rectangle(frame_anotado, (abs_x - 3, abs_y - 3), (abs_x + cw + 3, abs_y + ch + 3), (255, 255, 255), 2)
                        cv2.putText(frame_anotado, "Chef", (abs_x - 4, abs_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

        # 3. MESEROS, CLIENTES Y CAJERO EN SALÓN
        mask_piel_clara = mascara_color_rgb(img_rgb, (255, 224, 189), tolerancia=8)
        contornos_piel, _ = cv2.findContours(mask_piel_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meseros_pos = []
        clientes = []
        cajeros = []

        for cnt in contornos_piel:
            area = cv2.contourArea(cnt)
            px, py, pw, ph = cv2.boundingRect(cnt)
            if 35 <= area <= 350 and 6 <= pw <= 24 and 6 <= ph <= 24:
                cx, cy = px + pw // 2, py + ph // 2
                if bbox_cocina:
                    kx1, ky1, kx2, ky2 = bbox_cocina
                    if (kx1 - 5) <= cx <= (kx2 + 5) and (ky1 - 5) <= cy <= (ky2 + 5):
                        continue

                y1_box, y2_box = max(0, py - 8), min(h_img, py + ph + 16)
                x1_box, x2_box = max(0, px - 6), min(w_img, px + pw + 6)

                roi_caja = cv2.countNonZero(mask_caja[y1_box:y2_box, x1_box:x2_box])
                roi_mesero_negro = cv2.countNonZero(mask_traje_mesero[y1_box:y2_box, x1_box:x2_box])
                roi_camisa_blanca = cv2.countNonZero(mask_blanco[y1_box:y2_box, x1_box:x2_box])
                roi_piel_clara = cv2.countNonZero(mask_piel_clara[py:py + ph, px:px + pw])
                roi_cajero = cv2.countNonZero(mask_traje_cajero[y1_box:y2_box, x1_box:x2_box])
                roi_ropa_cli = cv2.countNonZero(mask_ropa_clientes[y1_box:y2_box, x1_box:x2_box])

                # 1. Cajero: sobre zona verde de caja + traje gris oscuro + corbata azul
                if roi_caja > 35 and (roi_cajero > 12 or roi_ropa_cli > 8):
                    cajeros.append((cx, cy))
                    cv2.rectangle(frame_anotado, (px - 4, py - 5), (px + pw + 4, py + ph + 12), (0, 255, 127), 2)
                    cv2.putText(frame_anotado, "Cajero", (px - 6, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 127), 1)
                # 2. Mesero: obligatoriamente piel clara + traje negro (30,30,30) + camisa blanca (240,240,240)
                elif roi_piel_clara >= 25 and roi_mesero_negro >= 25 and roi_camisa_blanca >= 15 and roi_caja == 0:
                    meseros_pos.append((cx, cy))
                # 3. Cliente: ropa de color de cliente (rojo, azul, verde agua, amarillo o morado)
                elif roi_ropa_cli >= 15 and roi_caja == 0:
                    clientes.append((cx, cy))
                    cv2.rectangle(frame_anotado, (px - 4, py - 5), (px + pw + 4, py + ph + 12), (255, 0, 255), 2)
                    cv2.putText(frame_anotado, "Cliente", (px - 6, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 0, 255), 1)

        meseros_dict = self._asignar_ids_meseros(meseros_pos, max_dist=w_img * 0.12)
        for mid, (mx, my) in meseros_dict.items():
            cv2.rectangle(frame_anotado, (mx - 10, my - 10), (mx + 10, my + 14), (0, 165, 255), 2)
            cv2.putText(frame_anotado, mid, (mx - 16, my - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 165, 255), 1)

        # 4. PLATOS DE COMIDA EXACTOS POR MESA
        mask_comida_naranja = mascara_color_rgb(img_rgb, (220, 100, 40), tolerancia=18)
        contornos_comida, _ = cv2.findContours(mask_comida_naranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        platos_comida = 0

        for cnt in contornos_comida:
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if 10 <= area <= 140 and 3 <= bw <= 16 and 3 <= bh <= 16:
                bcx, bcy = bx + bw // 2, by + bh // 2
                y1_p, y2_p = max(0, by - 5), min(h_img, by + bh + 5)
                x1_p, x2_p = max(0, bx - 5), min(w_img, bx + bw + 5)

                if cv2.countNonZero(mask_blanco[y1_p:y2_p, x1_p:x2_p]) >= 12:
                    for mesa in mesas_finales:
                        mx1, my1, mx2, my2 = mesa["bbox"]
                        if (mx1 - 4) <= bcx <= (mx2 + 4) and (my1 - 4) <= bcy <= (my2 + 4):
                            mesa["platos"] += 1
                            platos_comida += 1
                            cv2.circle(frame_anotado, (bcx, bcy), 7, (0, 255, 255), 2)
                            break

        # 5. PROXIMIDAD DE MESEROS A MESAS (Solo la mesa más cercana dentro del radio de servicio)
        umbral_visita = w_img * 0.085
        visitas_actuales = []
        for mid, (mx, my) in meseros_dict.items():
            mejor_mesa = None
            mejor_dist = umbral_visita
            for mesa in mesas_finales:
                mcx, mcy = mesa["centro"]
                d = np.hypot(mx - mcx, my - mcy)
                if d < mejor_dist:
                    mejor_dist = d
                    mejor_mesa = mesa
            if mejor_mesa is not None:
                visitas_actuales.append({"mesero_id": mid, "mesa": mejor_mesa["id"]})
                cv2.line(frame_anotado, (mx, my), mejor_mesa["centro"], (0, 255, 255), 2)

        frame_rgb_out = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)

        datos_frame = {
            "timestamp": timestamp,
            "zonas": list(self.zonas_estaticas.keys()),
            "mesas": mesas_finales,
            "total_mesas": len(mesas_finales),
            "total_sillas": sum(m["sillas"] for m in mesas_finales),
            "chefs": len(chefs),
            "meseros": len(meseros_dict),
            "clientes": len(clientes),
            "cajeros": len(cajeros),
            "platos_comida": platos_comida,
            "visitas_meseros": visitas_actuales,
        }
        return frame_rgb_out, datos_frame

    def _actualizar_zonas_estaticas(self, resultados_ocr):
        palabras_clave = {
            "KITCHEN": "Cocina",
            "BATHROOM": "Baño",
            "ESCALERAS": "Escaleras",
            "RECEPCION": "Recepción",
            "PEDIDOS": "Recepción",
            "IN": "Entrada",
            "OUT": "Salida",
            "CAJA": "Caja",
            "WINDOWS": "Ventanas",
            "PASILLO": "Pasillo Meseros",
        }
        for (bbox, texto, conf) in resultados_ocr:
            txt_upper = texto.upper().strip