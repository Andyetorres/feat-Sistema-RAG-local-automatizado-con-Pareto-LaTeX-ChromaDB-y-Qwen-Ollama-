import re
import time
import duckdb
import pandas as pd


class AnalizadorServicio:
    DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

    def __init__(self, db_path: str = "intento_2/dw_restaurante.duckdb"):
        self.db_path = db_path
        self._inicializar_dw()
        # {(mesero_id, mesa_id): timestamp_inicio_float}
        self.visitas_activas = {}
        self.mesas_ocupadas_prev = set()
        self.platos_por_mesa_prev = {}
        self.clientes_atendidos_hoy = 0

    def _inicializar_dw(self):
        con = duckdb.connect(self.db_path)
        con.execute("DROP TABLE IF EXISTS fact_servicio_live")
        con.execute("DROP TABLE IF EXISTS dim_mesas")
        con.execute("DROP TABLE IF EXISTS fact_visitas_meseros")

        con.execute("""
            CREATE TABLE fact_servicio_live (
                id_evento INTEGER,
                timestamp VARCHAR,
                dia_semana VARCHAR,
                hora_simulada INTEGER,
                semana_mes INTEGER,
                total_mesas INTEGER,
                mesas_ocupadas INTEGER,
                total_sillas INTEGER,
                chefs INTEGER,
                meseros INTEGER,
                clientes_en_salon INTEGER,
                clientes_acumulados_dia INTEGER,
                platos_comida INTEGER,
                tasa_ocupacion DOUBLE,
                zonas_detectadas VARCHAR
            )
        """)

        con.execute("""
            CREATE TABLE dim_mesas (
                mesa_id VARCHAR PRIMARY KEY,
                cantidad_sillas INTEGER,
                veces_ocupada INTEGER,
                clientes_acumulados INTEGER,
                visitas_mesero INTEGER,
                tiempo_promedio_atencion_seg DOUBLE,
                estado_actual VARCHAR
            )
        """)

        con.execute("""
            CREATE TABLE fact_visitas_meseros (
                timestamp VARCHAR,
                mesero_id VARCHAR,
                mesa_id VARCHAR,
                duracion_seg DOUBLE
            )
        """)
        con.close()
        self.contador_eventos = 0

    def registrar_frame(self, datos: dict):
        self.contador_eventos += 1
        ahora_epoch = time.time()
        con = duckdb.connect(self.db_path)

        minutos_totales = (self.contador_eventos - 1) * 15
        horas_transcurridas = minutos_totales // 60
        hora_sim = 8 + (horas_transcurridas % 15)
        dia_idx = (horas_transcurridas // 15) % 7
        dia_sem = self.DIAS_SEMANA[dia_idx]
        semana_m = ((horas_transcurridas // (15 * 7)) % 4) + 1

        ids_actuales = [m["id"] for m in datos["mesas"]]
        if ids_actuales:
            placeholders = ", ".join(["?"] * len(ids_actuales))
            con.execute(f"DELETE FROM dim_mesas WHERE mesa_id NOT IN ({placeholders})", ids_actuales)

        ocupadas_ahora = set()
        nuevos_comensales_servidos_frame = 0

        for mesa in datos["mesas"]:
            mid = mesa["id"]
            esta_oc = mesa.get("ocupada", False)
            estado_str = "OCUPADA" if esta_oc else "LIBRE"
            if esta_oc:
                ocupadas_ahora.add(mid)

            platos_ahora = int(mesa.get("platos", 0)) if esta_oc else 0
            platos_antes = self.platos_por_mesa_prev.get(mid, 0)
            incremento_clientes = max(0, platos_ahora - platos_antes)
            self.platos_por_mesa_prev[mid] = platos_ahora
            nuevos_comensales_servidos_frame += incremento_clientes

            nueva_ocupacion = 1 if (esta_oc and mid not in self.mesas_ocupadas_prev) else 0
            sillas_mesa = mesa["sillas"]

            existe = con.execute("SELECT veces_ocupada FROM dim_mesas WHERE mesa_id = ?", [mid]).fetchone()
            if existe is None:
                con.execute(
                    "INSERT INTO dim_mesas VALUES (?, ?, ?, ?, 0, 0.0, ?)",
                    [mid, sillas_mesa, 1 if esta_oc else 0, platos_ahora, estado_str],
                )
            else:
                con.execute(
                    """
                    UPDATE dim_mesas
                    SET cantidad_sillas = ?,
                        veces_ocupada = veces_ocupada + ?,
                        clientes_acumulados = clientes_acumulados + ?,
                        estado_actual = ?
                    WHERE mesa_id = ?
                    """,
                    [sillas_mesa, nueva_ocupacion, incremento_clientes, estado_str, mid],
                )

        self.mesas_ocupadas_prev = ocupadas_ahora
        self.clientes_atendidos_hoy += nuevos_comensales_servidos_frame

        mesas_ocupadas_count = len(ocupadas_ahora)
        tasa_ocup = round((mesas_ocupadas_count / max(1, datos["total_mesas"])) * 100, 2)

        con.execute(
            "INSERT INTO fact_servicio_live VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.contador_eventos,
                datos["timestamp"],
                dia_sem,
                hora_sim,
                semana_m,
                datos["total_mesas"],
                mesas_ocupadas_count,
                datos["total_sillas"],
                datos["chefs"],
                datos["meseros"],
                datos["clientes"],
                self.clientes_atendidos_hoy,
                datos["platos_comida"],
                tasa_ocup,
                ", ".join(sorted(datos["zonas"])),
            ],
        )

        # Cronometrar visitas de cada mesero a cada mesa
        pares_actuales = set()
        for v in datos["visitas_meseros"]:
            par = (v["mesero_id"], v["mesa"])
            pares_actuales.add(par)
            if par not in self.visitas_activas:
                # Inicia atención en mesa
                self.visitas_activas[par] = ahora_epoch
                con.execute("UPDATE dim_mesas SET visitas_mesero = visitas_mesero + 1 WHERE mesa_id = ?", [v["mesa"]])

        # Cerrar las visitas que terminaron en este frame y guardar su duración en segundos
        terminadas = [par for par in self.visitas_activas if par not in pares_actuales]
        for par in terminadas:
            t_inicio = self.visitas_activas.pop(par)
            duracion = round(max(0.3, ahora_epoch - t_inicio), 2)
            mesero_id, mesa_id = par
            con.execute(
                "INSERT INTO fact_visitas_meseros VALUES (?, ?, ?, ?)",
                [datos["timestamp"], mesero_id, mesa_id, duracion],
            )
            # Actualizar tiempo promedio de atención en dim_mesas
            avg_mesa = con.execute(
                "SELECT AVG(duracion_seg) FROM fact_visitas_meseros WHERE mesa_id = ?", [mesa_id]
            ).fetchone()[0]
            if avg_mesa is not None:
                con.execute(
                    "UPDATE dim_mesas SET tiempo_promedio_atencion_seg = ? WHERE mesa_id = ?",
                    [round(float(avg_mesa), 2), mesa_id],
                )

        con.close()

    def obtener_datasets_dw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        con = duckdb.connect(self.db_path)
        df_hechos = con.execute("SELECT * FROM fact_servicio_live ORDER BY id_evento").df()
        df_mesas = con.execute("SELECT * FROM dim_mesas ORDER BY mesa_id").df()
        df_meseros = con.execute("""
            SELECT
                mesero_id AS Mesero,
                COUNT(*) AS Mesas_Atendidas,
                COUNT(DISTINCT mesa_id) AS Mesas_Distintas,
                ROUND(AVG(duracion_seg), 2) AS Tiempo_Promedio_Seg,
                ROUND(SUM(duracion_seg), 1) AS Tiempo_Total_Seg
            FROM fact_visitas_meseros
            GROUP BY mesero_id
            ORDER BY Mesas_Atendidas DESC, Tiempo_Promedio_Seg ASC
        """).df()
        con.close()
        return df_hechos, df_mesas, df_meseros

    def responder_pregunta_usuario(self, pregunta: str) -> tuple[str, pd.DataFrame, str]:
        df_hechos, df_mesas, df_meseros = self.obtener_datasets_dw()
        if df_hechos.empty:
            return "Esperando transmisión en vivo para poblar el Data Warehouse...", pd.DataFrame(), "general"

        row = df_hechos.iloc[-1]
        p = pregunta.lower()
        respuestas = []
        tema = "general"

        if any(k in p for k in ["cliente", "vinieron", "llegaron", "hoy", "afluencia"]):
            tema = "clientes"
            respuestas.append(
                f"👥 **Clientes Atendidos Hoy (con plato servido):** **{int(row['clientes_acumulados_dia'])} clientes** "
                f"(actualmente hay **{int(row['clientes_en_salon'])}** en el salón)."
            )

        if any(k in p for k in ["ocuparon", "ocupada", "mesa", "silla", "a1", "a2", "a3", "b1", "b2", "usada"]):
            tema = "mesas"
            match_mesa = re.search(r"\b([ab][1-3])\b", p)
            if match_mesa and not df_mesas.empty:
                cod = match_mesa.group(1).upper()
                sub = df_mesas[df_mesas["mesa_id"].str.startswith(cod)]
                if not sub.empty:
                    r_m = sub.iloc[0]
                    respuestas.append(
                        f"🪑 **Detalle Mesa {r_m['mesa_id']}:** Ocupada **{int(r_m['veces_ocupada'])} veces**, "
                        f"atendió a **{int(r_m['clientes_acumulados'])} clientes**, **{int(r_m['visitas_mesero'])} visitas** "
                        f"(tiempo promedio de atención: **{r_m['tiempo_promedio_atencion_seg']}s**)."
                    )
            elif not df_mesas.empty:
                mesa_top = df_mesas.sort_values(by=["clientes_acumulados", "veces_ocupada"], ascending=False).iloc[0]
                detalle_oc = ", ".join([f"{r['mesa_id']}: {r['clientes_acumulados']} cli ({r['veces_ocupada']}x)" for _, r in df_mesas.iterrows()])
                respuestas.append(f"🪑 **Uso de Mesas (Más usada: {mesa_top['mesa_id']}):** {detalle_oc}.")

        if any(k in p for k in ["mesero", "trabajo", "trabajó", "atendio", "atendió", "tiempo", "demora", "demoran", "tarda"]):
            tema = "meseros"
            if not df_meseros.empty:
                lider = df_meseros.iloc[0]
                respuestas.append(
                    f"🤵 **Desempeño de Meseros:** El que más mesas ha atendido es **{lider['Mesero']}** "
                    f"con **{int(lider['Mesas_Atendidas'])} atenciones** y un tiempo promedio de **{lider['Tiempo_Promedio_Seg']} segundos** por mesa."
                )
            else:
                respuestas.append(
                    f"🤵 **Meseros en turno ({int(row['meseros'])} activos):** Cronometrando visitas a mesas en curso..."
                )

        if not respuestas:
            respuestas.append(
                f"📊 **Resumen Live DW:** {int(row['clientes_acumulados_dia'])} clientes servidos hoy | "
                f"{int(row['mesas_ocupadas'])}/{int(row['total_mesas'])} mesas ocupadas | "
                f"{int(row['chefs'])} chefs | {int(row['meseros'])} meseros | {int(row['platos_comida'])} platos en mesa."
            )

        tabla_mostrar = df_meseros if (tema == "meseros" and not df_meseros.empty) else df_mesas
        return "\n\n".join(respuestas), tabla_mostrar, tema