import re
import duckdb
import pandas as pd


class AnalizadorServicio:
    DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

    def __init__(self, db_path: str = "intento_2/dw_restaurante.duckdb"):
        self.db_path = db_path
        self._inicializar_dw()
        self.visitas_activas = set()
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
                estado_actual VARCHAR
            )
        """)

        con.execute("""
            CREATE TABLE fact_visitas_meseros (
                timestamp VARCHAR,
                mesero_id VARCHAR,
                mesa_id VARCHAR
            )
        """)
        con.close()
        self.contador_eventos = 0

    def registrar_frame(self, datos: dict):
        self.contador_eventos += 1
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

            # Solo contamos clientes cuando reciben su plato en la mesa (1 plato = 1 cliente atendido)
            platos_ahora = int(mesa.get("platos", 0)) if esta_oc else 0
            platos_antes = self.platos_por_mesa_prev.get(mid, 0)

            # Si la mesa estaba sin platos (0) y ahora le sirven platos, sumamos esa tanda exacta
            # Si ya tenía platos y llegó un plato extra a la misma mesa, sumamos la diferencia
            incremento_clientes = max(0, platos_ahora - platos_antes)
            self.platos_por_mesa_prev[mid] = platos_ahora
            nuevos_comensales_servidos_frame += incremento_clientes

            nueva_ocupacion = 1 if (esta_oc and mid not in self.mesas_ocupadas_prev) else 0
            sillas_mesa = mesa["sillas"]

            existe = con.execute("SELECT veces_ocupada FROM dim_mesas WHERE mesa_id = ?", [mid]).fetchone()
            if existe is None:
                con.execute(
                    "INSERT INTO dim_mesas VALUES (?, ?, ?, ?, 0, ?)",
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

        pares_actuales = set()
        for v in datos["visitas_meseros"]:
            par = (v["mesero_id"], v["mesa"])
            pares_actuales.add(par)
            if par not in self.visitas_activas:
                con.execute("INSERT INTO fact_visitas_meseros VALUES (?, ?, ?)", [datos["timestamp"], v["mesero_id"], v["mesa"]])
                con.execute("UPDATE dim_mesas SET visitas_mesero = visitas_mesero + 1 WHERE mesa_id = ?", [v["mesa"]])
        self.visitas_activas = pares_actuales

        con.close()

    def obtener_datasets_dw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        con = duckdb.connect(self.db_path)
        df_hechos = con.execute("SELECT * FROM fact_servicio_live ORDER BY id_evento").df()
        df_mesas = con.execute("SELECT * FROM dim_mesas ORDER BY mesa_id").df()
        df_meseros = con.execute("""
            SELECT mesero_id, COUNT(*) AS mesas_atendidas, COUNT(DISTINCT mesa_id) AS mesas_distintas
            FROM fact_visitas_meseros
            GROUP BY mesero_id
            ORDER BY mesas_atendidas DESC
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
                f"👥 **Clientes Atendidos Hoy (con plato servido en mesa):** **{int(row['clientes_acumulados_dia'])} clientes** "
                f"(actualmente se observan **{int(row['clientes_en_salon'])}** en todo el salón)."
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
                        f"y su estado actual es **{r_m['estado_actual']}**."
                    )
            else:
                if not df_mesas.empty:
                    mesa_top = df_mesas.sort_values(by=["clientes_acumulados", "veces_ocupada"], ascending=False).iloc[0]
                    detalle_oc = ", ".join([f"{r['mesa_id']}: {r['clientes_acumulados']} cli ({r['veces_ocupada']}x)" for _, r in df_mesas.iterrows()])
                    respuestas.append(
                        f"🪑 **Uso de Mesas (Más usada: {mesa_top['mesa_id']}):** {detalle_oc}."
                    )

        if any(k in p for k in ["mesero", "trabajo", "trabajó", "atendio", "atendió", "visita", "personal"]):
            tema = "meseros"
            if not df_meseros.empty:
                top_m = df_meseros.iloc[0]
                ranking = ", ".join([f"{r['mesero_id']} ({r['mesas_atendidas']} atenciones)" for _, r in df_meseros.iterrows()])
                respuestas.append(
                    f"🤵 **Productividad de Meseros:** El que más ha trabajado es **{top_m['mesero_id']}** "
                    f"con **{int(top_m['mesas_atendidas'])} atenciones** en mesas. Ranking: {ranking}."
                )
            else:
                respuestas.append(f"🤵 **Meseros activos:** Hay **{int(row['meseros'])} meseros** en turno (aún sin visitas registradas).")

        if not respuestas:
            respuestas.append(
                f"📊 **Resumen Live DW:** {int(row['clientes_acumulados_dia'])} clientes servidos hoy | "
                f"{int(row['mesas_ocupadas'])}/{int(row['total_mesas'])} mesas ocupadas | "
                f"{int(row['chefs'])} chefs | {int(row['meseros'])} meseros | {int(row['platos_comida'])} platos en mesa."
            )

        tabla_mostrar = df_meseros if (tema == "meseros" and not df_meseros.empty) else df_mesas
        return "\n\n".join(respuestas), tabla_mostrar, tema