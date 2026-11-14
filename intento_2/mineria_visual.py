import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
import streamlit as st


def _generar_proyeccion_temporal(df_hechos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    horas = list(range(8, 23))

    curva_horaria = {
        8: 15, 9: 25, 10: 35, 11: 55, 12: 88, 13: 95, 14: 75,
        15: 40, 16: 30, 17: 38, 18: 60, 19: 85, 20: 98, 21: 80, 22: 45
    }
    factor_dia = {
        "Lunes": 0.65, "Martes": 0.72, "Miércoles": 0.80,
        "Jueves": 0.90, "Viernes": 1.15, "Sábado": 1.30, "Domingo": 1.10
    }

    media_clientes_real = max(1.0, float(df_hechos["clientes_en_salon"].mean()))
    escala = media_clientes_real / 6.0

    filas_horarias = []
    for d in dias:
        for h in horas:
            reales = df_hechos[(df_hechos["dia_semana"] == d) & (df_hechos["hora_simulada"] == h)]
            if not reales.empty:
                afluencia = round(float(reales["clientes_en_salon"].mean()) * 10, 1)
                origen = "Captura Live (DW)"
            else:
                afluencia = round(curva_horaria[h] * factor_dia[d] * escala, 1)
                origen = "Proyección DW"
            filas_horarias.append({
                "Dia": d,
                "Hora": f"{h:02d}:00",
                "Hora_Num": h,
                "Afluencia": min(100.0, afluencia),
                "Origen": origen
            })
    df_diario = pd.DataFrame(filas_horarias)

    base_clientes_dia = max(12, int(df_hechos["clientes_acumulados_dia"].max()))
    df_semanal = pd.DataFrame({
        "Dia": dias,
        "Clientes_Totales": [int(base_clientes_dia * factor_dia[d] * 3.2) for d in dias],
        "Ocupacion_Promedio_%": [min(98, int(52 * factor_dia[d])) for d in dias]
    })

    semanas = ["Semana 1", "Semana 2", "Semana 3", "Semana 4"]
    factores_sem = [0.88, 1.02, 0.95, 1.18]
    df_mensual = pd.DataFrame({
        "Semana": semanas,
        "Clientes_Mensuales": [int(df_semanal["Clientes_Totales"].sum() * f) for f in factores_sem],
        "Platos_Servidos": [int(df_semanal["Clientes_Totales"].sum() * f * 1.15) for f in factores_sem]
    })

    return df_diario, df_semanal, df_mensual


def renderizar_capa_mineria(
    df_hechos: pd.DataFrame,
    df_mesas: pd.DataFrame,
    df_meseros: pd.DataFrame,
    pregunta_usuario: str,
    tema: str,
    ciclo: int = 0,
):
    st.header("🔬 Capa 3: Minería de Datos, Frecuencias Temporales & Modelos Predictivos")
    st.caption(f"Análisis filtrado dinámicamente según tu consulta: **«{pregunta_usuario}»**")

    if len(df_hechos) < 4:
        st.info("⏳ Capturando registros del simulador... Espera unos segundos (mínimo 4 eventos) para construir las gráficas y modelos.")
        return

    with st.expander("🏛️ Arquitectura Data Warehouse + Fases KDD & CRISP-DM", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Data Warehouse (Esquema Estrella)**")
            st.markdown(
                f"- **Tabla de Hechos (`fact_servicio_live`):** {len(df_hechos)} eventos.\n"
                f"- **Dimensión Mesas (`dim_mesas`):** {len(df_mesas)} mesas (`A1`–`B2`).\n"
                f"- **Hechos Atención (`fact_visitas_meseros`):** {int(df_meseros['mesas_atendidas'].sum()) if not df_meseros.empty else 0} visitas."
            )
        with c2:
            st.markdown("**2. Proceso KDD (Knowledge Discovery)**")
            st.markdown(
                "- **Selección:** Visión RGB + OCR autónomo.\n"
                "- **Preprocesamiento:** Conteo filtrado por platos servidos.\n"
                "- **Transformación:** Mapeo temporal y ocupación.\n"
                "- **Minería & Evaluación:** K-Means + Ensamble Híbrido."
            )
        with c3:
            st.markdown("**3. Metodología CRISP-DM**")
            st.markdown(
                "- **Negocio:** Identificar horas pico y rotación por mesa.\n"
                "- **Preparación:** Escalado `StandardScaler`.\n"
                "- **Modelado:** Clustering + Árbol/MLP.\n"
                "- **Despliegue:** Analítica en vivo."
            )

    # =========================================================================
    # 2. GRÁFICA DE PASTEL DE USO DE MESAS (A1 - B2) + EVOLUCIÓN EN VIVO
    # =========================================================================
    st.subheader("🥧 Uso de Mesas por Clientes Atendidos (A1 – B2) & Flujo en Vivo")
    col_pie, col_live = st.columns([1.15, 1.35])

    with col_pie:
        if not df_mesas.empty:
            df_pastel = df_mesas.copy()
            total_clientes_mesas = int(df_pastel["clientes_acumulados"].sum())

            if total_clientes_mesas == 0:
                df_pastel["clientes_mostrar"] = df_pastel["cantidad_sillas"]
                subtitulo_pie = "Capacidad por mesa (esperando que sirvan platos a los clientes)"
            else:
                df_pastel["clientes_mostrar"] = df_pastel["clientes_acumulados"]
                subtitulo_pie = f"Total acumulado hoy: {total_clientes_mesas} clientes servidos en mesa"

            mesa_lider = df_pastel.sort_values(by="clientes_mostrar", ascending=False).iloc[0]["mesa_id"]

            fig_pie = px.pie(
                df_pastel,
                names="mesa_id",
                values="clientes_mostrar",
                hole=0.45,
                title=f"Clientes Servidos por Mesa (Más usada: {mesa_lider})",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Clientes servidos: %{value}<br>Porcentaje: %{percent}<extra></extra>",
            )
            fig_pie.add_annotation(
                text=f"Total<br><b>{total_clientes_mesas}</b><br>clientes",
                x=0.5, y=0.5, font_size=13, showarrow=False
            )
            fig_pie.update_layout(height=360, margin=dict(l=15, r=15, t=45, b=15))
            st.plotly_chart(fig_pie, width="stretch", key=f"chart_pie_mesas_{ciclo}")
            st.caption(subtitulo_pie)
        else:
            st.info("Esperando detección de mesas A1 – B2...")

    with col_live:
        fig_tiempo = go.Figure()
        fig_tiempo.add_trace(go.Scatter(
            x=df_hechos["id_evento"], y=df_hechos["clientes_en_salon"],
            mode="lines+markers", name="Clientes en Salón", line=dict(color="#e84393", width=3)
        ))
        fig_tiempo.add_trace(go.Scatter(
            x=df_hechos["id_evento"], y=df_hechos["platos_comida"],
            mode="lines", name="Platos Servidos", line=dict(color="#fdcb6e", width=2)
        ))
        fig_tiempo.add_trace(go.Scatter(
            x=df_hechos["id_evento"], y=df_hechos["mesas_ocupadas"],
            mode="lines", name="Mesas Ocupadas", line=dict(color="#00b894", width=2, dash="dot")
        ))
        fig_tiempo.update_layout(
            title="Evolución en Tiempo Real (Clientes, Mesas Ocupadas y Platos)",
            xaxis_title="Eventos Capturados",
            yaxis_title="Cantidad",
            height=360,
            margin=dict(l=20, r=20, t=45, b=15),
        )
        st.plotly_chart(fig_tiempo, width="stretch", key=f"chart_live_tiempo_{ciclo}")

    # =========================================================================
    # 3. FRECUENCIAS DIARIAS (ESTILO GOOGLE MAPS), SEMANALES Y MENSUALES
    # =========================================================================
    st.subheader("📅 Frecuencias de Movimiento: Diaria (Horas Punta), Semanal y Mensual")
    df_diario, df_semanal, df_mensual = _generar_proyeccion_temporal(df_hechos)

    tab_diaria, tab_semanal, tab_mensual = st.tabs([
        "🕒 Frecuencia Diaria (Horas Punta tipo Google Maps)",
        "📆 Frecuencia Semanal (Días más concurridos)",
        "🗓️ Frecuencia Mensual (Tendencia por Semanas)"
    ])

    with tab_diaria:
        dias_lista = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        if "dia_gmaps_actual" not in st.session_state:
            st.session_state.dia_gmaps_actual = "Sábado"

        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            dia_elegido = st.selectbox(
                "Selecciona el día de la semana:",
                dias_lista,
                index=dias_lista.index(st.session_state.dia_gmaps_actual),
                key=f"selector_dia_gmaps_{ciclo}",
            )
            st.session_state.dia_gmaps_actual = dia_elegido

        df_dia_filtrado = df_diario[df_diario["Dia"] == dia_elegido]
        hora_pico = df_dia_filtrado.sort_values(by="Afluencia", ascending=False).iloc[0]

        with col_info:
            st.info(
                f"📍 **Horas Punta el {dia_elegido}:** El mayor movimiento ocurre alrededor de las **{hora_pico['Hora']}** "
                f"(nivel de ocupación estimado: **{hora_pico['Afluencia']:.0f}%**)."
            )

        colores_barras = [
            "#ff4b4b" if val >= 80 else ("#ffa502" if val >= 55 else "#3498db")
            for val in df_dia_filtrado["Afluencia"]
        ]
        fig_gmaps = go.Figure(
            data=[
                go.Bar(
                    x=df_dia_filtrado["Hora"],
                    y=df_dia_filtrado["Afluencia"],
                    marker_color=colores_barras,
                    text=[f"{int(v)}%" for v in df_dia_filtrado["Afluencia"]],
                    textposition="outside",
                )
            ]
        )
        fig_gmaps.update_layout(
            title=f"Horas Punta — {dia_elegido} (Estilo Google Maps)",
            xaxis_title="Franja Horaria",
            yaxis_title="Nivel de Concurrencia (%)",
            yaxis=dict(range=[0, 115]),
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gmaps, width="stretch", key=f"chart_gmaps_{ciclo}")

    with tab_semanal:
        fig_sem = px.bar(
            df_semanal,
            x="Dia",
            y="Clientes_Totales",
            color="Ocupacion_Promedio_%",
            text="Clientes_Totales",
            title="Afluencia Semanal por Día (Identificación de Días Pico)",
            labels={"Dia": "Día de la Semana", "Clientes_Totales": "Clientes Proyectados", "Ocupacion_Promedio_%": "Ocupación %"},
            color_continuous_scale="YlOrRd",
        )
        fig_sem.update_traces(textposition="outside")
        fig_sem.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sem, width="stretch", key=f"chart_semanal_{ciclo}")

    with tab_mensual:
        fig_mes = go.Figure()
        fig_mes.add_trace(go.Bar(
            x=df_mensual["Semana"], y=df_mensual["Clientes_Mensuales"],
            name="Clientes Totales", marker_color="#6c5ce7"
        ))
        fig_mes.add_trace(go.Scatter(
            x=df_mensual["Semana"], y=df_mensual["Platos_Servidos"],
            name="Platos Servidos", mode="lines+markers", line=dict(color="#00cec9", width=3)
        ))
        fig_mes.update_layout(
            title="Frecuencia Mensual: Volumen de Clientes y Platos por Semana",
            xaxis_title="Periodo del Mes",
            yaxis_title="Unidades / Personas",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_mes, width="stretch", key=f"chart_mensual_{ciclo}")

    # =========================================================================
    # 4. CLUSTERING (K-MEANS)
    # =========================================================================
    st.subheader("🧩 Segmentación No Supervisada (Clustering K-Means)")
    features_cluster = ["clientes_en_salon", "mesas_ocupadas", "platos_comida", "tasa_ocupacion"]
    X_raw = df_hechos[features_cluster].fillna(0)

    col_k1, col_k2 = st.columns([1.4, 1])

    if len(X_raw) >= 8 and X_raw.nunique().sum() > 4:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        n_clusters = min(3, len(X_raw.drop_duplicates()))
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_hechos["cluster"] = kmeans.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, df_hechos["cluster"]) if len(set(df_hechos["cluster"])) > 1 else 0.0

            nombres_perfil = {0: "Demanda Baja / Valle", 1: "Servicio Moderado", 2: "Hora Pico / Alta Saturación"}
            df_hechos["perfil_operativo"] = df_hechos["cluster"].map(lambda c: nombres_perfil.get(c, f"Clúster {c}"))

            with col_k1:
                fig_cluster = px.scatter(
                    df_hechos,
                    x="clientes_en_salon",
                    y="platos_comida",
                    color="perfil_operativo",
                    size="tasa_ocupacion",
                    hover_data=["mesas_ocupadas", "meseros"],
                    title=f"Clústeres de Operación (Silhouette Score: {sil_score:.2f})",
                    labels={"clientes_en_salon": "Clientes en Salón", "platos_comida": "Platos Servidos"},
                )
                fig_cluster.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cluster, width="stretch", key=f"chart_kmeans_{ciclo}")

            with col_k2:
                st.markdown("**Centroides y Reglas de Segmentación:**")
                resumen_c = df_hechos.groupby("perfil_operativo")[features_cluster].mean().round(2)
                st.dataframe(resumen_c, width="stretch")
                st.caption("Agrupa automáticamente los estados del simulador en niveles de carga operativa.")
        else:
            st.warning("Se necesita mayor variabilidad de movimiento en el simulador para formar clústeres.")
    else:
        st.info("Acumulando más estados operativos distintos para calcular los clústeres K-Means...")

    # =========================================================================
    # 5. MODELO HÍBRIDO (ÁRBOL DE DECISIÓN + RED NEURONAL MLP) & PREDICCIÓN
    # =========================================================================
    st.subheader("🤖 Modelo Híbrido Predictivo (Árbol de Decisión + Red Neuronal MLP)")

    def obtener_probabilidad_clase_1(modelo, X_datos) -> np.ndarray:
        probs = modelo.predict_proba(X_datos)
        clases = list(modelo.classes_)
        if 1 in clases:
            return probs[:, clases.index(1)]
        return np.zeros(len(X_datos))

    df_ml = df_hechos.copy()
    df_ml["alta_demanda"] = ((df_ml["tasa_ocupacion"] >= 50.0) | (df_ml["clientes_en_salon"] >= 6)).astype(int)

    conteo_clases = df_ml["alta_demanda"].value_counts()

    if len(conteo_clases) >= 2 and conteo_clases.min() >= 2 and len(df_ml) >= 10:
        X = df_ml[["clientes_en_salon", "mesas_ocupadas", "meseros", "platos_comida"]]
        y = df_ml["alta_demanda"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler_ml = StandardScaler()
        X_train_sc = scaler_ml.fit_transform(X_train)
        X_test_sc = scaler_ml.transform(X_test)

        tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree_clf.fit(X_train, y_train)
        acc_tree = accuracy_score(y_test, tree_clf.predict(X_test))

        mlp_clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
        mlp_clf.fit(X_train_sc, y_train)
        acc_mlp = accuracy_score(y_test, mlp_clf.predict(X_test_sc))

        prob_tree = obtener_probabilidad_clase_1(tree_clf, X_test)
        prob_mlp = obtener_probabilidad_clase_1(mlp_clf, X_test_sc)
        pred_hibrida = ((0.5 * prob_tree + 0.5 * prob_mlp) >= 0.5).astype(int)
        acc_hibrido = accuracy_score(y_test, pred_hibrida)

        ultimo_x = X.iloc[[-1]]
        ultimo_x_sc = scaler_ml.transform(ultimo_x)
        p_tree_act = obtener_probabilidad_clase_1(tree_clf, ultimo_x)[0]
        p_mlp_act = obtener_probabilidad_clase_1(mlp_clf, ultimo_x_sc)[0]
        prob_saturacion_actual = float(0.5 * p_tree_act + 0.5 * p_mlp_act)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precisión Árbol Decisión", f"{acc_tree * 100:.1f}%")
        m2.metric("Precisión Red Neuronal MLP", f"{acc_mlp * 100:.1f}%")
        m3.metric("Precisión Modelo Híbrido", f"{acc_hibrido * 100:.1f}%")
        m4.metric("Riesgo Saturación Próximo", f"{prob_saturacion_actual * 100:.1f}%")

        c_reglas, c_pred = st.columns([1.2, 1])
        with c_reglas:
            st.markdown("**Reglas Extraídas del Árbol de Decisión (CRISP-DM):**")
            reglas_txt = export_text(tree_clf, feature_names=list(X.columns))
            st.code(reglas_txt, language="text")

        with c_pred:
            st.markdown("**Recomendación Predictiva del Sistema:**")
            if prob_saturacion_actual >= 0.65:
                st.error("⚠️ **Alerta de Alta Demanda:** Se proyecta saturación en comedor. Se recomienda asignar un mesero adicional al salón.")
            elif prob_saturacion_actual >= 0.35:
                st.warning("⚖️ **Flujo Equilibrado:** Capacidad operativa óptima con margen disponible.")
            else:
                st.success("✅ **Demanda Baja:** Mesas disponibles y personal con baja carga.")
    else:
        st.info("⏳ Acumulando muestras balanceadas (demanda baja y alta) para entrenar el Modelo Híbrido sin sesgo...")