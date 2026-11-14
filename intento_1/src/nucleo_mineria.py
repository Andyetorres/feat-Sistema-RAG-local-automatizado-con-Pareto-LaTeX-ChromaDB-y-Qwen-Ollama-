"""
Núcleo de minería de datos (sin interfaz).

Curso de Minería de Datos · Universidad del Tolima

Cada función corresponde a una etapa del flujo CRISP-DM y a un tema del curso:

  Unidad 1  cargar_csv, perfil, almacen              (comprensión de los datos, almacén de datos)
  Unidad 3  preparar                                  (validación de datos erróneos, partición)
  Unidad 2  entrenar_arbol, entrenar_red              (modelos de clasificación / predicción)
  Unidad 3  segmentar, pronosticar, asociacion        (clústeres, series de tiempo, asociación)
  Unidad 4  construir_informe                         (documento con pasos y resultados)

Solo depende de pandas, numpy y scikit-learn.
"""
from __future__ import annotations

import io
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

NA_VALORES = ["NA", "N/A", "n/a", "null", "NULL", "NaN", "nan", "None", "none", "?", "-"]


# --------------------------------------------------------------------------- utilidades
def _p(x: float, d: int = 1) -> str:
    """Porcentaje con coma decimal."""
    return f"{x * 100:.{d}f}%".replace(".", ",")


def _f(x: float, d: int = 2) -> str:
    """Número con coma decimal y separador de miles."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def es_numerica(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)


def tipos(df: pd.DataFrame) -> dict[str, str]:
    """'num' o 'cat' para cada columna."""
    return {c: ("num" if es_numerica(df[c]) else "cat") for c in df.columns}


def es_identificador(nombre: str) -> bool:
    """Columnas que normalmente no sirven como variable de entrada (id, mes, fecha...)."""
    import re

    return bool(re.match(r"^(id|mes|periodo|per[ií]odo|fecha|a[nñ]o|codigo|c[oó]digo)", nombre, re.I))


# --------------------------------------------------------------------------- Creatividad e Inteligencia de Negocio
def _bautizar_cluster(medias: pd.Series) -> str:
    """Asigna nombres comerciales atractivos (Buyer Personas) según el perfil del clúster."""
    gasto = medias.get("gasto_miles_cop", 0)
    compras = medias.get("compras_mes", 0)
    edad = medias.get("edad", 35)
    antiguedad = medias.get("antiguedad_meses", 12)
    
    if gasto >= 320 and compras >= 5:
        return "💎 VIPs de Alto Valor"
    elif antiguedad >= 22 and compras >= 4:
        return "🛡️ Leales de Largo Plazo"
    elif gasto < 220:
        return "🔍 Compradores Ocasionales / Bajo Gasto"
    elif edad <= 28:
        return "⚡ Jóvenes Exploradores"
    else:
        return "⭐ Segmento Core / Promedio"


def generar_smart_insights(df: pd.DataFrame, prep: dict, resultados: dict) -> list[str]:
    """Genera tarjetas de conocimiento automático (Smart Insights) para la toma de decisiones."""
    insights = []
    if df is None:
        return ["Carga un conjunto de datos para generar insights inteligentes."]
    
    if "gasto_miles_cop" in df.columns:
        gasto_prom = df["gasto_miles_cop"].mean()
        insights.append(f"💡 **Tendencia Financiera:** El gasto promedio analizado se sitúa en **${gasto_prom:,.1f}**, reflejando la dinámica de consumo de la base.")
    
    if "responde_campana" in df.columns:
        try:
            tasa_resp = (df["responde_campana"].astype(str).str.lower().isin(["si", "s", "1", "true"])).mean() * 100
            insights.append(f"🎯 **Efectividad Comercial:** La tasa global de respuesta a campañas detectada es del **{tasa_resp:.1f}%**.")
        except Exception:
            pass
        
    if "segmentos" in resultados:
        k = resultados["segmentos"]["k"]
        insights.append(f"🧩 **Arquitectura de Clientes:** El algoritmo K-Medias segmentó la base en **{k} Buyer Personas** comerciales diferenciadas.")
        
    return insights


# --------------------------------------------------------------------------- Unidad 1: datos
def _leer_uno(origen) -> pd.DataFrame:
    if hasattr(origen, "read"):
        crudo = origen.read()
    else:
        with open(origen, "rb") as fh:
            crudo = fh.read()
    if isinstance(crudo, bytes):
        try:
            texto = crudo.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = crudo.decode("latin-1")
    else:
        texto = crudo
    df = pd.read_csv(io.StringIO(texto), sep=None, engine="python", na_values=NA_VALORES)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            s = df[c].dropna().astype(str).str.strip()
            if len(s) and s.str.fullmatch(r"-?\d+(,\d+)?").mean() >= 0.9:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    return df


def cargar_csv(archivos) -> pd.DataFrame:
    """Lee uno o varios CSV (rutas o archivos subidos). Si son varios, los integra
    alineando las columnas por nombre (integración de datos)."""
    if not isinstance(archivos, (list, tuple)):
        archivos = [archivos]
    dfs = [_leer_uno(a) for a in archivos]
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True, join="outer")


def datos_ejemplo(n: int = 600, semilla: int = 2026) -> pd.DataFrame:
    """Clientes de una tienda en línea, generados con una semilla fija.
    Incluye vacíos, atípicos y duplicados a propósito para practicar la preparación."""
    rng = np.random.default_rng(semilla)
    edad = np.clip(np.round(38 + rng.normal(size=n) * 12), 18, 70)
    ing = np.round(np.maximum(1, 3 + (edad - 18) * 0.06 + rng.normal(size=n) * 1.8), 1)
    canal = rng.choice(["Web", "App", "Tienda"], size=n, p=[0.40, 0.35, 0.25])
    cat = rng.choice(["Tecnología", "Moda", "Hogar", "Alimentos", "Deportes"], size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    ciudad = rng.choice(["Ibagué", "Bogotá", "Medellín", "Cali", "Otra"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    mes = rng.integers(1, 25, size=n)
    compras = np.maximum(0, np.round(2 + np.where(canal == "App", 2, np.where(canal == "Web", 1, 0)) + ing * 0.3 + rng.normal(size=n) * 1.5))
    estacion = 1 + 0.18 * np.sin(2 * np.pi * mes / 12) + 0.012 * mes
    gasto = np.round(np.maximum(20, (60 + ing * 25 + compras * 18 + rng.normal(size=n) * 30) * estacion))
    antig = np.round(np.maximum(1, 18 + rng.normal(size=n) * 12))
    z = -3.4 + 0.6 * compras + np.where(canal == "App", 1.4, 0) - 0.05 * (edad - 38) + 0.006 * (gasto - 150) + rng.normal(size=n) * 0.3
    resp = np.where(1 / (1 + np.exp(-z)) > rng.random(n), "Si", "No")
    df = pd.DataFrame({
        "edad": edad.astype(int), "ingresos_mill_cop": ing, "gasto_miles_cop": gasto.astype(float),
        "compras_mes": compras.astype(int), "antiguedad_meses": antig.astype(int),
        "canal": canal, "categoria": cat, "ciudad": ciudad, "mes": mes, "responde_campana": resp,
    })
    df.loc[rng.choice(n, 18, replace=False), "ingresos_mill_cop"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "canal"] = np.nan
    atip = rng.choice(n, 6, replace=False)
    df.loc[atip, "gasto_miles_cop"] = df.loc[atip, "gasto_miles_cop"] * 7
    return pd.concat([df, df.iloc[rng.choice(n, 8, replace=False)]], ignore_index=True)


def perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por columna: tipo, vacíos, únicos y estadísticos básicos."""
    filas = []
    for c in df.columns:
        s = df[c]
        if es_numerica(s):
            resumen, mn, mx = s.mean(), s.min(), s.max()
        else:
            moda = s.mode(dropna=True)
            resumen, mn, mx = (moda.iloc[0] if len(moda) else ""), "", ""
        filas.append({"columna": c, "tipo": "numérica" if es_numerica(s) else "categórica", "vacíos": int(s.isna().sum()),
                      "únicos": int(s.nunique()), "media / más frecuente": resumen, "mín": mn, "máx": mx})
    out = pd.DataFrame(filas)
    for c in ["media / más frecuente", "mín", "máx"]:
        out[c] = out[c].map(lambda v: "" if v == "" else (f"{v:,.2f}" if isinstance(v, (int, float, np.number)) else str(v)))
    return out


def almacen(df: pd.DataFrame, dimension: str, medida: str | None, agregacion: str = "promedio") -> pd.DataFrame:
    """Vista multidimensional simple: agrupa por una dimensión y resume una medida."""
    g = df.groupby(dimension, dropna=True)
    if agregacion == "conteo" or medida is None:
        out = g.size().rename("valor").reset_index()
    else:
        fn = "sum" if agregacion == "suma" else "mean"
        out = g[medida].agg(fn).rename("valor").reset_index()
    return out.sort_values("valor", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- Unidad 3: preparación
def preparar(df: pd.DataFrame, quitar_duplicados: bool = True, vacios: str = "imputar", atipicos: str = "recortar",
             pct_train: int = 70, semilla: int = 42, excluir_atipicos: tuple = ()) -> dict:
    log: list[str] = []
    d = df.copy()
    tip = tipos(d)
    nums = [c for c, t in tip.items() if t == "num"]
    info = {"filas_antes": len(d), "vacios_antes": int(d.isna().sum().sum()), "atipicos": {}, "negativos": {}}
    for c in nums:
        k = int((d[c] < 0).sum())
        if k:
            info["negativos"][c] = k

    if quitar_duplicados:
        antes = len(d)
        d = d.drop_duplicates().reset_index(drop=True)
        info["duplicados"] = antes - len(d)
        log.append(f"Duplicados: se eliminaron {info['duplicados']} filas repetidas.")

    if vacios == "eliminar":
        antes = len(d)
        d = d.dropna().reset_index(drop=True)
        log.append(f"Valores faltantes: se eliminaron {antes - len(d)} filas con al menos un dato vacío ({info['vacios_antes']} celdas vacías).")
    elif vacios == "imputar":
        n_imp = int(d.isna().sum().sum())
        for c in d.columns:
            if d[c].isna().any():
                if tip[c] == "num":
                    d[c] = d[c].fillna(d[c].median())
                else:
                    d[c] = d[c].fillna(d[c].mode().iloc[0])
        log.append(f"Valores faltantes: se imputaron {n_imp} celdas (mediana en numéricas, moda en categóricas).")
    else:
        log.append(f"Valores faltantes: se dejaron sin cambios ({info['vacios_antes']} celdas vacías).")

    if atipicos != "ninguno":
        limites = {}
        for c in nums:
            if c in excluir_atipicos:
                continue
            s = d[c].dropna()
            if len(s) < 8:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            limites[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        if atipicos == "recortar":
            for c, (lo, hi) in limites.items():
                fuera = int(((d[c] < lo) | (d[c] > hi)).sum())
                if fuera:
                    info["atipicos"][c] = fuera
                    d[c] = d[c].clip(lo, hi)
            log.append(f"Atípicos (regla 1,5×IQR): {sum(info['atipicos'].values())} valores recortados al límite.")
        else:
            antes = len(d)
            mask = pd.Series(True, index=d.index)
            for c, (lo, hi) in limites.items():
                mask &= d[c].isna() | ((d[c] >= lo) & (d[c] <= hi))
            d = d[mask].reset_index(drop=True)
            log.append(f"Atípicos (regla 1,5×IQR): se eliminaron {antes - len(d)} filas.")
    else:
        log.append("Atípicos: no se modificaron.")

    train, test = train_test_split(d, train_size=pct_train / 100, random_state=semilla, shuffle=True)
    log.append(f"Partición aleatoria (semilla {semilla}): {len(train)} filas de entrenamiento y {len(test)} de prueba.")
    info["filas_despues"] = len(d)
    return {"limpio": d, "train": train.reset_index(drop=True), "test": test.reset_index(drop=True), "log": log, "info": info}


# --------------------------------------------------------------------------- Unidad 2: clasificación
def _X(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    X = df[variables].copy()
    for c in variables:
        if not es_numerica(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def _preprocesador(df: pd.DataFrame, variables: list[str], escalar: bool) -> ColumnTransformer:
    nums = [c for c in variables if es_numerica(df[c])]
    cats = [c for c in variables if c not in nums]
    partes = []
    if nums:
        partes.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("esc", StandardScaler() if escalar else "passthrough")]), nums))
    if cats:
        partes.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=9, sparse_output=False))]), cats))
    return ColumnTransformer(partes)


def _evaluar(real, pred, clases: list[str]) -> dict:
    cm = confusion_matrix(real, pred, labels=clases)
    rep = classification_report(real, pred, labels=clases, output_dict=True, zero_division=0)
    por_clase = pd.DataFrame({c: rep[c] for c in clases}).T[["precision", "recall", "f1-score", "support"]]
    por_clase.columns = ["precisión", "sensibilidad", "F1", "casos"]
    return {"exactitud": float(accuracy_score(real, pred)),
            "matriz": pd.DataFrame(cm, index=[f"real {c}" for c in clases], columns=[f"predicho {c}" for c in clases]),
            "por_clase": por_clase}


def _limpiar_nombre(n: str) -> str:
    return n.replace("num__", "").replace("cat__", "")


def _agrupar_importancias(nombres, valores, variables) -> pd.Series:
    out = {v: 0.0 for v in variables}
    for nm, val in zip(nombres, valores):
        for v in sorted(variables, key=len, reverse=True):
            if nm == f"num__{v}" or nm.startswith(f"cat__{v}_"):
                out[v] += float(val)
                break
    return pd.Series(out).sort_values(ascending=False)


def _preparar_supervisado(train, test, objetivo, variables):
    tr = train.dropna(subset=[objetivo]).reset_index(drop=True)
    te = test.dropna(subset=[objetivo]).reset_index(drop=True)
    if len(tr) < 20 or len(te) < 5:
        raise ValueError("Hay muy pocas filas para entrenar y probar. Revisa la preparación de datos.")
    return tr, te


def entrenar_arbol(train, test, objetivo, variables, profundidad=4, min_hoja=10, semilla=42) -> dict:
    tr, te = _preparar_supervisado(train, test, objetivo, variables)
    pre = _preprocesador(tr, variables, escalar=False)
    modelo = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=profundidad, min_samples_leaf=min_hoja, random_state=semilla))])
    modelo.fit(_X(tr, variables), tr[objetivo].astype(str))
    clases = [str(c) for c in modelo.classes_]
    ev_te = _evaluar(te[objetivo].astype(str), modelo.predict(_X(te, variables)), clases)
    ev_tr = _evaluar(tr[objetivo].astype(str), modelo.predict(_X(tr, variables)), clases)
    base = float(te[objetivo].astype(str).value_counts(normalize=True).max())
    nombres = list(pre.get_feature_names_out())
    clf = modelo.named_steps["clf"]
    imp = _agrupar_importancias(nombres, clf.feature_importances_, variables)
    reglas = export_text(clf, feature_names=[_limpiar_nombre(n) for n in nombres])
    lineas = [
        f"Objetivo: {objetivo}. Variables: {', '.join(variables)}.",
        f"Exactitud en prueba {_p(ev_te['exactitud'])} (entrenamiento {_p(ev_tr['exactitud'])}). "
        f"Referencia al predecir siempre la clase más frecuente: {_p(base)}.",
        f"Árbol con {clf.tree_.node_count} nodos y profundidad {clf.get_depth()}.",
    ]
    top = imp[imp > 0].head(3)
    lineas.append("Variables que más pesan: " + ", ".join(f"{k} ({_f(v * 100, 1)})" for k, v in top.items()) + "."
                  if len(top) else "El árbol no encontró divisiones útiles.")
    return {"tipo": "arbol", "titulo": "Árbol de decisión", "modelo": modelo, "objetivo": objetivo, "variables": variables,
            "clases": clases, "ev_test": ev_te, "ev_train": ev_tr, "base": base, "importancias": imp, "reglas": reglas,
            "nodos": int(clf.tree_.node_count), "profundidad": int(clf.get_depth()), "train": tr, "lineas": lineas}


def entrenar_red(train, test, objetivo, variables, neuronas=8, epocas=60, tasa=0.02, semilla=7) -> dict:
    tr, te = _preparar_supervisado(train, test, objetivo, variables)
    pre = _preprocesador(tr, variables, escalar=True)
    clf = MLPClassifier(hidden_layer_sizes=(int(neuronas),), activation="tanh", solver="adam",
                        learning_rate_init=float(tasa), max_iter=int(epocas), random_state=semilla)
    modelo = Pipeline([("pre", pre), ("clf", clf)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(_X(tr, variables), tr[objetivo].astype(str))
    clases = [str(c) for c in modelo.classes_]
    ev_te = _evaluar(te[objetivo].astype(str), modelo.predict(_X(te, variables)), clases)
    ev_tr = _evaluar(tr[objetivo].astype(str), modelo.predict(_X(tr, variables)), clases)
    curva = pd.Series(modelo.named_steps["clf"].loss_curve_, name="pérdida")
    curva.index = np.arange(1, len(curva) + 1)
    dim = len(pre.get_feature_names_out())
    lineas = [
        f"Objetivo: {objetivo}. {dim} entradas codificadas, {int(neuronas)} neuronas ocultas, {int(epocas)} épocas.",
        f"Exactitud en prueba {_p(ev_te['exactitud'])} (entrenamiento {_p(ev_tr['exactitud'])}). Pérdida final {_f(curva.iloc[-1], 3)}.",
    ]
    return {"tipo": "red", "titulo": "Red neuronal", "modelo": modelo, "objetivo": objetivo, "variables": variables,
            "clases": clases, "ev_test": ev_te, "ev_train": ev_tr, "curva": curva, "train": tr, "lineas": lineas}


def predecir_caso(resultado: dict, valores: dict) -> pd.Series:
    X = _X(pd.DataFrame([valores]), resultado["variables"])
    proba = resultado["modelo"].predict_proba(X)[0]
    return pd.Series(proba, index=resultado["clases"], name="probabilidad")


def valores_tipicos(train: pd.DataFrame, variables: list[str]) -> dict:
    out = {}
    for v in variables:
        out[v] = float(train[v].median()) if es_numerica(train[v]) else str(train[v].mode().iloc[0])
    return out


# --------------------------------------------------------------------------- Unidad 3: clústeres
def segmentar(df: pd.DataFrame, variables: list[str], k: int = 3, semilla: int = 11) -> dict:
    d = df.dropna(subset=variables).reset_index(drop=True)
    if len(d) < k * 5:
        raise ValueError("Hay muy pocas filas completas para este número de clústeres.")
    X = StandardScaler().fit_transform(d[variables])
    km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(X)
    etiquetas = km.labels_
    codo = pd.Series([KMeans(n_clusters=i, n_init=5, random_state=semilla).fit(X).inertia_ for i in range(1, 9)],
                     index=range(1, 9), name="inercia")
    sil = float(silhouette_score(X, etiquetas, sample_size=min(len(X), 800), random_state=semilla)) if k > 1 else None
    media, desv = d[variables].mean(), d[variables].std(ddof=1).replace(0, 1)
    filas, lineas = [], [f"{k} clústeres sobre {len(d)} registros con {', '.join(variables)}. "
                         f"Silueta {'n/d' if sil is None else _f(sil, 2)}."]
    for c in range(k):
        g = d[etiquetas == c]
        z = ((g[variables].mean() - media) / desv)
        marcados = z[z.abs() >= 0.5].reindex(z[z.abs() >= 0.5].abs().sort_values(ascending=False).index).head(3)
        rasgos = ", ".join(("alto " if v > 0 else "bajo ") + n for n, v in marcados.items()) or "cercano al promedio"
        
        perfil_comercial = _bautizar_cluster(g[variables].mean())
        
        fila = {"clúster": c + 1, "perfil_comercial": perfil_comercial, "registros": len(g), "%": round(len(g) / len(d) * 100, 1)}
        fila.update({v: round(float(g[v].mean()), 2) for v in variables})
        fila["rasgos distintivos"] = rasgos
        filas.append(fila)
        lineas.append(f"Clúster {c + 1} ({perfil_comercial}) - {_p(len(g) / len(d), 0)}: {rasgos}.")
    return {"tipo": "segmentos", "titulo": "Segmentación (k-medias con Buyer Personas)", "k": k, "variables": variables,
            "datos": d.assign(cluster=etiquetas + 1), "perfil": pd.DataFrame(filas), "codo": codo,
            "silueta": sil, "lineas": lineas}


# --------------------------------------------------------------------------- Unidad 3: series de tiempo
def holt(y, alfa: float, beta: float, h: int):
    y = np.asarray(y, float)
    nivel, tend = y[0], (y[1] - y[0] if len(y) > 1 else 0.0)
    ajuste = [np.nan]
    for t in range(1, len(y)):
        ajuste.append(nivel + tend)
        nuevo = alfa * y[t] + (1 - alfa) * (nivel + tend)
        tend = beta * (nuevo - nivel) + (1 - beta) * tend
        nivel = nuevo
    return np.array(ajuste), np.array([nivel + i * tend for i in range(1, h + 1)])


def _errores(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    ok = ~np.isnan(f)
    mae = float(np.mean(np.abs(y[ok] - f[ok]))) if ok.any() else float("nan")
    nz = ok & (y != 0)
    mape = float(np.mean(np.abs((y[nz] - f[nz]) / y[nz])) * 100) if nz.any() else float("nan")
    return mae, mape


def pronosticar(df, periodo, medida, agregacion="suma", ventana=3, alfa=0.4, beta=0.2, horizonte=6) -> dict:
    g = df.groupby(periodo)[medida]
    serie = (g.size() if agregacion == "conteo" else g.sum() if agregacion == "suma" else g.mean()).sort_index()
    if len(serie) < 6:
        raise ValueError("La columna de periodo tiene menos de 6 valores distintos. Elige otra columna.")
    y = serie.to_numpy(float)
    ajuste, fc = holt(y, alfa, beta, horizonte)
    media_movil = pd.Series(y).rolling(min(ventana, len(y))).mean().to_numpy()
    mae, mape = _errores(y, ajuste)
    mae_ref, _ = _errores(y[1:], y[:-1])
    if pd.api.types.is_numeric_dtype(serie.index):
        ult = serie.index[-1]
        etiquetas_fc = [str(ult + i + 1) for i in range(horizonte)]
    else:
        etiquetas_fc = [f"+{i + 1}" for i in range(horizonte)]
    etiquetas = [str(i) for i in serie.index]
    tabla = pd.DataFrame({
        "periodo": etiquetas + etiquetas_fc,
        "observado": list(y) + [np.nan] * horizonte,
        f"media móvil ({ventana})": list(media_movil) + [np.nan] * horizonte,
        "ajuste Holt": list(ajuste) + [np.nan] * horizonte,
        "pronóstico": [np.nan] * (len(y) - 1) + [y[-1]] + list(fc),
    }).set_index("periodo")
    nombre_agg = {"suma": "Suma", "promedio": "Promedio", "conteo": "Conteo"}[agregacion]
    sentido = "al alza" if fc[-1] > y[-1] else "a la baja"
    lineas = [f"{nombre_agg} de {medida} por {periodo}, {len(y)} periodos. Pronóstico a {horizonte} periodos {sentido}: de {_f(y[-1], 0)} a {_f(fc[-1], 0)}.",
              f"Error medio absoluto {_f(mae, 0)} ({_f(mape, 1)}% MAPE); repetir el valor anterior da {_f(mae_ref, 0)}."]
    return {"tipo": "serie", "titulo": "Series de tiempo (Holt)", "tabla": tabla, "pronostico": pd.Series(fc, index=etiquetas_fc),
            "mae": mae, "mape": mape, "mae_ref": mae_ref, "medida": medida, "periodo": periodo,
            "y_ultimo": float(y[-1]), "fc_ultimo": float(fc[-1]), "lineas": lineas}


# --------------------------------------------------------------------------- Unidad 3: asociación
def a_transacciones(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    partes = {}
    for c in columnas:
        s = df[c]
        if es_numerica(s):
            etq = ["bajo", "medio", "alto"]
            try:
                disc = pd.qcut(s, 3, labels=etq, duplicates="drop")
            except ValueError:
                disc = pd.cut(s, 3, labels=etq)
            valores = disc.astype(object)
        else:
            valores = s.astype(object)
        valores = valores.where(s.notna(), None)
        for v in pd.unique(valores.dropna()):
            partes[f"{c}={v}"] = (valores == v).to_numpy(bool)
    return pd.DataFrame(partes, index=df.index)


def _apriori(tx: pd.DataFrame, soporte_min: float, largo_max: int = 3):
    M = tx.to_numpy(bool)
    items = list(tx.columns)
    sup: dict[tuple, float] = {}
    nivel: list[tuple] = []
    for j in range(len(items)):
        s = float(M[:, j].mean())
        if s >= soporte_min:
            sup[(j,)] = s
            nivel.append((j,))
    for k in range(2, largo_max + 1):
        siguiente = []
        for a, b in itertools.combinations(nivel, 2):
            if a[:-1] != b[:-1]:
                continue
            u = a + (b[-1],)
            if not all(tuple(x for i, x in enumerate(u) if i != quitar) in sup for quitar in range(k)):
                continue
            s = float(M[:, list(u)].all(axis=1).mean())
            if s >= soporte_min:
                sup[u] = s
                siguiente.append(u)
        if not siguiente:
            break
        nivel = sorted(siguiente)
    return items, sup


def asociacion(df, columnas, soporte_min=0.10, confianza_min=0.60, largo_max=3) -> dict:
    tx = a_transacciones(df, columnas)
    items, sup = _apriori(tx, soporte_min, largo_max)
    conjuntos = pd.DataFrame([{"ítems": " y ".join(items[i] for i in k), "tamaño": len(k), "soporte": s}
                              for k, s in sup.items()]).sort_values(["tamaño", "soporte"], ascending=[True, False]) if sup else pd.DataFrame(columns=["ítems", "tamaño", "soporte"])
    reglas = []
    for k, s_full in sup.items():
        if len(k) < 2:
            continue
        for r in range(1, len(k)):
            for ant in itertools.combinations(k, r):
                cons = tuple(x for x in k if x not in ant)
                conf = s_full / sup[ant]
                if conf >= confianza_min:
                    reglas.append({"si": " y ".join(items[i] for i in ant), "entonces": " y ".join(items[i] for i in cons),
                                   "soporte": s_full, "confianza": conf, "lift": conf / sup[cons]})
    tabla = pd.DataFrame(reglas, columns=["si", "entonces", "soporte", "confianza", "lift"])
    tabla = tabla.sort_values(["lift", "confianza"], ascending=False).reset_index(drop=True)
    lineas = [f"{len(columnas)} columnas, soporte mínimo {_p(soporte_min, 0)}, confianza mínima {_p(confianza_min, 0)}: "
              f"{len(conjuntos)} conjuntos frecuentes y {len(tabla)} reglas."]
    mejor = tabla[tabla["lift"] > 1].head(1)
    if len(mejor):
        r = mejor.iloc[0]
        lineas.append(f"Regla destacada: si {r['si']}, entonces {r['entonces']} (confianza {_p(r['confianza'], 0)}, lift {_f(r['lift'], 2)}).")
    else:
        lineas.append("No se encontraron reglas con lift mayor que 1 con estos umbrales.")
    return {"tipo": "asociacion", "titulo": "Reglas de asociación (Apriori)", "reglas": tabla, "conjuntos": conjuntos,
            "columnas": columnas, "lineas": lineas}


# --------------------------------------------------------------------------- Unidad 4: informe
def construir_informe(nombre: str, objetivo: str, df: pd.DataFrame | None, n_archivos: int,
                      prep: dict | None, resultados: dict, fecha: str) -> str:
    sec = [f"# Informe de minería de datos (Consultoría Ejecutiva)\n\nConjunto de datos: {nombre or 'sin cargar'}.\nFecha: {fecha}.",
           f"## 1. Comprensión del negocio\n\n{objetivo.strip() or 'Sin definir. Escribe la pregunta de negocio en la pestaña Datos.'}"]
    if df is None:
        sec.append("## 2. Datos\n\nAún no se han cargado datos.")
        return "\n\n".join(sec) + "\n"
    tip = tipos(df)
    n_num = sum(t == "num" for t in tip.values())
    texto = (f"{_f(len(df), 0)} filas y {df.shape[1]} columnas ({n_num} numéricas, {df.shape[1] - n_num} categóricas) "
             f"con {_f(int(df.isna().sum().sum()), 0)} celdas vacías.")
    if n_archivos > 1:
        texto += f"\nSe integraron {n_archivos} archivos alineando las columnas por nombre."
    sec.append(f"## 2. Comprensión de los datos\n\n{texto}")
    sec.append("## 3. Preparación de los datos\n\n" + ("\n".join("- " + x for x in prep["log"]) if prep else "Aún no se aplicó la preparación."))

    bloques = []
    for k in ["arbol", "red", "segmentos", "serie", "asociacion"]:
        if k in resultados:
            bloques.append(f"**{resultados[k]['titulo']}**\n" + "\n".join("- " + l for l in resultados[k]["lineas"]))
    sec.append("## 4. Modelado y evaluación\n\n" + ("\n\n".join(bloques) if bloques else "Aún no se ha ejecutado ninguna técnica."))

    con = []
    a, r = resultados.get("arbol"), resultados.get("red")
    if a and r:
        ea, er = a["ev_test"]["exactitud"], r["ev_test"]["exactitud"]
        con.append(f"- En los casos de prueba, {'el árbol de decisión' if ea >= er else 'la red neuronal'} predijo mejor ({_p(max(ea, er))} frente a {_p(min(ea, er))}).")
    if a or r:
        mejor = max(x["ev_test"]["exactitud"] for x in (a, r) if x)
        base = (a or r).get("base") if a else None
        if base is not None and mejor < base + 0.03:
            con.append("- Ningún clasificador supera con claridad la referencia de predecir siempre la clase más frecuente; conviene revisar las variables o conseguir más datos.")
        elif r and not a:
            con.append(f"- La red neuronal alcanzó {_p(r['ev_test']['exactitud'])} de exactitud en prueba.")
    if "segmentos" in resultados:
        con.append(f"- Los registros se organizan en {resultados['segmentos']['k']} Buyer Personas con perfiles comerciales diferenciados.")
    if "serie" in resultados:
        s = resultados["serie"]
        con.append(f"- La serie de {s['medida']} por {s['periodo']} se proyecta {'al alza' if s['fc_ultimo'] > s['y_ultimo'] else 'a la baja'} en los próximos periodos.")
    if "asociacion" in resultados and len(resultados["asociacion"]["reglas"]):
        con.append(f"- Se hallaron {len(resultados['asociacion']['reglas'])} reglas de asociación ejecutivas para optimización de estrategias.")
    sec.append("## 5. Conclusiones y Recomendaciones\n\n" + ("\n".join(con) if con else "Ejecuta al menos una técnica para generar conclusiones."))
    
    # SECCIÓN 6 CORREGIDA (Sin errores de sintaxis o strings rotos)
    sec.append(
        "## 6. Limitaciones\n\n"
        "- La evaluación usa una sola partición de entrenamiento y prueba; con pocos datos los resultados pueden variar con otra semilla.\n"
        "- Una asociación o un patrón no demuestra causa; valida las hipótesis con conocimiento del contexto.\n"
        "- Los resultados dependen de la calidad y representatividad de los datos cargados."
    )
    return "\n\n".join(sec) + "\n"