import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Compatibilidad sklearn (sparse_output vs sparse) ──────────────────────
import sklearn
SKLEARN_NEW = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2)
OHE_KWARGS = {"handle_unknown": "ignore", "sparse_output": False} if SKLEARN_NEW \
             else {"handle_unknown": "ignore", "sparse": False}

RANDOM_STATE = 42

st.set_page_config(
    page_title="GIAM – Costos Oncológicos SIS",
    page_icon="🩺",
    layout="wide",
)

# ── Constantes ────────────────────────────────────────────────────────────
DATA_FILE     = "Data_Pacientes_Oncologicos.xlsx"
TARGET_COL    = "MONTO_BRUTO"
MODEL_COLS    = ["EDAD","SEXO","REGION","TIPO_SEGURO","SERVICIO","GRUPO_DIAGNOSTICOS"]
NUM_FEATURES  = ["EDAD"]
CAT_FEATURES  = ["SEXO","REGION","TIPO_SEGURO","SERVICIO","GRUPO_DIAGNOSTICOS"]


# ─────────────────────────────────────────────────────────────────────────
# CARGA Y LIMPIEZA  (cache_data → hasheable sin problemas)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Cargando dataset...")
def load_data():
    try:
        df = pd.read_excel(DATA_FILE, sheet_name="Hoja1", engine="openpyxl")
    except Exception:
        df = pd.read_excel(DATA_FILE, engine="openpyxl")

    df_c = df.drop_duplicates().reset_index(drop=True)

    # Imputación MONTO_BRUTO
    df_c[TARGET_COL] = pd.to_numeric(df_c[TARGET_COL], errors="coerce")
    df_c[TARGET_COL] = df_c[TARGET_COL].fillna(df_c[TARGET_COL].median())

    # Imputación categóricas
    for col in CAT_FEATURES:
        if col in df_c.columns and df_c[col].isna().any():
            df_c[col] = df_c[col].fillna(df_c[col].mode()[0])

    # Capping IQR en EDAD
    if "EDAD" in df_c.columns:
        df_c["EDAD"] = pd.to_numeric(df_c["EDAD"], errors="coerce")
        q1, q3 = df_c["EDAD"].quantile(0.25), df_c["EDAD"].quantile(0.75)
        iqr = q3 - q1
        df_c["EDAD"] = df_c["EDAD"].clip(lower=max(q1 - 1.5*iqr, 0), upper=q3 + 1.5*iqr)

    cols_ok = [c for c in MODEL_COLS if c in df_c.columns]
    df_model = df_c[cols_ok + [TARGET_COL]].dropna()
    return df, df_c, df_model


# ─────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO  (cache_data serializa bien con sklearn)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Entrenando modelos ML (1-2 min)...")
def train_models(_df_model):          # prefijo _ → Streamlit no hashea el df
    y_log = np.log1p(_df_model[TARGET_COL])
    X     = _df_model.drop(columns=[TARGET_COL])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(**OHE_KWARGS), CAT_FEATURES),
    ])

    candidates = {
        "Linear Regression" : LinearRegression(),
        "Decision Tree"     : DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest"     : RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Bagging"           : BaggingRegressor(n_estimators=30, random_state=RANDOM_STATE, n_jobs=1),
    }

    fitted, metrics = {}, []
    for name, est in candidates.items():
        pipe = Pipeline([("pre", preprocessor), ("model", est)])
        pipe.fit(X_tr, y_tr)
        yp = pipe.predict(X_te)
        fitted[name] = pipe
        metrics.append({
            "Modelo"   : name,
            "RMSE_test": float(np.sqrt(mean_squared_error(y_te, yp))),
            "MAE_test" : float(mean_absolute_error(y_te, yp)),
            "R2_test"  : float(r2_score(y_te, yp)),
        })

    # GridSearchCV reducido para no hacer timeout
    gb_pipe  = Pipeline([("pre", preprocessor), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    grid     = GridSearchCV(
        gb_pipe,
        {"model__n_estimators":[100,200], "model__learning_rate":[0.05,0.1], "model__max_depth":[3,5]},
        cv=3, scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    grid.fit(X_tr, y_tr)
    best_gb = grid.best_estimator_
    yp_gb   = best_gb.predict(X_te)
    fitted["GB Optimizado"] = best_gb
    metrics.append({
        "Modelo"   : "GB Optimizado",
        "RMSE_test": float(np.sqrt(mean_squared_error(y_te, yp_gb))),
        "MAE_test" : float(mean_absolute_error(y_te, yp_gb)),
        "R2_test"  : float(r2_score(y_te, yp_gb)),
    })

    df_met    = pd.DataFrame(metrics).sort_values("RMSE_test").reset_index(drop=True)
    best_name = df_met.iloc[0]["Modelo"]

    # Importancia de variables (Random Forest)
    rf    = fitted["Random Forest"]
    feats = rf.named_steps["pre"].get_feature_names_out()
    imp   = (pd.DataFrame({"variable": feats,
                            "importancia": rf.named_steps["model"].feature_importances_})
               .sort_values("importancia", ascending=False).head(15))

    return fitted, df_met, best_name, imp, grid.best_params_


@st.cache_data(show_spinner="🔵 Calculando clusters K-Means...")
def train_clustering(_df_model, k=3):
    Xc  = _df_model[["EDAD", TARGET_COL]].dropna()
    sc  = StandardScaler()
    Xcs = sc.fit_transform(Xc)

    sil = {}
    for kk in range(2, 7):
        km       = KMeans(n_clusters=kk, random_state=RANDOM_STATE, n_init=10).fit(Xcs)
        sil[kk]  = float(silhouette_score(Xcs, km.labels_))

    km_final = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df_cl    = Xc.copy()
    df_cl["cluster"] = km_final.fit_predict(Xcs).astype(str)   # str → seaborn no falla

    stats = (df_cl.assign(cluster_int=df_cl["cluster"].astype(int))
               .groupby("cluster_int")
               .agg(n_pacientes=("EDAD","count"),
                    edad_media=("EDAD","mean"),
                    monto_medio=(TARGET_COL,"mean"),
                    monto_mediana=(TARGET_COL,"median"))
               .round(2))
    return km_final, sc, df_cl, stats, sil


# ─────────────────────────────────────────────────────────────────────────
# CARGA INICIAL
# ─────────────────────────────────────────────────────────────────────────
try:
    df_raw, df_clean, df_model = load_data()
except Exception as e:
    st.error(f"""
**Error al leer el archivo Excel.**

Verifica que `{DATA_FILE}` está subido en la raíz de tu repositorio de GitHub y que el nombre del archivo es exactamente ese.

Detalle técnico: `{e}`
""")
    st.stop()

if df_model.empty:
    st.error("El dataset quedó vacío tras la limpieza. Revisa que las columnas del Excel coincidan con las esperadas.")
    st.stop()

fitted_models, metrics_df, best_name, importancias, best_gb_params = train_models(df_model)
kmeans, cluster_scaler, df_cluster, cluster_stats, sil_scores     = train_clustering(df_model)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.title("🩺 GIAM – Grupo 5")
st.sidebar.markdown("Factores asociados al costo de atención oncológica (SIS, Perú 2023–2025).")
seccion = st.sidebar.radio("Navegación", [
    "🏠 Resumen del proyecto",
    "📊 Exploración de datos (EDA)",
    "💰 Predictor de costo",
    "🔵 Segmentación de pacientes",
    "📈 Comparación de modelos",
    "📝 Conclusiones",
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Registros totales: {len(df_raw):,}")
st.sidebar.caption(f"Usados en modelo:  {len(df_model):,}")
st.sidebar.caption(f"Mejor modelo: {best_name}")


# ─────────────────────────────────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────────────────────────────────
if seccion == "🏠 Resumen del proyecto":
    st.title("Análisis de Factores Asociados al Costo de Atención Oncológica (SIS)")
    st.markdown("""
App del **Grupo 5 – GIAM**. Se modela `MONTO_BRUTO` con regresión supervisada
(5 algoritmos + GridSearchCV) y clustering K-Means para segmentar perfiles de pacientes.

**Objetivo:** anticipar el costo esperado de una atención oncológica para apoyar la planificación presupuestaria del SIS.
""")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total registros",    f"{len(df_raw):,}")
    c2.metric("Costo medio (S/)",   f"{df_raw[TARGET_COL].mean():,.2f}")
    c3.metric("Costo mediano (S/)", f"{df_raw[TARGET_COL].median():,.2f}")
    c4.metric("Costo máximo (S/)",  f"{df_raw[TARGET_COL].max():,.2f}")
    st.markdown("### Vista previa del dataset")
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.markdown("### Diccionario de variables del modelo")
    st.table(pd.DataFrame({
        "Variable"       : MODEL_COLS + [TARGET_COL],
        "Tipo"           : ["Numérica","Categórica","Categórica","Categórica","Categórica","Categórica","Numérica (target)"],
        "Descripción"    : ["Edad del paciente","Sexo","Región de atención","Modalidad del SIS",
                            "Tipo de servicio oncológico","Grupo de diagnóstico CIE-10",
                            "Costo bruto de la atención (S/)"],
    }))


# ─────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────
elif seccion == "📊 Exploración de datos (EDA)":
    st.title("Exploración de datos (EDA)")

    st.subheader("Valores nulos por columna (%)")
    nulos = (df_raw.isna().mean()*100).sort_values(ascending=False).round(2)
    nulos_show = nulos[nulos > 0]
    if nulos_show.empty:
        st.success("No hay valores nulos en el dataset.")
    else:
        st.dataframe(nulos_show.rename("% nulos"), use_container_width=True)

    st.subheader("Distribución de MONTO_BRUTO")
    c1,c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df_raw[TARGET_COL].dropna(), bins=50, edgecolor="white")
        ax.set_title("Escala original"); ax.set_xlabel("S/")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(np.log1p(df_raw[TARGET_COL].dropna()), bins=50, color="steelblue", edgecolor="white")
        ax.set_title("Escala log1p"); ax.set_xlabel("log(S/ + 1)")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Costo medio por categoría (bivariado vs. MONTO_BRUTO)")
    cat_col = st.selectbox("Variable categórica", CAT_FEATURES)
    if cat_col in df_model.columns:
        resumen = (df_model.groupby(cat_col)[TARGET_COL]
                   .agg(["mean","median","count"])
                   .sort_values("mean", ascending=False))
        resumen = resumen[resumen["count"] >= 10]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4*len(resumen))))
        sns.barplot(x=resumen["mean"], y=resumen.index.astype(str),
                    palette="viridis", ax=ax)
        ax.set_title(f"Costo medio por {cat_col}"); ax.set_xlabel("S/")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Correlación entre variables numéricas")
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(max(6, len(num_cols)*1.2), max(5, len(num_cols)*1.0)))
        sns.heatmap(df_raw[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Solo hay una columna numérica; no se puede calcular correlación.")


# ─────────────────────────────────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────────────────────────────────
elif seccion == "💰 Predictor de costo":
    st.title("Predictor de costo de atención (MONTO_BRUTO)")
    r2_b   = metrics_df.set_index("Modelo").loc[best_name, "R2_test"]
    rmse_b = metrics_df.set_index("Modelo").loc[best_name, "RMSE_test"]
    st.markdown(f"**Modelo recomendado:** {best_name} — R²=`{r2_b:.3f}` | RMSE=`{rmse_b:.3f}`")
    st.info("El modelo opera en escala log1p. La predicción se convierte automáticamente a S/ con `expm1`.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.slider("Edad del paciente", 0, 100, 55)
            sexo = st.selectbox("Sexo", sorted(df_model["SEXO"].dropna().unique().tolist()))
            region = st.selectbox("Región", sorted(df_model["REGION"].dropna().unique().tolist()))
        with c2:
            tipo_seguro = st.selectbox("Tipo de seguro", sorted(df_model["TIPO_SEGURO"].dropna().unique().tolist()))
            servicio    = st.selectbox("Servicio",        sorted(df_model["SERVICIO"].dropna().unique().tolist()))
            grupo_dx    = st.selectbox("Grupo diagnóstico", sorted(df_model["GRUPO_DIAGNOSTICOS"].dropna().unique().tolist()))
        modelo_elegido = st.selectbox("Modelo a usar", list(fitted_models.keys()),
                                      index=list(fitted_models.keys()).index(best_name))
        submitted = st.form_submit_button("🔍 Predecir costo")

    if submitted:
        entrada = pd.DataFrame([{
            "EDAD": edad, "SEXO": sexo, "REGION": region,
            "TIPO_SEGURO": tipo_seguro, "SERVICIO": servicio,
            "GRUPO_DIAGNOSTICOS": grupo_dx,
        }])
        try:
            pred_log   = fitted_models[modelo_elegido].predict(entrada)[0]
            pred_soles = np.expm1(pred_log)
            st.success(f"💰 Costo estimado: **S/ {pred_soles:,.2f}**")
            st.caption(f"Valor en escala log1p: {pred_log:.4f}")
        except Exception as ex:
            st.error(f"Error al predecir: {ex}")


# ─────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────
elif seccion == "🔵 Segmentación de pacientes":
    st.title("Segmentación de pacientes — K-Means (K = 3)")

    st.subheader("Elección del K óptimo (coeficiente de silueta)")
    sil_df = pd.DataFrame({"K": list(sil_scores.keys()), "Silhouette": list(sil_scores.values())})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sil_df["K"], sil_df["Silhouette"], "go-", linewidth=2)
    ax.axvline(3, color="red", linestyle="--", alpha=0.6, label="K = 3 (elegido)")
    ax.set_xlabel("Número de clusters K"); ax.set_ylabel("Silhouette score"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Perfil de cada cluster")
    st.dataframe(cluster_stats, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_cluster, x="EDAD", y=TARGET_COL,
                    hue="cluster", palette="Set1", alpha=0.4, s=18, ax=ax)
    ax.set_title("Clusters: Edad vs Monto Bruto")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Asignar un paciente nuevo a su cluster")
    c1, c2 = st.columns(2)
    edad_c  = c1.slider("Edad", 0, 100, 55, key="cl_edad")
    monto_c = c2.number_input("Monto de atención estimado (S/)", min_value=0.0, value=100.0, step=10.0)
    if st.button("Asignar cluster"):
        punto  = cluster_scaler.transform([[edad_c, monto_c]])
        cid    = int(kmeans.predict(punto)[0])
        st.success(f"🧩 Cluster asignado: **{cid}**")
        if cid in cluster_stats.index:
            st.dataframe(cluster_stats.loc[[cid]], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# COMPARACIÓN DE MODELOS
# ─────────────────────────────────────────────────────────────────────────
elif seccion == "📈 Comparación de modelos":
    st.title("Comparación de modelos de regresión")
    st.markdown(f"**6 modelos** entrenados. Mejores parámetros GB optimizado: `{best_gb_params}`")

    st.dataframe(
        metrics_df.style.format({"RMSE_test": "{:.4f}", "MAE_test": "{:.4f}", "R2_test": "{:.4f}"}),
        use_container_width=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.barplot(data=metrics_df, x="RMSE_test", y="Modelo", ax=axes[0], palette="Blues_r")
    axes[0].set_title("RMSE (↓ mejor)")
    sns.barplot(data=metrics_df, x="MAE_test",  y="Modelo", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("MAE (↓ mejor)")
    sns.barplot(data=metrics_df, x="R2_test",   y="Modelo", ax=axes[2], palette="Greens_r")
    axes[2].set_title("R² (↑ mejor)")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Importancia de variables (Random Forest)")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importancias, x="importancia", y="variable", color="steelblue", ax=ax)
    ax.set_title("Top 15 variables más importantes")
    st.pyplot(fig); plt.close(fig)

    st.info(f"✅ Mejor modelo: **{best_name}** — R² = {metrics_df.iloc[0]['R2_test']:.3f} | RMSE = {metrics_df.iloc[0]['RMSE_test']:.3f}")


# ─────────────────────────────────────────────────────────────────────────
# CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────
elif seccion == "📝 Conclusiones":
    st.title("Conclusiones")
    r2_f    = metrics_df.iloc[0]['R2_test']
    rmse_f  = metrics_df.iloc[0]['RMSE_test']
    med_val = df_raw[TARGET_COL].median()
    max_val = df_raw[TARGET_COL].max()
    st.markdown(f"""
1. El dataset contiene **{len(df_raw):,} registros** con alta calidad en las variables del modelo.
2. `MONTO_BRUTO` tiene distribución fuertemente asimétrica (mediana S/ {med_val:,.2f} vs. máximo S/ {max_val:,.2f}),
   lo que justificó aplicar la transformación `log1p` antes del modelado.
3. Ninguna variable numérica correlaciona linealmente con el costo (|r| < 0.05);
   los factores determinantes son **categóricos** (`SERVICIO`, `GRUPO_DIAGNOSTICOS`, `REGION`) y sus interacciones.
4. El mejor modelo fue **{best_name}** con R² = **{r2_f:.3f}** y RMSE = **{rmse_f:.3f}** (escala log).
5. Según el Random Forest, **`SERVICIO`** es la variable más importante, seguida de `GRUPO_DIAGNOSTICOS` y `EDAD`.
6. El clustering K-Means identificó **3 perfiles** de pacientes:
   ambulatorio de bajo costo, costo intermedio, y **alto costo** (cirugías/hospitalización).
7. Variables ausentes del dataset (etapa clínica del cáncer, número de procedimientos, duración de internamiento)
   probablemente explican la varianza residual no capturada por los modelos actuales.
""")
