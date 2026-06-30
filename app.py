import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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

RANDOM_STATE = 42
st.set_page_config(page_title="GIAM - Costos Oncológicos SIS", page_icon="🩺", layout="wide")

DATA_FILE = "Data_Pacientes_Oncologicos.xlsx" 
MODEL_COLS = ["EDAD","SEXO","REGION","TIPO_SEGURO","SERVICIO","GRUPO_DIAGNOSTICOS"]
TARGET_COL = "MONTO_BRUTO"
NUMERIC_FEATURES = ["EDAD"]
CATEGORICAL_FEATURES = ["SEXO","REGION","TIPO_SEGURO","SERVICIO","GRUPO_DIAGNOSTICOS"]

@st.cache_data(show_spinner="Descargando y preparando el dataset...")
def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name="Hoja1")
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_clean[TARGET_COL] = df_clean[TARGET_COL].fillna(df_clean[TARGET_COL].median())
    for col in CATEGORICAL_FEATURES:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    q1, q3 = df_clean["EDAD"].quantile(0.25), df_clean["EDAD"].quantile(0.75)
    iqr = q3 - q1
    df_clean["EDAD"] = df_clean["EDAD"].clip(lower=max(q1 - 1.5*iqr, 0), upper=q3 + 1.5*iqr)
    df_model = df_clean[MODEL_COLS + [TARGET_COL]].dropna()
    return df, df_clean, df_model

@st.cache_resource(show_spinner="Entrenando modelos (puede tardar 1-2 minutos)...")
def train_models(df_model):
    y_log = np.log1p(df_model[TARGET_COL])
    X = df_model.drop(columns=[TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Bagging": BaggingRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
    }
    fitted, metrics = {}, []
    for name, est in candidate_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fitted[name] = pipe
        metrics.append({"Modelo": name,
                        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "MAE_test": mean_absolute_error(y_test, y_pred),
                        "R2_test": r2_score(y_test, y_pred)})
    # GridSearchCV Gradient Boosting
    gb_pipe = Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    param_grid = {"model__n_estimators":[50,100,200], "model__learning_rate":[0.05,0.1], "model__max_depth":[3,5]}
    grid = GridSearchCV(gb_pipe, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_gb = grid.best_estimator_
    y_pred_gb = best_gb.predict(X_test)
    fitted["Gradient Boosting (optimizado)"] = best_gb
    metrics.append({"Modelo": "Gradient Boosting (optimizado)",
                    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_gb)),
                    "MAE_test": mean_absolute_error(y_test, y_pred_gb),
                    "R2_test": r2_score(y_test, y_pred_gb)})
    metrics_df = pd.DataFrame(metrics).sort_values("RMSE_test").reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Modelo"]
    # Importancia variables RF
    rf_pipe = fitted["Random Forest"]
    feat_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    importancias = pd.DataFrame({"variable": feat_names,
                                  "importancia": rf_pipe.named_steps["model"].feature_importances_})\
                    .sort_values("importancia", ascending=False).head(15)
    return fitted, metrics_df, best_name, fitted[best_name], importancias, grid.best_params_

@st.cache_resource(show_spinner="Calculando segmentación K-Means...")
def train_clustering(df_model, k=3):
    Xc = df_model[["EDAD", TARGET_COL]].dropna()
    scaler = StandardScaler()
    Xcs = scaler.fit_transform(Xc)
    sil_scores = {}
    for kk in range(2, 8):
        km = KMeans(n_clusters=kk, random_state=RANDOM_STATE, n_init=10).fit(Xcs)
        sil_scores[kk] = silhouette_score(Xcs, km.labels_)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df_cl = Xc.copy()
    df_cl["cluster"] = kmeans.fit_predict(Xcs)
    stats = df_cl.groupby("cluster").agg(
        n_pacientes=("EDAD","count"), edad_media=("EDAD","mean"),
        monto_medio=(TARGET_COL,"mean"), monto_mediana=(TARGET_COL,"median")).round(2)
    return kmeans, scaler, df_cl, stats, sil_scores

try:
    df_raw, df_clean, df_model = load_data()
except Exception as e:
    st.error(f"No se pudo descargar el dataset. Detalle: {e}")
    st.stop()

fitted_models, metrics_df, best_name, best_model, importancias, best_gb_params = train_models(df_model)
kmeans, cluster_scaler, df_cluster, cluster_stats, sil_scores = train_clustering(df_model)

# ── SIDEBAR ──
st.sidebar.title("🩺 GIAM – Grupo 5")
st.sidebar.markdown("**Factores asociados al costo de atención oncológica (SIS, Perú 2023–2025).**")
seccion = st.sidebar.radio("Navegación", [
    "Resumen del proyecto", "Exploración de datos (EDA)",
    "Predictor de costo", "Segmentación de pacientes",
    "Comparación de modelos", "Conclusiones"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Registros cargados: {len(df_raw):,}")
st.sidebar.caption(f"Usados en modelo: {len(df_model):,}")
st.sidebar.caption(f"Mejor modelo: {best_name}")

# ── RESUMEN ──
if seccion == "Resumen del proyecto":
    st.title("Análisis de Factores Asociados al Costo de Atención Oncológica (SIS)")
    st.markdown("""
App del **Grupo 5 – GIAM**. Se modela `MONTO_BRUTO` con regresión supervisada (5 algoritmos + GridSearch)
y clustering K-Means para segmentar perfiles de pacientes.

**Objetivo:** anticipar el costo esperado de una atención oncológica para apoyar la planificación presupuestaria del SIS.
""")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Registros", f"{len(df_raw):,}")
    c2.metric("Costo medio (S/)", f"{df_raw[TARGET_COL].mean():,.2f}")
    c3.metric("Costo mediano (S/)", f"{df_raw[TARGET_COL].median():,.2f}")
    c4.metric("Costo máximo (S/)", f"{df_raw[TARGET_COL].max():,.2f}")
    st.markdown("### Vista previa del dataset")
    st.dataframe(df_raw.head(20), use_container_width=True)

# ── EDA ──
elif seccion == "Exploración de datos (EDA)":
    st.title("Exploración de datos (EDA)")
    st.subheader("Valores nulos por columna (%)")
    nulos = (df_raw.isna().mean()*100).sort_values(ascending=False).round(2)
    st.dataframe(nulos[nulos>0].rename("% nulos"), use_container_width=True)

    st.subheader("Distribución de MONTO_BRUTO")
    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(6,4))
        df_raw[TARGET_COL].hist(bins=50, ax=ax)
        ax.set_title("Escala original"); st.pyplot(fig)
    with c2:
        fig,ax = plt.subplots(figsize=(6,4))
        np.log1p(df_raw[TARGET_COL]).hist(bins=50, ax=ax, color="steelblue")
        ax.set_title("Escala log1p"); st.pyplot(fig)

    st.subheader("Análisis bivariado: costo medio por categoría vs. MONTO_BRUTO")
    cat_col = st.selectbox("Variable categórica", CATEGORICAL_FEATURES)
    resumen = df_model.groupby(cat_col)[TARGET_COL].agg(["mean","median","count"]).sort_values("mean", ascending=False)
    resumen = resumen[resumen["count"]>=10]
    fig,ax = plt.subplots(figsize=(8, max(3, 0.35*len(resumen))))
    sns.barplot(x=resumen["mean"], y=resumen.index, palette="viridis", ax=ax)
    ax.set_title(f"Costo medio por {cat_col}"); ax.set_xlabel("S/")
    st.pyplot(fig)
    st.caption("Las variables categóricas explican la variabilidad del costo mejor que las numéricas.")

    st.subheader("Correlación entre variables numéricas")
    num_cols = df_raw.select_dtypes(include=np.number).columns
    if len(num_cols)>1:
        fig,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df_raw[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ── PREDICTOR ──
elif seccion == "Predictor de costo":
    st.title("Predictor de costo de atención (MONTO_BRUTO)")
    r2_best = metrics_df.set_index("Modelo").loc[best_name,"R2_test"]
    rmse_best = metrics_df.set_index("Modelo").loc[best_name,"RMSE_test"]
    st.markdown(f"Modelo recomendado: **{best_name}** — R²={r2_best:.3f} | RMSE={rmse_best:.3f}")
    st.info("El modelo opera en escala log1p. La predicción ya está convertida de vuelta a S/.")

    with st.form("predict_form"):
        c1,c2 = st.columns(2)
        with c1:
            edad = st.slider("Edad", 0, 100, 63)
            sexo = st.selectbox("Sexo", sorted(df_model["SEXO"].dropna().unique()))
            region = st.selectbox("Región", sorted(df_model["REGION"].dropna().unique()))
        with c2:
            tipo_seguro = st.selectbox("Tipo de seguro", sorted(df_model["TIPO_SEGURO"].dropna().unique()))
            servicio = st.selectbox("Servicio", sorted(df_model["SERVICIO"].dropna().unique()))
            grupo_dx = st.selectbox("Grupo diagnóstico", sorted(df_model["GRUPO_DIAGNOSTICOS"].dropna().unique()))
        modelo_elegido = st.selectbox("Modelo", list(fitted_models.keys()),
                                       index=list(fitted_models.keys()).index(best_name))
        submitted = st.form_submit_button("Predecir costo")

    if submitted:
        entrada = pd.DataFrame([{"EDAD":edad,"SEXO":sexo,"REGION":region,
                                  "TIPO_SEGURO":tipo_seguro,"SERVICIO":servicio,"GRUPO_DIAGNOSTICOS":grupo_dx}])
        pred_log = fitted_models[modelo_elegido].predict(entrada)[0]
        pred_soles = np.expm1(pred_log)
        st.success(f"💰 Costo estimado: **S/ {pred_soles:,.2f}**")
        st.caption(f"log1p predicho: {pred_log:.3f}")

# ── CLUSTERING ──
elif seccion == "Segmentación de pacientes":
    st.title("Segmentación de pacientes (K-Means, K=3)")
    st.subheader("Coeficiente de silueta por K")
    sil_df = pd.DataFrame({"K": list(sil_scores.keys()), "Silhouette": list(sil_scores.values())})
    fig,ax = plt.subplots(figsize=(7,3.5))
    ax.plot(sil_df["K"], sil_df["Silhouette"], "go-")
    ax.axvline(3, color="red", linestyle="--", alpha=0.5, label="K=3 (elegido)")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.legend()
    st.pyplot(fig)

    st.subheader("Perfil de cada cluster")
    st.dataframe(cluster_stats, use_container_width=True)

    fig,ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df_cluster, x="EDAD", y=TARGET_COL, hue="cluster",
                    palette="Set1", alpha=0.4, s=20, ax=ax)
    ax.set_title("Clusters: Edad vs Monto Bruto"); st.pyplot(fig)

    st.subheader("Asignar un paciente nuevo a su cluster")
    c1,c2 = st.columns(2)
    edad_c = c1.slider("Edad", 0, 100, 63, key="cl_edad")
    monto_c = c2.number_input("Monto estimado (S/)", min_value=0.0, value=100.0, step=10.0)
    if st.button("Asignar cluster"):
        punto = cluster_scaler.transform([[edad_c, monto_c]])
        cid = int(kmeans.predict(punto)[0])
        st.success(f"🧩 Cluster asignado: **{cid}**")
        st.dataframe(cluster_stats.loc[[cid]], use_container_width=True)

# ── COMPARACIÓN ──
elif seccion == "Comparación de modelos":
    st.title("Comparación de modelos de regresión")
    st.markdown(f"**6 modelos** entrenados. Mejor GB params: `{best_gb_params}`")
    st.dataframe(metrics_df.style.format({"RMSE_test":"{:.4f}","MAE_test":"{:.4f}","R2_test":"{:.4f}"}),
                 use_container_width=True)
    fig,axes = plt.subplots(1,3,figsize=(16,4))
    sns.barplot(data=metrics_df, x="RMSE_test", y="Modelo", ax=axes[0], palette="Blues_r")
    axes[0].set_title("RMSE (menor=mejor)")
    sns.barplot(data=metrics_df, x="MAE_test", y="Modelo", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("MAE (menor=mejor)")
    sns.barplot(data=metrics_df, x="R2_test", y="Modelo", ax=axes[2], palette="Greens_r")
    axes[2].set_title("R² (mayor=mejor)")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Importancia de variables (Random Forest)")
    fig,ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importancias, x="importancia", y="variable", color="steelblue", ax=ax)
    ax.set_title("Top 15 variables más importantes"); st.pyplot(fig)
    st.info(f"Mejor modelo: **{best_name}** — R²={metrics_df.iloc[0]['R2_test']:.3f}")

# ── CONCLUSIONES ──
elif seccion == "Conclusiones":
    st.title("Conclusiones")
    st.markdown(f"""
- Dataset con **{len(df_raw):,} registros**; alta calidad en variables del modelo.
- `MONTO_BRUTO` sesgado a la derecha (mediana S/ {df_raw[TARGET_COL].median():,.2f} vs máximo S/ {df_raw[TARGET_COL].max():,.2f}); se usó transformación `log1p`.
- Correlación numérica con el costo: |r| < 0.05 → los determinantes son **categóricos** (`SERVICIO`, `GRUPO_DIAGNOSTICOS`, `REGION`).
- Mejor modelo: **{best_name}** (R²={metrics_df.iloc[0]['R2_test']:.3f}, RMSE={metrics_df.iloc[0]['RMSE_test']:.3f}).
- `SERVICIO` es la variable más importante según Random Forest, seguida de `GRUPO_DIAGNOSTICOS` y `EDAD`.
- K-Means identificó **3 perfiles** de pacientes: ambulatorio bajo costo, intermedio, y alto costo (cirugías/hospitalización).
- Variables ausentes (etapa del cáncer, n° de procedimientos, duración de internamiento) explican la varianza residual del modelo.
""")
