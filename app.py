import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, json

st.set_page_config(page_title="MVP Leguminosas · MS sales", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
META_PATH = BASE_DIR / "metadata.json"

FILES = {
    "Soja": DATA_DIR / "soja_sales_discrete.csv",
    "Poroto": DATA_DIR / "poroto_sales_discrete.csv",
    "Lupino": DATA_DIR / "lupino_sales_discrete.csv",
}

FEATURES = [
    "NH4NO3_mg_L","KNO3_mg_L","CaCl2_2H2O_mg_L",
    "MgSO4_7H2O_mg_L","KH2PO4_mg_L",
    "luminosity_umol_m2_s","temp_C","humidity_%"
]
TARGET = "protein"

DISCRETE_LEVELS = {
    "NH4NO3_mg_L": [800, 1200, 1600, 2000, 2400],
    "KNO3_mg_L": [950, 1400, 1900, 2350, 2850],
    "CaCl2_2H2O_mg_L": [220, 330, 440, 550, 660],
    "MgSO4_7H2O_mg_L": [185, 277, 370, 462, 555],
    "KH2PO4_mg_L": [85, 128, 170, 213, 255],
    "luminosity_umol_m2_s": [200, 350, 500, 650, 800, 950],
}
CONTINUOUS_RANGES = {"temp_C": (20.0, 30.0), "humidity_%": (40.0, 80.0)}

@st.cache_data
def load_data():
    dfs = []
    for name, path in FILES.items():
        df = pd.read_csv(path)
        df["Species"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

data = load_data()

# ---------------- Model loader (si hay .pkl) ----------------
def load_pickled_models():
    models = {}
    if MODELS_DIR.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        for name, spec in meta.get("models", {}).items():
            p = BASE_DIR / spec["path"]
            if p.exists():
                models[name] = {"model": joblib.load(p), "params": spec["hyperparams"]}
    return models

@st.cache_resource
def train_models(dataframe: pd.DataFrame):
    # Si hay .pkl, los usa; si no, entrena al vuelo
    pickled = load_pickled_models()
    X = dataframe[FEATURES]; y = dataframe[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(pickled) > 0:
        models = pickled
    else:
        models = {}
        # Ajusta n_estimators aquí si quieres acelerar el deploy
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_train, y_train)
        models["ElasticNet"] = {"model": enet, "params": {"alpha": 0.1, "l1_ratio": 0.5, "random_state": 42}}

        poly_model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                               ("lin", LinearRegression())]).fit(X_train, y_train)
        models["Poly2"] = {"model": poly_model, "params": {"degree": 2}}

        rf = RandomForestRegressor(n_estimators=150, max_depth=None, random_state=42).fit(X_train, y_train)
        models["RandomForest"] = {"model": rf, "params": {"n_estimators": 150, "max_depth": None, "random_state": 42}}

        gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42).fit(X_train, y_train)
        models["GradientBoosting"] = {"model": gb, "params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}}

    # Métricas
    metrics = {}
    for name, spec in models.items():
        mdl = spec["model"]
        y_pred = mdl.predict(X_test)
        metrics[name] = {
            "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
            "R2": float(r2_score(y_test, y_pred))
        }
    return models, metrics

st.sidebar.title("MVP – Leguminosas (MS Sales)")
st.sidebar.write("Visualiza **datos originales** y predice **protein (%)** moviendo widgets.")

tab1, tab2, tab3 = st.tabs(["🔎 Explorar datos", "🔮 Predicción", "📈 Comparar"])

# ---------------- Tab 1: Explorar ----------------
with tab1:
    c1, c2 = st.columns([2,1])
    with c2:
        species = st.multiselect("Especie", options=sorted(data["Species"].unique()), default=sorted(data["Species"].unique()))
        cfg_ids = st.multiselect("config_id (setup)", options=sorted(data["config_id"].unique()), default=[])

        # --- Filtros de sales (NO incluye la luz aquí) ---
        st.markdown("**Filtros de sales (mg/L)**")
        filt = {}
        for f in ["NH4NO3_mg_L","KNO3_mg_L","CaCl2_2H2O_mg_L","MgSO4_7H2O_mg_L","KH2PO4_mg_L"]:
            levels = DISCRETE_LEVELS[f]
            sel = st.multiselect(f, options=levels, default=[])
            if len(sel) > 0:
                filt[f] = sel

        # --- Filtros ambientales ---
        st.markdown("**Filtros ambientales**")
        lum_sel = st.multiselect("luminosity_umol_m2_s", options=DISCRETE_LEVELS["luminosity_umol_m2_s"], default=[])
        tmin, tmax = st.slider("temp_C", min_value=20.0, max_value=30.0, value=(20.0, 30.0), step=0.1)
        hmin, hmax = st.slider("humidity_%", min_value=40.0, max_value=80.0, value=(40.0, 80.0), step=1.0)

    df = data.copy()
    if species: df = df[df["Species"].isin(species)]
    if cfg_ids: df = df[df["config_id"].isin(cfg_ids)]
    for k, vals in filt.items():
        df = df[df[k].isin(vals)]
    if lum_sel: df = df[df["luminosity_umol_m2_s"].isin(lum_sel)]
    df = df[(df["temp_C"]>=tmin)&(df["temp_C"]<=tmax)&(df["humidity_%"]>=hmin)&(df["humidity_%"]<=hmax)]

    with c1:
        st.subheader("Tabla filtrada")
        st.dataframe(df.head(500), use_container_width=True, height=350)
        st.caption(f"Mostrando hasta 500 filas · Total filtradas: {len(df)}")

        st.subheader("Distribuciones")
        colA, colB, colC = st.columns(3)
        with colA:
            fig = px.histogram(df, x="protein", color="Species", nbins=30, barmode="overlay",
                               histnorm="probability density", title="Protein (%)")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.box(df, x="Species", y="stem", points="outliers", title="Stem (cm)")
            st.plotly_chart(fig, use_container_width=True)
        with colC:
            fig = px.box(df, x="Species", y="leafArea", points="outliers", title="LeafArea (cm²)")
            st.plotly_chart(fig, use_container_width=True)

# Entrena o carga modelos
models, metrics = train_models(data)

# ---------------- Tab 2: Predicción ----------------
with tab2:
    st.subheader("Predicción de protein (%)")
    left, right = st.columns([1,1])

    with left:
        model_choice = st.selectbox("Modelo", options=list(models.keys()),
                                    index=list(models.keys()).index("RandomForest") if "RandomForest" in models else 0)
        st.markdown("**Entradas**")
        inputs = {}
        # Sales + luz en selectboxes
        for k in ["NH4NO3_mg_L","KNO3_mg_L","CaCl2_2H2O_mg_L","MgSO4_7H2O_mg_L","KH2PO4_mg_L"]:
            levels = DISCRETE_LEVELS[k]
            inputs[k] = st.selectbox(k, options=levels, index=len(levels)//2)
        inputs["luminosity_umol_m2_s"] = st.selectbox("luminosity_umol_m2_s",
                                                      options=DISCRETE_LEVELS["luminosity_umol_m2_s"],
                                                      index=DISCRETE_LEVELS["luminosity_umol_m2_s"].index(650))

        # sliders continuos
        tmin, tmax = CONTINUOUS_RANGES["temp_C"]
        hmin, hmax = CONTINUOUS_RANGES["humidity_%"]
        inputs["temp_C"] = st.slider("temp_C", min_value=float(tmin), max_value=float(tmax), value=26.0, step=0.1)
        inputs["humidity_%"] = st.slider("humidity_%", min_value=float(hmin), max_value=float(hmax), value=60.0, step=1.0)

    with right:
        st.markdown("**Métricas (hold-out 20%)**")
        mdf = pd.DataFrame(metrics).T
        st.dataframe(mdf, use_container_width=True)

        xrow = pd.DataFrame([inputs])[FEATURES]
        pred = models[model_choice]["model"].predict(xrow)[0]
        st.metric("Predicción de proteína (%)", f"{pred:.2f}")

# ---------------- Tab 3: Comparar ----------------
with tab3:
    st.subheader("Comparar: barrer una variable")
    colA, colB = st.columns([1,1])

    with colA:
        model_choice2 = st.selectbox("Modelo", options=list(models.keys()),
                                     index=list(models.keys()).index("RandomForest") if "RandomForest" in models else 0,
                                     key="mdl2")
        var_to_sweep = st.selectbox("Variable a barrer", options=FEATURES, index=FEATURES.index("KNO3_mg_L"))
        base_vals = {
            "NH4NO3_mg_L": 1600, "KNO3_mg_L": 1900, "CaCl2_2H2O_mg_L": 440, "MgSO4_7H2O_mg_L": 370,
            "KH2PO4_mg_L": 170, "luminosity_umol_m2_s": 650, "temp_C": 26.0, "humidity_%": 60.0
        }
        st.write("Valores fijos (baseline):")
        st.json(base_vals)

        if var_to_sweep in DISCRETE_LEVELS:
            sweep_vals = DISCRETE_LEVELS[var_to_sweep]
        else:
            lo, hi = CONTINUOUS_RANGES[var_to_sweep]
            sweep_vals = np.linspace(lo, hi, 25).tolist()

    with colB:
        preds = []
        for v in sweep_vals:
            row = base_vals.copy(); row[var_to_sweep] = float(v)
            xrow = pd.DataFrame([row])[FEATURES]
            yhat = models[model_choice2]["model"].predict(xrow)[0]
            preds.append({"x": v, "protein_pred": yhat})
        pdf = pd.DataFrame(preds)
        fig = px.line(pdf, x="x", y="protein_pred", markers=True,
                      title=f"Respuesta de {var_to_sweep} – {model_choice2}")
        fig.update_layout(xaxis_title=var_to_sweep, yaxis_title="protein pred (%)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("MVP construido con Streamlit · CSV originales. Si no hay .pkl en /models, entrena al iniciar.")
