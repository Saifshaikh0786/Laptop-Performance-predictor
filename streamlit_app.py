# streamlit_app.py (FINAL FIXED VERSION)
# streamlit_app.py (FINAL FIXED VERSION)

# --------------------------------------------------------------------
# FIX for sklearn model loaded on newer versions (Streamlit Cloud)
# --------------------------------------------------------------------
import sklearn.compose._column_transformer as ct

# Dummy class to replace old sklearn internal class missing in new versions
class _RemainderColsList(list):
    pass

# Register the dummy class so joblib can unpickle the old model
ct._RemainderColsList = _RemainderColsList

# FIX for older HistGradientBoosting missing types
import sklearn.ensemble._hist_gradient_boosting.binning as binning
class OldBinMapper:
    pass
binning.BinMapper = getattr(binning, "BinMapper", OldBinMapper)

# Additional fallback for ColumnTransformer internals
import sklearn.utils._estimator_html_repr

import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle, os
from datetime import datetime

# --------------------------
# Page config & custom style
# --------------------------
st.set_page_config(page_title="Laptop Performance Pro", page_icon="💻", layout="wide")

DARK_CSS = """
<style>
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
:root { --card-bg: rgba(18, 22, 28, 0.7); --blur: 12px; --border: 1px solid rgba(255,255,255,0.08); }
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b0f14, #0c0f15);
  border-right: 1px solid rgba(255,255,255,0.06);
}
div.stButton>button, .stDownloadButton button {
  border-radius: 14px; padding: 0.6rem 1rem;
  border: 1px solid rgba(255,255,255,0.1);
  background: linear-gradient(180deg, #12161c, #0e1318);
  color: #e8eef7;
}
.card {
  background: var(--card-bg); backdrop-filter: blur(var(--blur));
  border: var(--border); border-radius: 18px;
  padding: 1rem 1.2rem;
}
.badge {
  display:inline-block; padding: 0.25rem 0.6rem;
  border-radius: 999px; font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.15);
}
.badge.green { background: rgba(0, 160, 80, .15); color: #9fffb5; }
.badge.yellow{ background: rgba(220, 160, 0, .15); color: #ffe69f; }
.badge.red   { background: rgba(220, 40, 40, .15); color: #ffb1b1; }
.score-ring { height: 16px; background: #0f1520; border-radius: 20px;
  border: 1px solid rgba(255,255,255,.06); overflow: hidden; }
.score-bar  { height: 100%; background: linear-gradient(90deg, #5cc8ff, #7d5cff); }
.small { color: #8da2b6; font-size: 0.9rem; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# --------------------------
# Model Loading
# --------------------------
@st.cache_data(ttl=1800)
def load_pipeline():
    for fname in ("laptop_performance_pipeline.joblib", "laptop_performance_pipeline.pkl"):
        if os.path.exists(fname):
            try:
                if fname.endswith(".joblib"):
                    return joblib.load(fname), fname
                else:
                    with open(fname, "rb") as f:
                        return pickle.load(f), fname
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    return None, None


# ✅ FIX: Avoid hashing the Pipeline object
@st.cache_data(ttl=1800)
def get_expected_columns_from_pre(_pipeline):
    pre = _pipeline.named_steps.get("pre")
    if pre is None:
        for k, v in _pipeline.named_steps.items():
            if hasattr(v, "transformers_"):
                pre = v
                break
    if pre is None:
        raise RuntimeError("ColumnTransformer not found")

    cols = []
    for _, _, colnames in pre.transformers_:
        if isinstance(colnames, (list, tuple, np.ndarray)):
            cols.extend(colnames)
    return cols


@st.cache_data(ttl=1800)
def load_defaults(expected_cols):
    defaults = {}
    if os.path.exists("synthetic_laptops_sample.csv"):
        df = pd.read_csv("synthetic_laptops_sample.csv")
        for c in expected_cols:
            if c in df.columns:
                defaults[c] = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else str(df[c].mode().iloc[0])

    smart = {
        "manufacturer":"Dell", "year":2023, "cpu_brand":"Intel", "cpu_tier":"mid",
        "cores":6, "base_clock":2.7, "boost_clock":4.0, "cpu_score":650,
        "gpu_type":"Integrated", "gpu_score":150, "ram_gb":16, "ram_speed_mhz":3200,
        "storage_type":"NVMe SSD", "storage_gb":512, "storage_speed_mb_s":2500,
        "display_in":15.6, "resolution":"1920x1080", "weight_kg":1.9,
        "battery_mah":6000, "tdp_watt":45, "price_usd":900
    }

    for c in expected_cols:
        defaults.setdefault(c, smart.get(c, 0))

    return defaults


def fill_missing(row_dict, expected_cols, defaults):
    data = {c: defaults.get(c) for c in expected_cols}
    data.update(row_dict)
    df = pd.DataFrame([data])
    for c in expected_cols:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass
    return df


# --------------------------
# Load model
# --------------------------
pipe, pipe_file = load_pipeline()
if pipe is None:
    st.error("❌ Model file not found.")
    st.stop()

expected_cols = get_expected_columns_from_pre(pipe)
defaults = load_defaults(expected_cols)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("⚙️ Controls")
preset = st.sidebar.selectbox("Preset", ["Custom", "Gaming (RTX)", "Ultrabook", "Budget Office"])
st.sidebar.caption(f"Model: `{pipe_file}`")

# --------------------------
# Tabs
# --------------------------
tab_pred, tab_batch, tab_about = st.tabs(["🔮 Predict", "📦 Batch", "ℹ️ About"])

# ================================
# 🔮 Prediction Tab
# ================================
with tab_pred:
    st.markdown("### 💻 Laptop Performance Pro — Premium Predictor")
    st.markdown('<div class="card">Provide as few or many specs as you want.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # ------------------
    # INPUTS — FIXED TYPES
    # ------------------
    with c1:
        manufacturer = st.selectbox("Manufacturer",
            ["Dell","HP","Lenovo","Asus","Acer","Apple","MSI","Razer"])

        year = st.number_input("Year", min_value=2016, max_value=2027,
                               value=int(defaults["year"]), step=1)

        cpu_brand = st.selectbox("CPU Brand", ["Intel","AMD","Apple"])
        cpu_tier = st.selectbox("CPU Tier", ["low","mid","high"])
        cores = st.selectbox("Cores", [2,4,6,8,12,16])

        base_clock = st.number_input("Base Clock (GHz)", min_value=0.5, max_value=6.0,
                                     value=float(defaults["base_clock"]), step=0.1)

        boost_clock = st.number_input("Boost Clock (GHz)", min_value=1.0, max_value=6.5,
                                      value=float(defaults["boost_clock"]), step=0.1)

        cpu_score = st.number_input("CPU Score", min_value=50, max_value=5000,
                                    value=int(defaults["cpu_score"]), step=10)

    with c2:
        gpu_type = st.selectbox("GPU Type", ["Integrated","NVIDIA GTX","NVIDIA RTX","AMD Radeon"])
        gpu_score = st.number_input("GPU Score", min_value=20, max_value=10000,
                                    value=int(defaults["gpu_score"]), step=10)

        ram_gb = st.selectbox("RAM (GB)", [4,8,12,16,24,32,64])
        ram_speed_mhz = st.selectbox("RAM Speed (MHz)", [2133,2400,2666,3000,3200,3600,4266,4800])

        storage_type = st.selectbox("Storage Type", ["HDD","SATA SSD","NVMe SSD"])
        storage_gb = st.selectbox("Storage (GB)", [128,256,512,1024,2048])

        storage_speed_mb_s = st.number_input("Storage Speed (MB/s)", min_value=80.0, max_value=8000.0,
                                             value=float(defaults["storage_speed_mb_s"]), step=50.0)

    with c3:
        display_in = st.number_input("Display (inches)", min_value=11.0, max_value=19.0,
                                     value=float(defaults["display_in"]), step=0.1)

        resolution = st.selectbox("Resolution", ["1366x768","1920x1080","2560x1440","3840x2160"])

        weight_kg = st.number_input("Weight (kg)", min_value=0.8, max_value=6.0,
                                    value=float(defaults["weight_kg"]), step=0.05)

        battery_mah = st.number_input("Battery (mAh)", min_value=3000, max_value=12000,
                                      value=int(defaults["battery_mah"]), step=100)

        tdp_watt = st.number_input("TDP (W)", min_value=5, max_value=150,
                                   value=int(defaults["tdp_watt"]), step=1)

        price_usd = st.number_input("Price (USD)",
                                    min_value=100.0, max_value=10000.0,
                                    value=float(defaults["price_usd"]), step=10.0)

    # --------------------------
    # Final row
    # --------------------------
    user_row = {
        "manufacturer": manufacturer, "year": int(year), "cpu_brand": cpu_brand, "cpu_tier": cpu_tier,
        "cores": int(cores), "base_clock": float(base_clock), "boost_clock": float(boost_clock),
        "cpu_score": int(cpu_score), "gpu_type": gpu_type, "gpu_score": int(gpu_score),
        "ram_gb": int(ram_gb), "ram_speed_mhz": int(ram_speed_mhz),
        "storage_type": storage_type, "storage_gb": int(storage_gb),
        "storage_speed_mb_s": float(storage_speed_mb_s), "display_in": float(display_in),
        "resolution": resolution, "weight_kg": float(weight_kg), "battery_mah": int(battery_mah),
        "tdp_watt": int(tdp_watt), "price_usd": float(price_usd)
    }

    if st.button("✨ Predict Performance"):
        df_in = fill_missing(user_row, expected_cols, defaults)
        score = float(pipe.predict(df_in)[0])

        score_norm = float(np.clip(score, 0, 1000))
        pct = int(score_norm / 10)

        if score_norm > 700:
            tag = ("High Performance", "badge green")
        elif score_norm > 350:
            tag = ("Mid Range", "badge yellow")
        else:
            tag = ("Entry Level", "badge red")

        years = round(min((score_norm / 100) + 3, 10), 1)

        # Output card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 🎯 Prediction Result")
        st.markdown(f"### ✅ Performance Score: `{score:.2f} / 1000`")
        st.markdown(f"### ✅ Normalized Score: `{score_norm:.1f} / 1000`")
        st.markdown(f"### ✅ Category: <span class='{tag[1]}'>{tag[0]}</span>", unsafe_allow_html=True)
        st.markdown(f"### ✅ Expected Smooth Years: `{years} years`")
        st.markdown("<div class='score-ring'><div class='score-bar' style='width: {}%;'></div></div>".format(pct),
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("See full input row sent to model"):
            st.dataframe(df_in.T)

# ================================
# 📦 Batch Tab
# ================================
with tab_batch:
    st.markdown("### 📦 Batch Predictions")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        dfu = pd.read_csv(file)
        output_rows = []

        for _, row in dfu.iterrows():
            row_dict = {k: row[k] for k in dfu.columns if k in expected_cols}
            df_filled = fill_missing(row_dict, expected_cols, defaults)
            pred = float(pipe.predict(df_filled)[0])
            item = df_filled.iloc[0].to_dict()
            item["predicted_performance"] = np.clip(pred, 0, 1000)
            output_rows.append(item)

        out_df = pd.DataFrame(output_rows)
        st.dataframe(out_df.head(50))

        st.download_button("Download CSV",
                           out_df.to_csv(index=False),
                           file_name="predictions.csv")

# ================================
# ℹ️ About Tab
# ================================
with tab_about:
    st.write("### About")
    st.info("Laptop Performance Pro • ML-powered synthetic prediction engine.")
