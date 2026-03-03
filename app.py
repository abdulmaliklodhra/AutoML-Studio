"""
app.py — AutoML PyCaret Application
=====================================
Upload your data → Select target → Let the AI choose Classification or Regression
→ Run PyCaret pipeline → View rich results and reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import sys
import time
import traceback

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio | PyCaret",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════
   LIGHT COLOR PALETTE
   Background  : #f8fafc → #ffffff
   Primary     : #2563eb  (blue-600)
   Secondary   : #7c3aed  (violet-600)
   Accent      : #059669  (emerald-600)
   Warning     : #d97706  (amber-600)
   Danger      : #dc2626  (red-600)
   Text-dark   : #0f172a
   Text-muted  : #64748b
══════════════════════════════════════════════════════════ */

/* ── Global ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

.stApp {
    background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 50%, #f5f3ff 100%);
    min-height: 100vh;
}

/* ── Hero Banner ────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(37,99,235,0.07) 0%,
        rgba(124,58,237,0.05) 50%,
        rgba(255,255,255,0.95) 100%);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(37,99,235,0.08), 0 1px 4px rgba(0,0,0,0.04);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px;
    right: -60px;
    width: 480px;
    height: 480px;
    background: radial-gradient(circle, rgba(37,99,235,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px;
    left: -40px;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(124,58,237,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(120deg, #2563eb 0%, #7c3aed 50%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 12px;
    font-weight: 400;
    max-width: 680px;
    line-height: 1.6;
}

/* ── Glass Cards ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(37,99,235,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
}
.glass-card:hover {
    border-color: rgba(37,99,235,0.25);
    background: rgba(255,255,255,0.98);
    box-shadow: 0 6px 28px rgba(37,99,235,0.1);
    transform: translateY(-1px);
}

/* ── Metric Cards ───────────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, rgba(37,99,235,0.06), rgba(124,58,237,0.04));
    border: 1px solid rgba(37,99,235,0.12);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    color: #2563eb;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Task Badge ─────────────────────────────────────────────────── */
.badge-classification {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(5,150,105,0.1), rgba(16,185,129,0.07));
    border: 1px solid rgba(5,150,105,0.4);
    color: #059669;
    font-weight: 600;
    padding: 8px 22px;
    border-radius: 50px;
    font-size: 0.95rem;
    animation: pulse-teal 2.5s infinite;
    box-shadow: 0 0 16px rgba(5,150,105,0.08);
}
.badge-regression {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(96,165,250,0.07));
    border: 1px solid rgba(37,99,235,0.4);
    color: #2563eb;
    font-weight: 600;
    padding: 8px 22px;
    border-radius: 50px;
    font-size: 0.95rem;
    animation: pulse-sky 2.5s infinite;
    box-shadow: 0 0 16px rgba(37,99,235,0.08);
}
@keyframes pulse-teal {
    0%, 100% { box-shadow: 0 0 0 0 rgba(5,150,105,0); }
    50%       { box-shadow: 0 0 0 10px rgba(5,150,105,0.1); }
}
@keyframes pulse-sky {
    0%, 100% { box-shadow: 0 0 0 0 rgba(37,99,235,0); }
    50%       { box-shadow: 0 0 0 10px rgba(37,99,235,0.1); }
}

/* ── Step Indicator ─────────────────────────────────────────────── */
.step-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 18px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    transition: all 0.3s;
}
.step-pending  { background: #f1f5f9; color: #94a3b8; }
.step-active   { background: rgba(37,99,235,0.07); color: #2563eb; border: 1px solid rgba(37,99,235,0.2); }
.step-done     { background: rgba(5,150,105,0.07);  color: #059669; border: 1px solid rgba(5,150,105,0.2); }

/* ── Section Titles ─────────────────────────────────────────────── */
.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(37,99,235,0.1);
    letter-spacing: -0.01em;
}

/* ── Sidebar styling ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8faff 0%, #ffffff 60%, #f5f3ff 100%);
    border-right: 1px solid rgba(37,99,235,0.1);
}
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
    padding: 14px 0;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 20px rgba(37,99,235,0.25);
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(37,99,235,0.35);
}

/* ── Main area buttons ───────────────────────────────────────────── */
.stDownloadButton>button {
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(124,58,237,0.06)) !important;
    color: #2563eb !important;
    border: 1px solid rgba(37,99,235,0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}
.stDownloadButton>button:hover {
    background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(124,58,237,0.1)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.18) !important;
}

/* ── Alert-like info boxes ───────────────────────────────────────── */
.info-box {
    background: rgba(37,99,235,0.05);
    border-left: 3px solid #2563eb;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #1d4ed8;
    font-size: 0.9rem;
    margin: 12px 0;
}
.warning-box {
    background: rgba(217,119,6,0.06);
    border-left: 3px solid #d97706;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #b45309;
    font-size: 0.9rem;
    margin: 12px 0;
}
.success-box {
    background: rgba(5,150,105,0.06);
    border-left: 3px solid #059669;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #047857;
    font-size: 0.9rem;
    margin: 12px 0;
}

/* ── Streamlit native widgets — tint overrides ───────────────────── */
[data-testid="stMetricValue"] {
    color: #2563eb !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stTab"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}

/* ── DataFrame styling ───────────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(37,99,235,0.1) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(248,250,255,0.8) !important;
    border-radius: 10px !important;
    color: #334155 !important;
    font-weight: 500 !important;
}

/* ── Progress bar ────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: rgba(37,99,235,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(37,99,235,0.4); }

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# ── Session State Initialization ───────────────────────────────────────────────
for key in ["df", "task_type", "results", "best_model", "final_model",
            "compare_df", "setup_df", "ran_pipeline", "target_col", "selected_features"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "ran_pipeline" not in st.session_state:
    st.session_state.ran_pipeline = False


# ── Helpers ────────────────────────────────────────────────────────────────────
def detect_task_type(df: pd.DataFrame, target: str) -> str:
    col = df[target].dropna()
    n_unique = col.nunique()
    dtype = col.dtype
    if dtype == "object" or dtype.name == "category" or dtype == bool:
        return "classification"
    if pd.api.types.is_integer_dtype(dtype) and n_unique <= 20:
        return "classification"
    if pd.api.types.is_float_dtype(dtype) and n_unique <= 10:
        return "classification"
    return "regression"


def format_number(n):
    if isinstance(n, float):
        return f"{n:.4f}"
    return str(n)


def get_eda_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include=np.number).shape[1]),
        "cat_cols": int(df.select_dtypes(include="object").shape[1]),
    }


def style_compare_df(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Return a display-ready version of the comparison DataFrame."""
    if task_type == "classification":
        key_cols = [c for c in ['Accuracy', 'AUC', 'F1', 'Kappa'] if c in df.columns]
    else:
        key_cols = [c for c in ['R2', 'RMSE', 'MAE'] if c in df.columns]
    return df


def build_metric_bar_chart(compare_df: pd.DataFrame, task_type: str) -> go.Figure:
    """Build a Plotly bar chart comparing model metrics."""
    df = compare_df.copy()

    # Get model names
    if 'Model' in df.columns:
        model_col = 'Model'
    elif df.index.name:
        df = df.reset_index()
        model_col = df.columns[0]
    else:
        df = df.reset_index()
        model_col = df.columns[0]

    # Pick the primary metric
    if task_type == "classification":
        metric = 'Accuracy' if 'Accuracy' in df.columns else df.columns[1]
        color_scale = 'Viridis'
    else:
        metric = 'R2' if 'R2' in df.columns else df.columns[1]
        color_scale = 'Plasma'

    df = df[[model_col, metric]].dropna()
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.dropna().sort_values(metric, ascending=(task_type == "regression"))

    fig = px.bar(
        df,
        x=metric,
        y=model_col,
        orientation='h',
        color=metric,
        color_continuous_scale=color_scale,
        labels={metric: metric, model_col: "Model"},
        title=f"Model Comparison — {metric}",
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155', family='Inter'),
        title_font=dict(size=16, color='#2563eb'),
        xaxis=dict(gridcolor='rgba(37,99,235,0.08)', color='#64748b'),
        yaxis=dict(gridcolor='rgba(37,99,235,0.08)', color='#64748b'),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return fig


def build_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9, "color": "white"},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(colors='#9ca3af')
    ax.set_title("Correlation Heatmap", color='#a78bfa', fontsize=13, pad=12)
    plt.tight_layout()
    return fig


def build_feature_importance_chart(model, feature_names: list) -> go.Figure:
    try:
        estimator = model
        if hasattr(model, 'steps'):
            estimator = model.steps[-1][1]
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            importances = np.abs(coef.mean(axis=0)) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        min_len = min(len(importances), len(feature_names))
        fi_df = pd.DataFrame({
            'Feature': feature_names[:min_len],
            'Importance': importances[:min_len]
        }).sort_values('Importance', ascending=True).tail(15)

        fig = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis',
            title="Feature Importance (Top 15)"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155', family='Inter'),
            title_font=dict(size=16, color='#2563eb'),
            xaxis=dict(gridcolor='rgba(37,99,235,0.08)', color='#64748b'),
            yaxis=dict(gridcolor='rgba(37,99,235,0.08)', color='#64748b'),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=50, b=10),
            height=400,
        )
        return fig
    except Exception:
        return None


def build_target_distribution(df: pd.DataFrame, target: str, task_type: str) -> go.Figure:
    if task_type == "classification":
        counts = df[target].value_counts().reset_index()
        counts.columns = [target, 'Count']
        fig = px.pie(
            counts, names=target, values='Count',
            title=f"Target Distribution: {target}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
    else:
        fig = px.histogram(
            df, x=target, nbins=30,
            title=f"Target Distribution: {target}",
            color_discrete_sequence=['#0ea5e9'],
        )
        fig.update_traces(marker_line_color='rgba(255,255,255,0.1)', marker_line_width=0.8)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155', family='Inter'),
        title_font=dict(size=15, color='#2563eb'),
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 14px;'>
        <!-- PyCaret SVG Monogram -->
        <div style='display:flex; justify-content:center; margin-bottom:10px;'>
            <svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="pyGrad" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#2563eb"/>
                        <stop offset="100%" stop-color="#7c3aed"/>
                    </linearGradient>
                </defs>
                <!-- Diamond shape -->
                <path d="M36 4 L68 36 L36 68 L4 36 Z" fill="url(#pyGrad)" rx="4"/>
                <!-- Inner white diamond outline -->
                <path d="M36 12 L60 36 L36 60 L12 36 Z" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="1.5"/>
                <!-- Py text -->
                <text x="36" y="41" font-family="'Plus Jakarta Sans', 'Inter', sans-serif"
                      font-size="20" font-weight="800" fill="white"
                      text-anchor="middle" dominant-baseline="middle"
                      letter-spacing="-1">Py</text>
            </svg>
        </div>
        <div style='font-family:"Plus Jakarta Sans",sans-serif; font-size:1.3rem;
                    font-weight:800; color:#2563eb; letter-spacing:-0.02em;'>AutoML Studio</div>
        <div style='color:#7c3aed; font-size:0.8rem; margin-top:4px; font-weight:600;
                    letter-spacing:0.04em; text-transform:uppercase;'>
            Powered by PyCaret
        </div>
    </div>
    <hr style='border-color:rgba(37,99,235,0.12); margin: 14px 0;'>
    """, unsafe_allow_html=True)

    # ── Sample Datasets ───────────────────────────────────────────────────────
    SAMPLE_DATASETS = {
        "── Classification ──": None,
        "🌸 Iris (flowers, 150 rows)":              ("iris",           "classification"),
        "🚢 Titanic (survival, 891 rows)":          ("titanic",        "classification"),
        "🩺 Diabetes (medical, 768 rows)":          ("diabetes",       "classification"),
        "❤️ Heart Disease (303 rows)":              ("heart_disease",  "classification"),
        "🌙 Moon (synthetic, 1000 rows)":           ("moon",           "classification"),
        "🏦 Bank Marketing (4521 rows)":            ("bank",           "classification"),
        "── Regression ──": None,
        "🏠 Boston Housing (506 rows)":             ("boston",         "regression"),
        "💎 Diamonds (price, 53940 rows)":          ("diamond",        "regression"),
        "🚗 Auto MPG (398 rows)":                   ("automobile",     "regression"),
        "⚡ Energy Efficiency (768 rows)":          ("energy",         "regression"),
        "🍷 Wine Quality (6497 rows)":              ("wine",           "regression"),
    }

    st.markdown("### 📦 Sample Dataset")
    ds_choice = st.selectbox(
        "Choose a built-in dataset",
        options=list(SAMPLE_DATASETS.keys()),
        index=0,
        label_visibility="collapsed",
        key="sample_ds_choice",
    )
    load_sample_btn = st.button(
        "⬇️ Load Sample Dataset",
        use_container_width=True,
        disabled=(SAMPLE_DATASETS.get(ds_choice) is None),
        key="load_sample_btn",
    )
    if load_sample_btn and SAMPLE_DATASETS.get(ds_choice) is not None:
        ds_name, _ = SAMPLE_DATASETS[ds_choice]
        try:
            from pycaret.datasets import get_data
            sample_df = get_data(ds_name, verbose=False)
            st.session_state.df = sample_df
            st.session_state.ran_pipeline = False
            st.session_state.compare_df = None
            st.session_state.selected_features = None
            st.session_state._prev_target_col = None
            st.session_state._prev_feature_cols = None
            st.success(f"✅ Loaded **{ds_choice.strip()}** ({len(sample_df):,} rows × {sample_df.shape[1]} cols)")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

    st.markdown("---")
    st.markdown("### 📂 Upload Your Own Dataset")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.ran_pipeline = False
            st.session_state.compare_df = None
            st.session_state.selected_features = None
            st.session_state._prev_target_col = None
            st.session_state._prev_feature_cols = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### 🎯 Select Target Column")
        columns = df.columns.tolist()
        target_col = st.selectbox(
            "Target column",
            columns,
            index=len(columns) - 1,
            label_visibility="collapsed",
        )
        st.session_state.target_col = target_col

        # Auto-detect task type
        task_type = detect_task_type(df, target_col)
        st.session_state.task_type = task_type

        if task_type == "classification":
            st.markdown("""<div class="badge-classification">🧠 Classification Detected</div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div class="badge-regression">📈 Regression Detected</div>""",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧩 Select Feature Variables")
        feature_cols = [c for c in columns if c != target_col]

        # Detect target or file change → reset widget key to all features
        _prev_target = st.session_state.get("_prev_target_col")
        _prev_features = st.session_state.get("_prev_feature_cols")
        if _prev_target != target_col or _prev_features != feature_cols:
            # Reset: set widget key directly (Streamlit reads this before rendering)
            st.session_state["feature_multiselect"] = feature_cols
            st.session_state._prev_target_col = target_col
            st.session_state._prev_feature_cols = feature_cols
        elif "feature_multiselect" not in st.session_state:
            # First load: initialize widget key
            st.session_state["feature_multiselect"] = feature_cols
        else:
            # Normal rerun: sanitize persisted value (remove invalid columns)
            _kept = [c for c in st.session_state["feature_multiselect"] if c in feature_cols]
            st.session_state["feature_multiselect"] = _kept if _kept else feature_cols

        selected_features = st.multiselect(
            "Choose features to include in the model",
            options=feature_cols,
            label_visibility="collapsed",
            placeholder="Select feature columns…",
            key="feature_multiselect",
        )
        # Keep a copy in selected_features for pipeline use
        st.session_state.selected_features = selected_features

        if len(selected_features) == 0:
            st.markdown("""<div class="warning-box">⚠️ Please select at least one feature column.</div>""",
                        unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### ⚙️ Pipeline Settings")
        fold_n = st.slider("Cross-validation Folds", 3, 10, 5, 1)
        session_id = st.number_input("Session ID (seed)", value=42, step=1)

        # Task type override
        _auto_task = detect_task_type(df, target_col)
        st.session_state.task_type = _auto_task
        task_override = st.radio(
            "Task Type",
            options=["🤖 Auto-detect", "🧠 Classification", "📈 Regression"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if task_override == "🧠 Classification":
            st.session_state.task_type = "classification"
        elif task_override == "📈 Regression":
            st.session_state.task_type = "regression"
        task_type = st.session_state.task_type

        with st.expander("🔬 Advanced Options", expanded=False):
            train_size = st.slider("Train Size", 0.60, 0.90, 0.80, 0.05,
                                   help="Fraction of data used for training")
            normalize = st.toggle("Normalize Features", value=False)
            if normalize:
                normalize_method = st.selectbox(
                    "Normalization Method",
                    ["zscore", "minmax", "maxabs", "robust"],
                    index=0,
                )
            else:
                normalize_method = "zscore"
            numeric_imputation = st.selectbox(
                "Numeric Imputation",
                ["mean", "median", "mode", "knn", "zero"],
                index=0,
            )
            remove_outliers = st.toggle("Remove Outliers", value=False)
            feature_selection = st.toggle("Auto Feature Selection", value=False)
            if task_type == "classification":
                sort_metric = st.selectbox(
                    "Optimization Metric",
                    ["Accuracy", "AUC", "F1", "Recall", "Precision", "Kappa", "MCC"],
                    index=0,
                )
            else:
                sort_metric = st.selectbox(
                    "Optimization Metric",
                    ["R2", "RMSE", "MAE", "MAPE", "RMSLE"],
                    index=0,
                )
            n_models = st.slider("Max Models to Compare", 3, 20, 10, 1)

        # Store advanced settings in session state
        st.session_state.adv = dict(
            train_size=train_size, normalize=normalize,
            normalize_method=normalize_method,
            numeric_imputation=numeric_imputation,
            remove_outliers=remove_outliers,
            feature_selection=feature_selection,
            sort_metric=sort_metric, n_models=n_models,
        )

        st.markdown("---")
        _no_features = len(st.session_state.get("selected_features") or []) == 0
        run_btn = st.button(
            "🚀 Run ML Pipeline",
            use_container_width=True,
            disabled=_no_features,
        )
    else:
        run_btn = False
        fold_n = 5
        session_id = 42

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(150,160,190,0.5); font-size:0.75rem; text-align:center; padding-bottom:10px;'>
        AutoML Studio v1.0<br>
        Built with PyCaret + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🤖 AutoML Studio</div>
    <div class="hero-subtitle">
        Upload your dataset → Select a target variable → Let AI automatically
        detect Classification or Regression → Run the full PyCaret pipeline in one click.
    </div>
</div>
""", unsafe_allow_html=True)

# ── No file uploaded state ─────────────────────────────────────────────────────
if st.session_state.df is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:32px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📤</div>
            <div style="color:#38bdf8; font-weight:600; font-size:1rem; margin-bottom:8px;">Upload Data</div>
            <div style="color:#64748b; font-size:0.85rem;">CSV or Excel files supported up to any size</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:32px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">🧠</div>
            <div style="color:#818cf8; font-weight:600; font-size:1rem; margin-bottom:8px;">Auto Detection</div>
            <div style="color:#64748b; font-size:0.85rem;">AI decides Classification vs Regression automatically</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:32px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📊</div>
            <div style="color:#34d399; font-weight:600; font-size:1rem; margin-bottom:8px;">Rich Reports</div>
            <div style="color:#64748b; font-size:0.85rem;">Compare all models, charts, feature importance & downloads</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        👈 <strong>Get started:</strong> Upload a CSV or Excel file using the sidebar uploader to begin.
    </div>
    """, unsafe_allow_html=True)

    # Sample pipeline flow diagram
    st.markdown("<div class='section-title'>📋 How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div style="text-align:center; flex:1; min-width:100px;">
                <div style="font-size:1.8rem;">📂</div>
                <div style="color:#38bdf8; font-size:0.85rem; font-weight:600; margin-top:6px;">1. Upload</div>
                <div style="color:#475569; font-size:0.78rem;">CSV / Excel</div>
            </div>
            <div style="color:#1e293b; font-size:1.5rem; flex:0;">→</div>
            <div style="text-align:center; flex:1; min-width:100px;">
                <div style="font-size:1.8rem;">🎯</div>
                <div style="color:#818cf8; font-size:0.85rem; font-weight:600; margin-top:6px;">2. Select Target</div>
                <div style="color:#475569; font-size:0.78rem;">Choose column</div>
            </div>
            <div style="color:#1e293b; font-size:1.5rem; flex:0;">→</div>
            <div style="text-align:center; flex:1; min-width:100px;">
                <div style="font-size:1.8rem;">🧠</div>
                <div style="color:#34d399; font-size:0.85rem; font-weight:600; margin-top:6px;">3. Auto Detect</div>
                <div style="color:#475569; font-size:0.78rem;">Classification / Regression</div>
            </div>
            <div style="color:#1e293b; font-size:1.5rem; flex:0;">→</div>
            <div style="text-align:center; flex:1; min-width:100px;">
                <div style="font-size:1.8rem;">⚙️</div>
                <div style="color:#fbbf24; font-size:0.85rem; font-weight:600; margin-top:6px;">4. PyCaret Pipeline</div>
                <div style="color:#475569; font-size:0.78rem;">Setup → Compare → Finalize</div>
            </div>
            <div style="color:#1e293b; font-size:1.5rem; flex:0;">→</div>
            <div style="text-align:center; flex:1; min-width:100px;">
                <div style="font-size:1.8rem;">📊</div>
                <div style="color:#f472b6; font-size:0.85rem; font-weight:600; margin-top:6px;">5. Results</div>
                <div style="color:#475569; font-size:0.78rem;">Metrics, Charts, Download</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Data Loaded ──────────────────────────────────────────────────────────
    df = st.session_state.df
    target_col = st.session_state.target_col
    task_type = st.session_state.task_type

    # ── EDA Summary ─────────────────────────────────────────────────────────
    eda = get_eda_summary(df)
    st.markdown("<div class='section-title'>🔬 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "📋 Overview", "📈 Distribution", "🔗 Correlations", "🔍 Scatter Explorer"
    ])

    with eda_tab1:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: st.metric("Rows", f"{eda['rows']:,}")
        with m2: st.metric("Columns", eda['cols'])
        with m3: st.metric("Missing", eda['missing'])
        with m4: st.metric("Duplicates", eda['duplicates'])
        with m5: st.metric("Numeric", eda['numeric_cols'])
        with m6: st.metric("Categorical", eda['cat_cols'])

        # Data types table
        _dtype_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.count().values,
            "Null": df.isnull().sum().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(1).values,
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.markdown("#### 📐 Column Profile")
        st.dataframe(_dtype_df, use_container_width=True, height=300)

        # Missing values bar chart
        _miss = df.isnull().sum()
        _miss = _miss[_miss > 0]
        if len(_miss) > 0:
            _mdf = _miss.reset_index()
            _mdf.columns = ["Column", "Missing Count"]
            _mdf["Missing %"] = (_mdf["Missing Count"] / len(df) * 100).round(2)
            _fig_miss = px.bar(_mdf, x="Column", y="Missing %",
                               color="Missing %", color_continuous_scale="RdYlGn_r",
                               title="Missing Values per Column (%)")
            _fig_miss.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", family="Inter"),
                title_font=dict(size=14, color="#d97706"), height=300,
            )
            st.plotly_chart(_fig_miss, use_container_width=True)
        else:
            st.markdown('<div class="success-box">✅ No missing values found in this dataset!</div>',
                        unsafe_allow_html=True)

        with st.expander("🔍 Data Preview (first 100 rows)", expanded=False):
            st.dataframe(df.head(100), use_container_width=True, height=260)

        # Descriptive stats
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.dataframe(df.describe(include="all"), use_container_width=True)

    with eda_tab2:
        _num_cols = df.select_dtypes(include=np.number).columns.tolist()
        _all_cols = df.columns.tolist()
        _dist_col = st.selectbox("Select Column", _all_cols, key="dist_col")
        _chart_type = st.radio("Chart Type", ["Histogram", "Box Plot", "Violin Plot"],
                               horizontal=True, key="dist_chart_type")

        _left_d, _right_d = st.columns(2)
        with _left_d:
            # Distribution chart
            if _chart_type == "Histogram":
                _fig_d = px.histogram(df, x=_dist_col, nbins=40,
                                      color_discrete_sequence=["#2563eb"],
                                      title=f"Histogram: {_dist_col}")
                _fig_d.update_traces(marker_line_color="white", marker_line_width=0.5)
            elif _chart_type == "Box Plot":
                _fig_d = px.box(df, y=_dist_col, color_discrete_sequence=["#7c3aed"],
                                title=f"Box Plot: {_dist_col}", points="outliers")
            else:
                _fig_d = px.violin(df, y=_dist_col, color_discrete_sequence=["#059669"],
                                   title=f"Violin: {_dist_col}", box=True)
            _fig_d.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#334155"), title_font=dict(color="#2563eb", size=15),
                                  height=380)
            st.plotly_chart(_fig_d, use_container_width=True)

        with _right_d:
            # Target distribution
            _tgt_fig = build_target_distribution(df, target_col, task_type)
            st.plotly_chart(_tgt_fig, use_container_width=True)

        # Column stats
        if _dist_col in _num_cols:
            _s = df[_dist_col].describe()
            _skew = df[_dist_col].skew()
            _kurt = df[_dist_col].kurtosis()
            _scols = st.columns(6)
            for _sc, (_lbl, _val) in zip(_scols, [
                ("Mean", f"{_s['mean']:.3f}"), ("Std", f"{_s['std']:.3f}"),
                ("Min", f"{_s['min']:.3f}"), ("Max", f"{_s['max']:.3f}"),
                ("Skewness", f"{_skew:.3f}"), ("Kurtosis", f"{_kurt:.3f}"),
            ]):
                _sc.metric(_lbl, _val)

    with eda_tab3:
        _corr_left, _corr_right = st.columns([3, 2])
        with _corr_left:
            _corr_fig = build_correlation_heatmap(df)
            if _corr_fig:
                st.pyplot(_corr_fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation.")
        with _corr_right:
            st.markdown("#### 🔝 Top Correlated Pairs")
            _num_df = df.select_dtypes(include=np.number)
            if _num_df.shape[1] >= 2:
                _corr_matrix = _num_df.corr().abs()
                _pairs = (
                    _corr_matrix.where(np.tril(np.ones(_corr_matrix.shape), k=-1).astype(bool))
                    .stack().reset_index()
                )
                _pairs.columns = ["Feature A", "Feature B", "Correlation"]
                _pairs = _pairs.sort_values("Correlation", ascending=False).head(15)
                _pairs["Correlation"] = _pairs["Correlation"].round(4)
                st.dataframe(_pairs, use_container_width=True, height=400)

    with eda_tab4:
        _sc_cols = df.columns.tolist()
        _sc1, _sc2, _sc3 = st.columns(3)
        _x_col = _sc1.selectbox("X Axis", _sc_cols, index=0, key="sc_x")
        _y_col = _sc2.selectbox("Y Axis", _sc_cols,
                                index=min(1, len(_sc_cols)-1), key="sc_y")
        _color_opts = ["None"] + _sc_cols
        _hue_col = _sc3.selectbox("Color By", _color_opts, index=0, key="sc_hue")
        _scatter_fig = px.scatter(
            df, x=_x_col, y=_y_col,
            color=_hue_col if _hue_col != "None" else None,
            opacity=0.7,
            title=f"{_x_col} vs {_y_col}",
            color_continuous_scale="Viridis",
        )
        _scatter_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter"),
            title_font=dict(size=15, color="#2563eb"), height=450,
        )
        _scatter_fig.update_traces(marker=dict(size=6))
        st.plotly_chart(_scatter_fig, use_container_width=True)

    # ── Task type display ─────────────────────────────────────────────────────
    st.markdown("---")
    _override_label = {
        "🤖 Auto-detect": "Auto-detected",
        "🧠 Classification": "Manually set",
        "📈 Regression": "Manually set",
    }.get(st.session_state.get("task_override_radio", "🤖 Auto-detect"), "Auto-detected")

    tc1, tc2 = st.columns([1, 2])
    with tc1:
        if task_type == "classification":
            n_classes = df[target_col].nunique()
            st.markdown(f"""
            <div class="badge-classification" style="font-size:1.1rem; padding:12px 24px;">
                🧠 Classification
            </div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:12px;">
                Target: <strong style="color:#2563eb">{target_col}</strong>
                · {n_classes} unique classes<br>
                Using <strong>PyCaret Classification</strong> pipeline
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="badge-regression" style="font-size:1.1rem; padding:12px 24px;">
                📈 Regression
            </div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:12px;">
                Target: <strong style="color:#2563eb">{target_col}</strong>
                · Continuous numeric values<br>
                Using <strong>PyCaret Regression</strong> pipeline
            </div>
            """, unsafe_allow_html=True)
    with tc2:
        st.markdown("""
        <div class="info-box">
            <strong>How detection works:</strong><br>
            • Object/category/boolean dtype → Classification<br>
            • Integer with ≤ 20 unique values → Classification<br>
            • Continuous float or large-cardinality integer → Regression<br>
            Use the <strong>Task Type</strong> radio in the sidebar to override.
        </div>
        """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  RUN PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    if run_btn:
        st.markdown("---")
        st.markdown("<div class='section-title'>⚙️ Running PyCaret Pipeline</div>",
                    unsafe_allow_html=True)

        steps = [
            "🔧 Setting up PyCaret environment...",
            "🔄 Comparing all models (cross-validation)...",
            "🏆 Selecting best model...",
            "✨ Finalizing model on full dataset...",
        ]

        progress_bar = st.progress(0)
        status_box = st.empty()

        try:
            for i, step in enumerate(steps):
                status_box.markdown(f"""
                <div class="step-item step-active">
                    <span style="font-size:1.2rem;">⟳</span> {step}
                </div>""", unsafe_allow_html=True)
                progress_bar.progress((i) / len(steps))

                if i == 0:
                    # ── Auto-clean: drop rows where target is NaN ──────────
                    n_missing_target = int(df[target_col].isnull().sum())
                    if n_missing_target > 0:
                        df = df.dropna(subset=[target_col]).reset_index(drop=True)
                        st.session_state.df = df   # update session with cleaned df
                        st.markdown(f"""
                        <div class="warning-box">
                            ⚠️ <strong>Auto-cleaned:</strong> Removed <strong>{n_missing_target} rows</strong>
                            with missing values in target column <code>{target_col}</code>.
                            Proceeding with <strong>{len(df):,} rows</strong>.
                        </div>""", unsafe_allow_html=True)

                    # Actually run the pipeline
                    _all_features = [c for c in df.columns if c != target_col]
                    _selected = st.session_state.get("selected_features") or _all_features
                    _ignore = [c for c in _all_features if c not in _selected]
                    _adv = st.session_state.get("adv") or {}

                    _setup_kwargs = dict(
                        data=df,
                        target=target_col,
                        session_id=int(session_id),
                        fold=fold_n,
                        train_size=_adv.get("train_size", 0.8),
                        normalize=_adv.get("normalize", False),
                        normalize_method=_adv.get("normalize_method", "zscore"),
                        numeric_imputation=_adv.get("numeric_imputation", "mean"),
                        remove_outliers=_adv.get("remove_outliers", False),
                        feature_selection=_adv.get("feature_selection", False),
                        ignore_features=_ignore if _ignore else None,
                        verbose=False,
                        html=False,
                    )

                    if task_type == "classification":
                        from pycaret.classification import (
                            setup, compare_models, predict_model,
                            finalize_model, pull, save_model
                        )
                    else:
                        from pycaret.regression import (
                            setup, compare_models, predict_model,
                            finalize_model, pull, save_model
                        )

                    env = setup(**_setup_kwargs)
                    setup_df = pull()
                    st.session_state.setup_df = setup_df

                elif i == 1:
                    _adv = st.session_state.get("adv") or {}
                    best_model = compare_models(
                        sort=_adv.get("sort_metric", "Accuracy" if task_type == "classification" else "R2"),
                        n_select=1,
                        verbose=False,
                    )
                    compare_df = pull()
                    st.session_state.best_model = best_model
                    st.session_state.compare_df = compare_df
                    # Store holdout predictions for CM / residuals
                    try:
                        holdout_preds = predict_model(best_model, verbose=False)
                        st.session_state.holdout_preds = holdout_preds
                    except Exception:
                        st.session_state.holdout_preds = None

                elif i == 3:
                    final_model = finalize_model(best_model)
                    st.session_state.final_model = final_model

                    # Save model
                    save_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        f"best_{task_type}_model"
                    )
                    save_model(final_model, save_path)
                    st.session_state.model_save_path = save_path + ".pkl"

                time.sleep(0.3)
                progress_bar.progress((i + 1) / len(steps))

            status_box.markdown("""
            <div class="success-box">
                ✅ <strong>Pipeline completed successfully!</strong> Scroll down to view results.
            </div>""", unsafe_allow_html=True)
            progress_bar.progress(1.0)
            st.session_state.ran_pipeline = True

        except Exception as e:
            error_msg = traceback.format_exc()
            status_box.markdown(f"""
            <div class="warning-box">
                ❌ <strong>Pipeline Error:</strong> {str(e)}
            </div>""", unsafe_allow_html=True)
            with st.expander("🔍 Error Details"):
                st.code(error_msg)
            st.session_state.ran_pipeline = False

    # ══════════════════════════════════════════════════════════════════════════
    #  RESULTS SECTION
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.ran_pipeline and st.session_state.compare_df is not None:
        st.markdown("---")
        st.markdown("""
        <div class="hero-banner" style="padding:28px 40px; margin-bottom:24px;">
            <div class="hero-title" style="font-size:2rem;">🏆 ML Pipeline Results</div>
            <div class="hero-subtitle">All models compared · Best model selected · Ready for deployment</div>
        </div>
        """, unsafe_allow_html=True)

        compare_df = st.session_state.compare_df
        best_model = st.session_state.best_model
        final_model = st.session_state.final_model
        setup_df = st.session_state.setup_df

        # ── Best Model Card ────────────────────────────────────────────────────
        best_name = type(best_model).__name__
        if hasattr(best_model, 'steps'):
            best_name = type(best_model.steps[-1][1]).__name__

        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(167,139,250,0.4);
             background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08));">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size:2.5rem;">🥇</div>
                <div>
                    <div style="color:#475569; font-size:0.8rem; text-transform:uppercase;
                                letter-spacing:0.1em; font-weight:600;">Best Model</div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.6rem;
                                font-weight:800; color:#38bdf8; margin-top:2px;">{best_name}</div>
                    <div style="color:#64748b; font-size:0.85rem; margin-top:4px;">
                        Task: <span style="color:#818cf8; font-weight:600;">
                        {'Classification' if task_type == 'classification' else 'Regression'}</span>
                        · Finalized on full dataset
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top Metrics Highlight ──────────────────────────────────────────────
        if len(compare_df) > 0:
            top_row = compare_df.iloc[0]
            if task_type == "classification":
                metrics_to_show = [c for c in ['Accuracy', 'AUC', 'F1', 'Kappa', 'MCC'] if c in compare_df.columns]
            else:
                metrics_to_show = [c for c in ['R2', 'MAE', 'RMSE', 'MAPE'] if c in compare_df.columns]

            if metrics_to_show:
                mcols = st.columns(len(metrics_to_show))
                for col, metric in zip(mcols, metrics_to_show):
                    val = top_row.get(metric, "N/A")
                    try:
                        val_fmt = f"{float(val):.4f}"
                    except Exception:
                        val_fmt = str(val)
                    with col:
                        st.metric(f"Best {metric}", val_fmt)

        # ── Tabs for Results ───────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Leaderboard",
            "📈 Metric Charts",
            "🕸️ Radar Chart",
            "🌟 Feature Importance",
            "🔬 Model Diagnostics",
            "🔮 Predict",
            "⚙️ Setup Config",
        ])

        with tab1:
            st.markdown("#### All Models Ranked by Performance")
            st.markdown("""
            <div class="info-box">
                🏅 The <strong>top row</strong> is the best-performing model selected by PyCaret.
            </div>""", unsafe_allow_html=True)
            st.dataframe(compare_df, use_container_width=True, height=450)
            csv_data = compare_df.to_csv(index=True).encode("utf-8")
            st.download_button("⬇️ Download Leaderboard CSV", csv_data,
                               file_name=f"model_comparison_{task_type}.csv", mime="text/csv")

        with tab2:
            st.markdown("#### Model Performance Comparison")
            bar_fig = build_metric_bar_chart(compare_df, task_type)
            st.plotly_chart(bar_fig, use_container_width=True)

            if task_type == "classification":
                alt_metrics = [c for c in ['F1', 'Kappa', 'Recall', 'Prec.', 'MCC'] if c in compare_df.columns]
            else:
                alt_metrics = [c for c in ['MAE', 'RMSE', 'MAPE', 'R2'] if c in compare_df.columns]

            if alt_metrics and 'Model' in compare_df.columns:
                _sel_metric = st.selectbox("Secondary Metric", alt_metrics, key="sec_metric")
                _df_p = compare_df[['Model', _sel_metric]].copy()
                _df_p[_sel_metric] = pd.to_numeric(_df_p[_sel_metric], errors='coerce')
                _df_p = _df_p.dropna().sort_values(_sel_metric)
                _fig2 = px.bar(_df_p, x='Model', y=_sel_metric, color='Model',
                               title=f"All Models — {_sel_metric}",
                               color_discrete_sequence=px.colors.qualitative.Set2)
                _fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#334155', family='Inter'),
                    title_font=dict(size=15, color='#2563eb'),
                    xaxis=dict(tickangle=-30, color='#64748b'),
                    yaxis=dict(color='#64748b'), showlegend=False, height=360,
                )
                st.plotly_chart(_fig2, use_container_width=True)

        with tab3:
            st.markdown("#### 🕸️ Multi-Metric Radar Chart (Top 5 Models)")
            if 'Model' in compare_df.columns:
                if task_type == "classification":
                    _radar_metrics = [c for c in ['Accuracy', 'AUC', 'F1', 'Recall', 'Prec.'] if c in compare_df.columns]
                else:
                    _radar_metrics = [c for c in ['R2', 'RMSE', 'MAE'] if c in compare_df.columns]

                _top5 = compare_df.head(5).copy()
                _top5[_radar_metrics] = _top5[_radar_metrics].apply(pd.to_numeric, errors='coerce')

                if len(_radar_metrics) >= 3:
                    _radar_fig = go.Figure()
                    _colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626']
                    for _idx, (_row_i, _row) in enumerate(zip(range(len(_top5)), _top5.itertuples())):
                        _vals = [getattr(_row, m, 0) or 0 for m in _radar_metrics]
                        _vals_closed = _vals + [_vals[0]]
                        _cats_closed = _radar_metrics + [_radar_metrics[0]]
                        _model_name = _row.Model if hasattr(_row, 'Model') else f"Model {_row_i+1}"
                        _radar_fig.add_trace(go.Scatterpolar(
                            r=_vals_closed, theta=_cats_closed,
                            fill='toself', name=_model_name,
                            line=dict(color=_colors[_idx % 5], width=2),
                            fillcolor=_colors[_idx % 5].replace('#', 'rgba(').replace(')', ', 0.08)') if False else f"rgba(0,0,0,0.02)",
                        ))
                    _radar_fig.update_layout(
                        polar=dict(
                            bgcolor='rgba(248,250,255,0.8)',
                            radialaxis=dict(visible=True, color='#94a3b8', gridcolor='rgba(37,99,235,0.1)'),
                            angularaxis=dict(color='#334155', gridcolor='rgba(37,99,235,0.1)'),
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#334155', family='Inter'),
                        legend=dict(font=dict(color='#334155')),
                        height=500,
                    )
                    st.plotly_chart(_radar_fig, use_container_width=True)
                else:
                    st.info("Not enough metrics for radar chart.")
            else:
                st.info("Model column not available for radar chart.")

        with tab4:
            _sel_feats = st.session_state.get("selected_features") or [c for c in df.columns if c != target_col]
            fi_fig = build_feature_importance_chart(final_model, _sel_feats)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ Feature importance not available for this model type.
                </div>""", unsafe_allow_html=True)
                st.markdown("#### 📊 Feature Statistics")
                st.dataframe(df.drop(columns=[target_col]).describe(), use_container_width=True)

        with tab5:
            st.markdown("#### 🔬 Model Diagnostics on Holdout Set")
            _holdout = st.session_state.get("holdout_preds")
            if _holdout is not None:
                if task_type == "classification":
                    # Confusion Matrix
                    _pred_col = "prediction_label" if "prediction_label" in _holdout.columns else "Label"
                    if _pred_col in _holdout.columns and target_col in _holdout.columns:
                        from sklearn.metrics import confusion_matrix, classification_report
                        _y_true = _holdout[target_col]
                        _y_pred = _holdout[_pred_col]
                        _labels = sorted(_y_true.unique())
                        _cm = confusion_matrix(_y_true, _y_pred, labels=_labels)
                        _cm_fig = px.imshow(
                            _cm, x=[str(l) for l in _labels], y=[str(l) for l in _labels],
                            color_continuous_scale="Blues", text_auto=True,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title="Confusion Matrix (Holdout Set)",
                        )
                        _cm_fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#334155', family='Inter'),
                            title_font=dict(size=16, color='#2563eb'), height=420,
                        )
                        st.plotly_chart(_cm_fig, use_container_width=True)

                        # Classification Report
                        with st.expander("📋 Full Classification Report"):
                            _report = classification_report(_y_true, _y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(_report).T.round(4), use_container_width=True)
                    else:
                        st.info("Prediction column not found in holdout results.")

                else:
                    # Residuals + Actual vs Predicted
                    _pred_col = "prediction_label" if "prediction_label" in _holdout.columns else None
                    if _pred_col and target_col in _holdout.columns:
                        _y_true = pd.to_numeric(_holdout[target_col], errors='coerce')
                        _y_pred = pd.to_numeric(_holdout[_pred_col], errors='coerce')
                        _residuals = _y_true - _y_pred

                        _d1, _d2 = st.columns(2)
                        with _d1:
                            _avp_fig = px.scatter(
                                x=_y_true, y=_y_pred,
                                labels={"x": "Actual", "y": "Predicted"},
                                title="Actual vs Predicted",
                                color_discrete_sequence=["#2563eb"],
                                opacity=0.7,
                            )
                            _avp_fig.add_shape(type="line",
                                x0=_y_true.min(), x1=_y_true.max(),
                                y0=_y_true.min(), y1=_y_true.max(),
                                line=dict(color="#dc2626", dash="dash", width=1.5))
                            _avp_fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#334155'), title_font=dict(color='#2563eb', size=15),
                                height=380,
                            )
                            st.plotly_chart(_avp_fig, use_container_width=True)

                        with _d2:
                            _res_fig = px.scatter(
                                x=_y_pred, y=_residuals,
                                labels={"x": "Predicted", "y": "Residuals"},
                                title="Residuals Plot",
                                color_discrete_sequence=["#7c3aed"], opacity=0.7,
                            )
                            _res_fig.add_hline(y=0, line_color="#dc2626", line_dash="dash")
                            _res_fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#334155'), title_font=dict(color='#7c3aed', size=15),
                                height=380,
                            )
                            st.plotly_chart(_res_fig, use_container_width=True)

                        # Residuals histogram
                        _rh_fig = px.histogram(_residuals, nbins=40, title="Residuals Distribution",
                                               color_discrete_sequence=["#059669"])
                        _rh_fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#334155'), title_font=dict(color='#059669', size=14),
                            height=300,
                        )
                        st.plotly_chart(_rh_fig, use_container_width=True)
                    else:
                        st.info("Prediction column not found in holdout results.")
            else:
                st.info("Run the pipeline first to see model diagnostics.")

        with tab6:
            st.markdown("#### 🔮 Live Prediction Panel")
            st.markdown("""
            <div class="info-box">
                Enter feature values below and click <strong>Predict</strong> to get
                a real-time prediction from the best trained model.
            </div>""", unsafe_allow_html=True)

            _feat_cols = st.session_state.get("selected_features") or [c for c in df.columns if c != target_col]
            _input_vals = {}
            _pred_cols_ui = st.columns(min(4, len(_feat_cols)))
            for _i, _fc in enumerate(_feat_cols):
                _col_ui = _pred_cols_ui[_i % min(4, len(_feat_cols))]
                _dtype = df[_fc].dtype
                if pd.api.types.is_numeric_dtype(_dtype):
                    _default_val = float(df[_fc].median()) if not df[_fc].isnull().all() else 0.0
                    _input_vals[_fc] = _col_ui.number_input(
                        _fc, value=_default_val, key=f"pred_{_fc}")
                else:
                    _uniq = df[_fc].dropna().unique().tolist()
                    _input_vals[_fc] = _col_ui.selectbox(
                        _fc, options=_uniq, key=f"pred_{_fc}")

            if st.button("🔮 Run Prediction", key="predict_btn", use_container_width=False):
                try:
                    _input_df = pd.DataFrame([_input_vals])
                    if task_type == "classification":
                        from pycaret.classification import predict_model as _pm
                    else:
                        from pycaret.regression import predict_model as _pm
                    _pred_result = _pm(final_model, data=_input_df, verbose=False)
                    _pred_label = _pred_result.get("prediction_label", _pred_result.iloc[:, -1]).values[0]

                    if task_type == "classification":
                        _pred_score_col = [c for c in _pred_result.columns if "score" in c.lower()]
                        _conf = f"  ·  Confidence: **{float(_pred_result[_pred_score_col[0]].values[0]):.2%}**" if _pred_score_col else ""
                        st.markdown(f"""
                        <div class="success-box" style="font-size:1.1rem;">
                            🎯 Predicted Class: <strong style="font-size:1.4rem; color:#059669;">{_pred_label}</strong>{_conf.replace('**','<b>').replace('**','</b>')}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box" style="font-size:1.1rem;">
                            📈 Predicted Value: <strong style="font-size:1.4rem; color:#2563eb;">{float(_pred_label):.4f}</strong>
                        </div>""", unsafe_allow_html=True)

                    with st.expander("📋 Full Prediction DataFrame"):
                        st.dataframe(_pred_result, use_container_width=True)

                except Exception as _pe:
                    st.error(f"Prediction error: {_pe}")

        with tab7:
            st.markdown("#### PyCaret Setup Configuration")
            if setup_df is not None:
                st.dataframe(setup_df, use_container_width=True)
            else:
                st.info("Setup config not available.")

        # ── Downloads ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='section-title'>⬇️ Download Results</div>",
                    unsafe_allow_html=True)

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            try:
                model_path = st.session_state.get("model_save_path")
                if model_path and os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="🤖 Download Best Model (.pkl)",
                            data=f.read(),
                            file_name=f"best_{task_type}_model.pkl",
                            mime="application/octet-stream",
                        )
            except Exception:
                pass

        with dl2:
            # Cleaned dataset download
            _clean_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="🗂️ Download Cleaned Dataset (.csv)",
                data=_clean_csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )

        with dl3:
            report_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>AutoML Report – {task_type.capitalize()}</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;margin:0;padding:32px}}
  h1{{color:#2563eb;font-size:2rem}} h2{{color:#7c3aed;border-bottom:1px solid #e2e8f0;padding-bottom:8px}}
  table{{border-collapse:collapse;width:100%;margin:16px 0}}
  th{{background:#eff6ff;color:#2563eb;padding:10px 14px;text-align:left}}
  td{{padding:8px 14px;border-bottom:1px solid #e2e8f0;color:#334155}}
  .badge{{display:inline-block;padding:6px 16px;border-radius:50px;font-weight:600;
          background:{'#d1fae5' if task_type=='classification' else '#dbeafe'};
          color:{'#065f46' if task_type=='classification' else '#1d4ed8'}}}
  footer{{color:#94a3b8;margin-top:40px;font-size:0.8rem}}
</style></head><body>
  <h1>📊 AutoML Studio Report</h1>
  <p><strong>Task:</strong> <span class="badge">{task_type.capitalize()}</span></p>
  <p><strong>Target:</strong> {target_col} &nbsp;|&nbsp;
     <strong>Dataset:</strong> {df.shape[0]:,} rows × {df.shape[1]} cols</p>
  <p><strong>Best Model:</strong> {best_name}</p>
  <h2>📊 Model Comparison</h2>
  {compare_df.to_html(classes='', border=0)}
  <footer>Generated by AutoML Studio (PyCaret + Streamlit) — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</footer>
</body></html>"""
            st.download_button(
                label="📄 Download HTML Report",
                data=report_html.encode("utf-8"),
                file_name=f"automl_report_{task_type}.html",
                mime="text/html",
            )

    elif not run_btn and not st.session_state.ran_pipeline:
        st.markdown("""
        <div class="info-box" style="margin-top:20px;">
            👈 <strong>Ready!</strong> Click <strong>"🚀 Run ML Pipeline"</strong> in the sidebar
            to start training. This may take 2–5 minutes depending on dataset size.
        </div>
        """, unsafe_allow_html=True)

