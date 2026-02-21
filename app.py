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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════
   COLOR PALETTE
   Background  : #080d1a  →  #0b1120  →  #0e1428
   Primary     : #38bdf8  (sky-400)   — cyan-blue
   Secondary   : #818cf8  (indigo-400)
   Accent      : #34d399  (emerald-400)
   Warning     : #fbbf24  (amber-400)
   Danger      : #f87171  (red-400)
   Text-bright : #f1f5f9
   Text-muted  : #94a3b8
══════════════════════════════════════════════════════════ */

/* ── Global ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #cbd5e1;
}

.stApp {
    background: radial-gradient(ellipse at 0% 0%, #0d1f35 0%, #080d1a 45%, #06091a 100%);
    min-height: 100vh;
}

/* ── Hero Banner ────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(14,30,58,0.95) 0%,
        rgba(10,20,45,0.95) 50%,
        rgba(8,13,26,0.98) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(56,189,248,0.05), 0 4px 40px rgba(0,0,0,0.5);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px;
    right: -60px;
    width: 480px;
    height: 480px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px;
    left: -40px;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(129,140,248,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 12px;
    font-weight: 400;
    max-width: 680px;
    line-height: 1.6;
}

/* ── Glass Cards ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(14, 22, 44, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.glass-card:hover {
    border-color: rgba(56,189,248,0.25);
    background: rgba(14,22,44,0.8);
    box-shadow: 0 4px 30px rgba(56,189,248,0.08);
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
    background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(129,140,248,0.06));
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    color: #38bdf8;
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
    background: linear-gradient(135deg, rgba(20,184,166,0.15), rgba(52,211,153,0.1));
    border: 1px solid rgba(52,211,153,0.5);
    color: #34d399;
    font-weight: 600;
    padding: 8px 22px;
    border-radius: 50px;
    font-size: 0.95rem;
    animation: pulse-teal 2.5s infinite;
    box-shadow: 0 0 16px rgba(52,211,153,0.1);
}
.badge-regression {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(96,165,250,0.1));
    border: 1px solid rgba(56,189,248,0.5);
    color: #38bdf8;
    font-weight: 600;
    padding: 8px 22px;
    border-radius: 50px;
    font-size: 0.95rem;
    animation: pulse-sky 2.5s infinite;
    box-shadow: 0 0 16px rgba(56,189,248,0.1);
}
@keyframes pulse-teal {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    50% { box-shadow: 0 0 0 10px rgba(52,211,153,0.12); }
}
@keyframes pulse-sky {
    0%, 100% { box-shadow: 0 0 0 0 rgba(56,189,248,0); }
    50% { box-shadow: 0 0 0 10px rgba(56,189,248,0.12); }
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
.step-pending  { background: rgba(15,23,42,0.5); color: #475569; }
.step-active   { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.step-done     { background: rgba(52,211,153,0.08); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }

/* ── Section Titles ─────────────────────────────────────────────── */
.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56,189,248,0.12);
    letter-spacing: -0.01em;
}

/* ── Sidebar styling ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b18 0%, #080d1a 60%, #060b18 100%);
    border-right: 1px solid rgba(56,189,248,0.08);
}
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
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
    box-shadow: 0 4px 20px rgba(14,165,233,0.3);
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(14,165,233,0.45);
}

/* ── Main area buttons ───────────────────────────────────────────── */
.stDownloadButton>button {
    background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.1)) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}
.stDownloadButton>button:hover {
    background: linear-gradient(135deg, rgba(14,165,233,0.25), rgba(99,102,241,0.18)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(56,189,248,0.25) !important;
}

/* ── Alert-like info boxes ───────────────────────────────────────── */
.info-box {
    background: rgba(56,189,248,0.06);
    border-left: 3px solid #38bdf8;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #7dd3fc;
    font-size: 0.9rem;
    margin: 12px 0;
}
.warning-box {
    background: rgba(251,191,36,0.07);
    border-left: 3px solid #fbbf24;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #fde68a;
    font-size: 0.9rem;
    margin: 12px 0;
}
.success-box {
    background: rgba(52,211,153,0.07);
    border-left: 3px solid #34d399;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    color: #6ee7b7;
    font-size: 0.9rem;
    margin: 12px 0;
}

/* ── Streamlit native widgets — tint overrides ───────────────────── */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stTab"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* ── DataFrame styling ───────────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.1) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(14,22,44,0.6) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Progress bar ────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #0ea5e9, #818cf8) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080d1a; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.45); }

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────────────
for key in ["df", "task_type", "results", "best_model", "final_model",
            "compare_df", "setup_df", "ran_pipeline", "target_col"]:
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
        font=dict(color='#cbd5e1', family='Inter'),
        title_font=dict(size=16, color='#38bdf8'),
        xaxis=dict(gridcolor='rgba(56,189,248,0.06)', color='#64748b'),
        yaxis=dict(gridcolor='rgba(56,189,248,0.06)', color='#64748b'),
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
            paper_bgcolor='rgba(8,13,26,0)',
            font=dict(color='#cbd5e1', family='Inter'),
            title_font=dict(size=16, color='#38bdf8'),
            xaxis=dict(gridcolor='rgba(56,189,248,0.06)', color='#64748b'),
            yaxis=dict(gridcolor='rgba(56,189,248,0.06)', color='#64748b'),
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
        font=dict(color='#cbd5e1', family='Inter'),
        title_font=dict(size=15, color='#38bdf8'),
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-family:"Plus Jakarta Sans",sans-serif; font-size:1.3rem;
                    font-weight:800; color:#38bdf8; letter-spacing:-0.02em;'>AutoML Studio</div>
        <div style='color:#475569; font-size:0.8rem; margin-top:6px; font-weight:500;'>
            Powered by PyCaret
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08); margin: 16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Dataset")
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

        st.markdown("")
        st.markdown("### ⚙️ Pipeline Settings")
        fold_n = st.slider("Cross-validation Folds", 3, 10, 5, 1)
        session_id = st.number_input("Session ID (seed)", value=42, step=1)

        st.markdown("---")
        run_btn = st.button("🚀 Run ML Pipeline", use_container_width=True)
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

    # ── EDA Summary ───────────────────────────────────────────────────────────
    eda = get_eda_summary(df)

    st.markdown("<div class='section-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Rows", f"{eda['rows']:,}")
    with m2:
        st.metric("Columns", eda['cols'])
    with m3:
        st.metric("Missing Values", eda['missing'])
    with m4:
        st.metric("Duplicates", eda['duplicates'])
    with m5:
        st.metric("Numeric Cols", eda['numeric_cols'])
    with m6:
        st.metric("Categorical Cols", eda['cat_cols'])

    # ── Data Preview ──────────────────────────────────────────────────────────
    with st.expander("🔍 Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True, height=280)

    # ── EDA Visuals ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-title'>🎯 Target Distribution</div>",
                    unsafe_allow_html=True)
        target_fig = build_target_distribution(df, target_col, task_type)
        st.plotly_chart(target_fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>",
                    unsafe_allow_html=True)
        corr_fig = build_correlation_heatmap(df)
        if corr_fig:
            st.pyplot(corr_fig, use_container_width=True)
        else:
            st.markdown("""<div class="warning-box">Not enough numeric columns for correlation heatmap.</div>""",
                        unsafe_allow_html=True)

    # ── Missing Values Chart ──────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        with st.expander("⚠️ Missing Values Analysis"):
            miss_df = missing.reset_index()
            miss_df.columns = ['Column', 'Missing Count']
            miss_df['Missing %'] = (miss_df['Missing Count'] / len(df) * 100).round(2)
            fig_miss = px.bar(
                miss_df, x='Column', y='Missing %',
                color='Missing %',
                color_continuous_scale='RdYlGn_r',
                title="Missing Values (%)",
            )
            fig_miss.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                title_font=dict(size=15, color='#f59e0b'),
                height=320,
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    # ── Task type display ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🔍 Detected Task Type</div>",
                unsafe_allow_html=True)

    tc1, tc2 = st.columns([1, 2])
    with tc1:
        if task_type == "classification":
            n_classes = df[target_col].nunique()
            st.markdown(f"""
            <div class="badge-classification" style="font-size:1.1rem; padding:12px 24px;">
                🧠 Classification
            </div>
            <div style="color:rgba(160,170,200,0.7); font-size:0.85rem; margin-top:12px;">
                Target: <strong style="color:#38bdf8">{target_col}</strong> 
                · {n_classes} unique classes<br>
                Using <strong>PyCaret Classification</strong> pipeline
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="badge-regression" style="font-size:1.1rem; padding:12px 24px;">
                📈 Regression
            </div>
            <div style="color:rgba(160,170,200,0.7); font-size:0.85rem; margin-top:12px;">
                Target: <strong style="color:#38bdf8">{target_col}</strong>
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
            You can override by selecting a different target column.
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
                    if task_type == "classification":
                        from pycaret.classification import (
                            setup, compare_models, finalize_model, pull, save_model
                        )
                        env = setup(
                            data=df,
                            target=target_col,
                            session_id=int(session_id),
                            fold=fold_n,
                            verbose=False,
                            html=False,
                        )
                        setup_df = pull()
                        st.session_state.setup_df = setup_df

                    else:
                        from pycaret.regression import (
                            setup, compare_models, finalize_model, pull, save_model
                        )
                        env = setup(
                            data=df,
                            target=target_col,
                            session_id=int(session_id),
                            fold=fold_n,
                            verbose=False,
                            html=False,
                        )
                        setup_df = pull()
                        st.session_state.setup_df = setup_df

                elif i == 1:
                    best_model = compare_models(verbose=False)
                    compare_df = pull()
                    st.session_state.best_model = best_model
                    st.session_state.compare_df = compare_df

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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Model Leaderboard",
            "📈 Metric Charts",
            "🌟 Feature Importance",
            "⚙️ Setup Config",
        ])

        with tab1:
            st.markdown("#### All Models Ranked by Performance")
            st.markdown("""
            <div class="info-box">
                🏅 The <strong>top row</strong> is the best-performing model selected by PyCaret.
                Yellow highlights indicate the best value per metric column.
            </div>""", unsafe_allow_html=True)
            st.dataframe(compare_df, use_container_width=True, height=450)

            # Download comparison as CSV
            csv_data = compare_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="⬇️ Download Model Comparison CSV",
                data=csv_data,
                file_name=f"model_comparison_{task_type}.csv",
                mime="text/csv",
            )

        with tab2:
            st.markdown("#### Model Performance Comparison (Charts)")
            bar_fig = build_metric_bar_chart(compare_df, task_type)
            st.plotly_chart(bar_fig, use_container_width=True)

            # Secondary metric chart
            if task_type == "classification":
                alt_metrics = [c for c in ['F1', 'Kappa', 'Recall', 'Prec.'] if c in compare_df.columns]
            else:
                alt_metrics = [c for c in ['MAE', 'RMSE', 'MAPE'] if c in compare_df.columns]

            if alt_metrics and 'Model' in compare_df.columns:
                sec_metric = alt_metrics[0]
                df_plot = compare_df[['Model', sec_metric]].copy()
                df_plot[sec_metric] = pd.to_numeric(df_plot[sec_metric], errors='coerce')
                df_plot = df_plot.dropna().sort_values(sec_metric)
                fig2 = px.line(
                    df_plot, x='Model', y=sec_metric,
                    markers=True,
                    title=f"Model Comparison by {sec_metric}",
                    color_discrete_sequence=['#60a5fa'],
                )
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', family='Inter'),
                    title_font=dict(size=15, color='#60a5fa'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', color='#9ca3af', tickangle=-30),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', color='#9ca3af'),
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            feature_names = [c for c in df.columns if c != target_col]
            fi_fig = build_feature_importance_chart(final_model, feature_names)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ Feature importance is not available for this model type
                    (e.g., SVM, KNN, or Linear models without explicit coefficients
                    may show limited importance).
                </div>""", unsafe_allow_html=True)

                # Show descriptive stats for the features instead
                st.markdown("#### 📊 Feature Statistics")
                st.dataframe(df.drop(columns=[target_col]).describe(), use_container_width=True)

        with tab4:
            st.markdown("#### PyCaret Setup Configuration")
            if setup_df is not None:
                st.dataframe(setup_df, use_container_width=True)
            else:
                st.info("Setup config not available.")

        # ── Downloads ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='section-title'>⬇️ Download Results</div>",
                    unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)
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
            # Generate HTML report
            report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoML Report – {task_type.capitalize()}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e2e8f0; margin: 0; padding: 32px; }}
  h1 {{ color: #a78bfa; font-size: 2rem; }}
  h2 {{ color: #60a5fa; border-bottom: 1px solid #1f2937; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #1e1b4b; color: #a78bfa; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #1f2937; color: #d1d5db; }}
  tr:hover td {{ background: rgba(99,102,241,0.1); }}
  .badge {{ display:inline-block; padding:6px 16px; border-radius:50px; font-weight:600;
            background:{'#065f46' if task_type=='classification' else '#1e3a5f'};
            color:{'#6ee7b7' if task_type=='classification' else '#93c5fd'}; }}
  .info {{ background:#1e2a3a; border-left:3px solid #3b82f6; padding:12px; border-radius:4px; }}
  footer {{ color: #4b5563; margin-top:40px; font-size:0.8rem; }}
</style>
</head>
<body>
  <h1>🤖 AutoML Studio Report</h1>
  <p><strong>Task Type:</strong> <span class="badge">{task_type.capitalize()}</span></p>
  <p><strong>Target Column:</strong> {target_col}</p>
  <p><strong>Dataset Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
  <p><strong>Best Model:</strong> {best_name}</p>
  
  <h2>📊 Model Comparison</h2>
  {compare_df.to_html(classes='', border=0)}
  
  <footer>Generated by AutoML Studio (PyCaret + Streamlit) — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</footer>
</body>
</html>"""
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
            to start training and comparing models. This may take 2–5 minutes depending on dataset size.
        </div>
        """, unsafe_allow_html=True)
