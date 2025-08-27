# style.py
# Estilos globales + helpers de UI/tema para el dashboard (título responsive fijo)

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

# ===== Paletas de variables para claro/oscuro =====
LIGHT_VARS = {
    "--bg": "#ffffff",
    "--panel": "#f8fafc",
    "--ink": "#0f172a",
    "--muted": "#64748b",
    "--border": "#e5e7eb",
    "--accent": "#0ea5e9",
}
DARK_VARS = {
    "--bg": "#0b1220",
    "--panel": "#0f172a",
    "--ink": "#e5e7eb",
    "--muted": "#cbd5e1",
    "--border": "#1f2937",
    "--accent": "#22d3ee",
}

def _vars_to_css(vars_map: dict[str, str]) -> str:
    return "; ".join(f"{k}:{v}" for k, v in vars_map.items())

def base_css(dark: bool) -> str:
    theme_vars = DARK_VARS if dark else LIGHT_VARS
    return f"""
<style>
:root {{
  {_vars_to_css(theme_vars)};
  --header-h: 64px;   /* altura del header de Streamlit */
  --pad-x: 16px;
}}

/* ====== Header fijo y separación del contenido ====== */
header[data-testid="stHeader"] {{
  height: var(--header-h) !important;
  background: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
}}

/* Empuja TODO el contenedor principal para que el header no tape el título */
div[data-testid="stAppViewContainer"] {{
  padding-top: calc(var(--header-h) + 16px) !important;
}}
/* Evita doble padding en hijos (muy importante) */
div[data-testid="stAppViewContainer"] > .main,
main .block-container {{
  padding-top: 0 !important;
}}

/* Fondo y colores base */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {{
  background-color: var(--bg) !important;
  color: var(--ink) !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}}

/* Inputs */
[data-baseweb="select"]>div {{
  border-radius: 10px; border:1px solid var(--border); background: var(--bg);
}}
[data-baseweb="select"]>div:focus-within {{
  box-shadow: 0 0 0 2px var(--accent);
  border-color: var(--accent);
}}
.stDateInput input, input, textarea {{
  background: { 'var(--panel)' if dark else 'var(--bg)' } !important;
  color: var(--ink) !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
}}

/* ====== Hero (título principal) ====== */
div.hero {{
  margin: 0 !important;
  width: 100%;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}}
h1.hero-title {{
  margin: 0;
  line-height: 1.2;
  font-weight: 800;
  /* tamaño fluido/responsive */
  font-size: clamp(22px, 2.8vw + 10px, 34px);
  text-wrap: balance;
  color: var(--ink);
}}
.hero-subtitle {{
  margin: 6px 0 0 0;
  font-size: clamp(12px, 1.1vw + 8px, 15px);
  color: var(--muted);
}}

/* ====== Secciones / tarjetas ====== */
div.section {{
  border:1px solid var(--border);
  border-radius:12px;
  padding:10px 12px;
  background: var(--panel);
  margin:14px 0 8px 0;
}}
h2.section-title {{
  font-weight:700; font-size:18px; margin:0; color: var(--ink);
}}
.kpi-card {{
  border:1px solid var(--border); background: var(--panel);
  border-radius:14px; padding:12px 14px; color: var(--ink);
}}
.kpi-title {{ display:flex; gap:8px; font-weight:800; font-size:20px; margin:2px 0 12px 0; }}
.kpi-item .label {{ color: var(--muted); font-size:13px; margin-bottom:2px; }}
.kpi-item .value {{ color: var(--ink); font-size:32px; font-weight:800; }}
</style>
"""

def inject(dark: bool) -> None:
    """Inyecta el CSS base en la página."""
    st.markdown(base_css(dark), unsafe_allow_html=True)

def render_hero(title: str, subtitle: str | None = None) -> None:
    """Pinta el hero del título."""
    sub = f'<div class="hero-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
<div class="hero">
  <h1 class="hero-title">{title}</h1>
  {sub}
</div>
""", unsafe_allow_html=True)

def apply_plot_theme(fig: go.Figure, dark: bool) -> None:
    """Tema de Plotly coherente con el modo claro/oscuro."""
    fig.update_layout(
        template=("plotly_dark" if dark else "plotly_white"),
        paper_bgcolor=("#0f172a" if dark else "#ffffff"),
        plot_bgcolor=("#0b1220" if dark else "#ffffff"),
        font=dict(color=("#e5e7eb" if dark else "#0f172a")),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")
