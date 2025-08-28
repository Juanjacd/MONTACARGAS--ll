# =========================================================
# DASHBOARD MONTACARGAS — TM + Órdenes OT + Inicio/Fin (PC claro, móvil oscuro)
# =========================================================

# ---------------- [S0] Imports y setup -------------------
from cfg import APP_TITLE, APP_TAGLINE

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3, hashlib, re, unicodedata
from datetime import time as dtime
from typing import Optional, List
from collections import Counter
import datetime

st.set_page_config(page_title=APP_TITLE, layout="wide")
BUILD = "v2025-08-27-6"

# =================== Estilos base ===================
st.markdown(f"""
<style>
:root{{
  --bg:#ffffff; --panel:#f8fafc; --ink:#0f172a; --muted:#64748b; --border:#e5e7eb; --accent:#0ea5e9;
  --header-h:64px;
}}
header[data-testid="stHeader"]{{height:var(--header-h)!important;background:var(--bg)!important;border-bottom:1px solid var(--border)!important}}
[data-testid="stAppViewContainer"]>.main{{padding-top:calc(var(--header-h)+12px)!important}}
main .block-container{{padding-top:0!important}}
html, body, #root, .stApp, main, .main, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{{
  background:var(--bg)!important; color:var(--ink)!important;
}}
section[data-testid="stSidebar"]{{background:var(--panel)!important;border-right:1px solid var(--border)}}

/* Hero */
div.hero{{margin:0!important;width:100%;border:1px solid var(--border);border-radius:14px;padding:14px 16px;background:var(--panel)}}
.hero-wrap{{display:flex;flex-direction:column;gap:.25rem;width:100%}}
h1.hero-title{{margin:0;line-height:1.15;font-weight:800;color:var(--ink);font-size:clamp(20px,2.6vw + 8px,34px);text-wrap:balance;overflow-wrap:anywhere}}
div.hero-sub{{font-size:clamp(12px,1.1vw + 8px,15px);color:var(--muted)}}

/* Expanders visibles */
div[data-testid="stExpander"] summary{{color:var(--ink)!important;font-weight:700!important;visibility:visible!important;opacity:1!important}}
div[data-testid="stExpander"] summary svg{{stroke:var(--ink)!important}}

/* Inputs */
[data-baseweb="select"]>div{{border-radius:10px;border:1px solid var(--border);background:var(--bg)}}
[data-baseweb="select"]>div:focus-within{{box-shadow:0 0 0 2px var(--accent);border-color:var(--accent)}}
input, textarea{{background:var(--bg)!important;color:var(--ink)!important;border-radius:10px!important;border:1px solid var(--border)!important}}

/* Compacto móvil */
@media (max-width:768px){{
  [data-testid="stHorizontalBlock"]>div,[data-testid="column"]{{flex:1 0 100%!important;width:100%!important}}
  .block-container{{padding-left:.5rem;padding-right:.5rem}}
}}
</style>

<div class="hero"><div class="hero-wrap">
  <h1 class="hero-title">{APP_TITLE}</h1>
  <div class="hero-sub">{APP_TAGLINE}</div>
</div></div>
""", unsafe_allow_html=True)

# =================== Sidebar ===================
with st.sidebar:
    st.caption(f"Build: {BUILD}")
    st.markdown("---")

# Detectar compacto (móvil vs PC) por preferencia de UI
def is_compact() -> bool:
    return st.session_state.get("compact", True)

def set_compact(v: bool):
    st.session_state["compact"] = bool(v)

# Configuración inicial de dark según “compact” (móvil oscuro por defecto, PC claro)
if "dark" not in st.session_state:
    st.session_state["dark"] = True if is_compact() else False
dark = st.sidebar.checkbox("🌙 Modo oscuro", value=st.session_state["dark"])
st.session_state["dark"] = dark

# CSS tema oscuro
if dark:
    st.markdown("""
    <style>
    :root{ --bg:#0b1220; --panel:#0f172a; --ink:#e5e7eb; --muted:#cbd5e1; --border:#1f2937; --accent:#22d3ee;}
    html, body, #root, .stApp, main, .main, .block-container,
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], header[data-testid="stHeader"]{
      background-color: var(--bg) !important; color: var(--ink) !important;
    }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] *{
      background-color: transparent !important; color: var(--ink) !important;
    }
    .hero, .section, .kpi-card,
    div[data-testid="stExpander"] details{ background: var(--panel) !important; color: var(--ink) !important; border-color: var(--border) !important; }
    [data-baseweb="select"] > div{ background: var(--panel) !important; border:1px solid var(--border)!important; border-radius:10px!important; }
    .stDateInput input, input, textarea{ background: var(--panel)!important; color: var(--ink)!important; border:1px solid var(--border)!important; }
    </style>
    """, unsafe_allow_html=True)

# =================== Funciones de estilo ===================
def apply_plot_theme(fig):
    base_font = 11 if is_compact() else 13
    leg_font = 9 if is_compact() else 11
    fig.update_layout(
        template=("plotly_dark" if dark else "plotly_white"),
        paper_bgcolor=("#0f172a" if dark else "#ffffff"),
        plot_bgcolor=("#0b1220" if dark else "#ffffff"),
        font=dict(color=("#e5e7eb" if dark else "#0f172a"), size=base_font),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=leg_font),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=60, b=(40 if not is_compact() else 60), l=10, r=10),
        showlegend=True
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickformat=".4f")
    fig.update_yaxes(showgrid=False, zeroline=False, tickformat=".4f")

# =========================================================
# [S2] Reglas de turnos
# =========================================================
THRESH_MIN = 15
TURNOS = {
    "Turno A": {"start": dtime(5, 0),  "end": dtime(13, 55),
                "lunch": (dtime(8, 20), dtime(9, 0)),
                "fuel":  (dtime(9, 0),  dtime(9, 18))},
    "Turno B": {"start": dtime(14, 0), "end": dtime(22, 55),
                "lunch": (dtime(17, 25), dtime(18, 0)),
                "fuel":  (dtime(18, 0),  dtime(18, 18))},
}
LATE_B_CUTOFF = dtime(3, 0)

# =========================================================
# [S3] Normalización / clasificación utilitaria
# =========================================================
def _norm_nfd_ascii(s: str) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()

def _norm_compact(s: str) -> str:
    s = _norm_nfd_ascii(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def pick_col(cols_map: dict, *aliases) -> Optional[str]:
    def normkey(x): return _norm_compact(x)
    norm2orig = {normkey(k): v for k, v in cols_map.items()}
    for alias in aliases:
        key = normkey(alias)
        for nk, orig in norm2orig.items():
            if key in nk or nk in key: return orig
    return None

def to_time(x):
    if pd.isna(x): return None
    try:
        if hasattr(x, "hour"):
            return dtime(int(x.hour), int(getattr(x,"minute",0)), int(getattr(x,"second",0)))
    except Exception: pass
    s = str(x).strip()
    for fmt in ("%H:%M:%S","%H:%M"):
        try:
            dtv = pd.to_datetime(s, format=fmt); return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv,"second",0)))
        except Exception: pass
    dtv = pd.to_datetime(s, errors="coerce")
    return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv,"second",0))) if pd.notnull(dtv) else None

def turno_by_time(t: dtime):
    if t is None: return None
    a_start, a_end = TURNOS["Turno A"]["start"], TURNOS["Turno A"]["end"]
    b_start, b_end = TURNOS["Turno B"]["start"], TURNOS["Turno B"]["end"]
    if a_start <= t < a_end:  return "Turno A"
    if b_start <= t <= b_end: return "Turno B"
    if t < LATE_B_CUTOFF or t > b_end: return "Turno B"
    return None

# =========================================================
# [S4] SQLite (persistencia)
# =========================================================
TABLE = "ordenes"
def ensure_db(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            id TEXT PRIMARY KEY,
            usuario TEXT, fecha TEXT, time TEXT, turno TEXT, datetime TEXT,
            orden TEXT, ubic_proced TEXT, ubic_destino TEXT, itemraw TEXT
        )
    """)
    con.commit(); con.close()

def make_uid(row):
    dtv = pd.to_datetime(row.get("Datetime"), errors="coerce")
    dtv = pd.NaT if pd.isna(dtv) else dtv.floor("min")
    base = f"{row.get('Usuario','')}|{row.get('Fecha','')}|{str(dtv)}|{str(row.get('Orden') or '')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def upsert_df(path, df):
    if df.empty: return 0
    con = sqlite3.connect(path); cur = con.cursor()
    df2 = df.copy(); df2["Datetime"] = pd.to_datetime(df2["Datetime"]).dt.floor("min")
    df2["id"] = df2.apply(make_uid, axis=1)
    rows = []
    for _, r in df2.iterrows():
        rows.append((r["id"], r.get("Usuario"),
                     str(r.get("Fecha")) if pd.notnull(r.get("Fecha")) else None,
                     str(r.get("Time")) if pd.notnull(r.get("Time")) else None,
                     r.get("Turno"),
                     r.get("Datetime").isoformat() if pd.notnull(r["Datetime"]) else None,
                     str(r.get("Orden") or None),
                     r.get("Ubic.proced"), r.get("Ubicación de destino"),
                     r.get("ItemRaw") if "ItemRaw" in df2.columns else None))
    cur.executemany(
        f"""INSERT OR REPLACE INTO {TABLE}
            (id,usuario,fecha,time,turno,datetime,orden,ubic_proced,ubic_destino,itemraw)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows
    )
    con.commit(); con.close()
    return len(rows)

def read_all(path) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    except Exception:
        df = pd.DataFrame(columns=["id","usuario","fecha","time","turno","datetime","orden","ubic_proced","ubic_destino","itemraw"])
    con.close()
    if df.empty: return df
    df.rename(columns={"usuario":"Usuario","fecha":"Fecha","time":"TimeStr",
                       "turno":"Turno","datetime":"Datetime","orden":"Orden",
                       "ubic_proced":"Ubic.proced","ubic_destino":"Ubicación de destino",
                       "itemraw":"ItemRaw"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.date
    df["Time"] = df["Datetime"].dt.time
    return df

def clear_db(path):
    con = sqlite3.connect(path); con.execute(f"DELETE FROM {TABLE}"); con.commit(); con.close()

# =========================================================
# [S5] Fecha operativa + marca de horas extra
# =========================================================
def apply_oper_day(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Turno" not in d.columns or d["Turno"].isna().any():
        d["Turno"] = d["Time"].apply(turno_by_time)

    extra_mask = (d["Turno"] == "Turno B") & (d["Time"].apply(lambda x: x is not None and x < LATE_B_CUTOFF))
    d["IsExtra"] = extra_mask

    d["FechaOper"] = pd.to_datetime(d["Fecha"])
    d.loc[extra_mask, "FechaOper"] = d.loc[extra_mask, "FechaOper"] - pd.Timedelta(days=1)

    d["DatetimeOper"] = d.apply(
        lambda r: pd.Timestamp.combine(pd.Timestamp(r["FechaOper"]).date(), r["Time"]) if (pd.notnull(r["FechaOper"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    return d

# =========================================================
# [S6] Carga Excel
# =========================================================
@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name="Hoja1") -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    if sheet_name not in xls.sheet_names: sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    cols = {c.lower().strip(): c for c in df.columns}

    col_usuario = pick_col(cols, "usuario")
    col_fecha   = pick_col(cols, "fecha", "fecha confirmacion")
    col_hora    = pick_col(cols, "hora", "hora confirmacion")
    col_orden   = pick_col(cols, "orden", "numero de orden de transporte", "ot")

    if not col_usuario: raise ValueError("Falta columna 'Usuario'.")
    if not col_fecha:   raise ValueError("Falta columna 'Fecha'.")
    if not col_hora:    raise ValueError("Falta columna 'Hora'.")

    out = pd.DataFrame({
        "Usuario": df[col_usuario].astype(str).str.strip(),
        "Fecha":   pd.to_datetime(df[col_fecha], errors="coerce").dt.date,
        "Time":    df[col_hora].apply(to_time),
    })
    out["Hora"] = out["Time"].apply(lambda t: t.hour if t else None)
    out["Orden"] = df[col_orden].astype(str).str.strip() if col_orden else None
    out["Turno"] = out["Time"].apply(turno_by_time)
    out["Datetime"] = out.apply(
        lambda r: pd.Timestamp.combine(r["Fecha"], r["Time"]) if (pd.notnull(r["Fecha"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    out = out.dropna(subset=["Fecha","Time","Datetime"])
    return out[["Usuario","Fecha","Hora","Time","Turno","Datetime","Orden"]]

# =========================================================
# [S7] Sidebar: carga + preferencias + filtros
# =========================================================
with st.sidebar:
    with st.expander("📥 Carga de datos", expanded=False):
        up = st.file_uploader("📎 Excel (.xlsx)", type=["xlsx"])
        if up is not None:
            try:
                xls_tmp = pd.ExcelFile(up); hojas = xls_tmp.sheet_names
                hoja = st.selectbox("Hoja", hojas, index=hojas.index("Hoja1") if "Hoja1" in hojas else 0)
            except Exception:
                hoja = st.text_input("Hoja", value="Hoja1")
            finally:
                if hasattr(up,"seek"): up.seek(0)
        else:
            hoja = st.text_input("Hoja", value="Hoja1")

        st.caption("Histórico SQLite")
        use_db = st.checkbox("Usar histórico", value=True)
        DB_PATH = st.text_input("Archivo DB", value="montacargas.db")
        col1, col2 = st.columns(2)
        with col1: btn_clear = st.button("🧹 Limpiar histórico")
        with col2: btn_reload = st.button("🔁 Recargar histórico")

    with st.expander("⚙️ Preferencias", expanded=False):
        chart_type = st.selectbox("Orientación (TM y Órdenes)", ["Barra horizontal", "Barra vertical"], index=0)
        compact_ui = st.checkbox("📱 Modo compacto (móvil)", value=True)
        set_compact(compact_ui)
        st.session_state["chart_type"] = chart_type

if up is None and 'use_db' in locals() and not use_db:
    st.warning("Sube un Excel para empezar o activa el histórico."); st.stop()

df_new = pd.DataFrame()
if up is not None:
    try: df_new = load_excel(up, hoja)
    except Exception as e:
        st.error(f"❌ No pude leer el Excel: {e}"); st.stop()

if 'use_db' not in locals(): use_db = True
if 'DB_PATH' not in locals(): DB_PATH = "montacargas.db"

ensure_db(DB_PATH)
if use_db:
    if btn_clear: clear_db(DB_PATH); st.success("Histórico limpiado.")
    if not df_new.empty: upsert_df(DB_PATH, df_new)
    if btn_reload:
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Recargado.")
        try: st.rerun()
        except Exception: st.experimental_rerun()
    df = read_all(DB_PATH)
else:
    df = df_new.copy()

if df.empty:
    st.info("No hay datos para visualizar aún."); st.stop()

# --- Fecha operativa aplicada
df = apply_oper_day(df)
df = df[df["Turno"].isin(["Turno A","Turno B"])].copy()

# ---------------- Filtros ----------------
with st.sidebar:
    users = sorted(df["Usuario"].dropna().unique().tolist())
    turns = ["Turno A","Turno B"]
    fmin, fmax = df["FechaOper"].min().date(), df["FechaOper"].max().date()

    sel_users = st.multiselect("Usuarios", users, [])
    sel_turns = st.multiselect("Turnos", turns, [])

    with st.form("f_fechas"):
        d0 = st.date_input("Desde", value=fmin, min_value=fmin, max_value=fmax, format="YYYY-MM-DD")
        d1 = st.date_input("Hasta",  value=fmax, min_value=fmin, max_value=fmax, format="YYYY-MM-DD")
        ok = st.form_submit_button("Aplicar")

if ok or "range" not in st.session_state:
    if d1 < d0: d0, d1 = d1, d0
    st.session_state["range"] = (d0, d1)

d0, d1 = st.session_state["range"]
start_ts = pd.Timestamp(d0)
end_ts   = pd.Timestamp(d1) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df_f = df.copy()
if sel_users: df_f = df_f[df_f["Usuario"].isin(sel_users)]
if sel_turns: df_f = df_f[df_f["Turno"].isin(sel_turns)]
df_f = df_f[(df_f["DatetimeOper"] >= start_ts) & (df_f["DatetimeOper"] <= end_ts)]

if df_f.empty:
    st.info("No hay datos con el filtro actual."); st.stop()

# =========================================================
# [S8] KPIs
# =========================================================
def render_kpis(df_filtered: pd.DataFrame):
    total = len(df_filtered); uniq_users = df_filtered["Usuario"].nunique()
    tA = int((df_filtered["Turno"] == "Turno A").sum())
    tB = int((df_filtered["Turno"] == "Turno B").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Órdenes (filtrado)", f"{total:,.4f}")
    c2.metric("Usuarios únicos", f"{uniq_users:,.4f}")
    c3.metric("Turno A", f"{tA:,.4f}")
    c4.metric("Turno B", f"{tB:,.4f}")

# =========================================================
# [S9] Vista — Tiempo Muerto
# =========================================================
def view_tm(df_in: pd.DataFrame):
    st.subheader("⏱ Tiempo muerto por usuario y turno")

    # gaps dentro de mismo día operativo y mismo turno
    df_g = df_in.sort_values(["Usuario","DatetimeOper"]).copy()
    df_g["prev_dt"] = df_g.groupby("Usuario")["DatetimeOper"].shift(1)
    df_g["prev_fecha"] = df_g.groupby("Usuario")["FechaOper"].shift(1)
    df_g["prev_turno"] = df_g.groupby("Usuario")["Turno"].shift(1)
    same = df_g["prev_dt"].notna() & (df_g["FechaOper"]==df_g["prev_fecha"]) & (df_g["Turno"]==df_g["prev_turno"])
    df_g = df_g[same].copy()

    rows = []
    for _, r in df_g.iterrows():
        start = r["prev_dt"]; end = r["DatetimeOper"]
        gap = (end - start).total_seconds()/60.0
        if gap > THRESH_MIN:
            rows.append({"Usuario":r["Usuario"],"Turno":r["Turno"],"GapMin":gap})
    dead = pd.DataFrame(rows)

    if dead.empty:
        st.info("No se detectaron gaps > 15 min."); return

    agg = dead.groupby(["Usuario","Turno"])["GapMin"].sum().reset_index()

    chart_is_h = (st.session_state.get("chart_type") == "Barra horizontal")
    cat_orders = {"Turno": ["Turno A","Turno B"]}

    if chart_is_h:
        fig = px.bar(
            agg, x="GapMin", y="Usuario", color="Turno", orientation="h",
            text=agg["GapMin"].round(4),
            category_orders=cat_orders
        )
    else:
        fig = px.bar(
            agg, x="Usuario", y="GapMin", color="Turno",
            text=agg["GapMin"].round(4),
            category_orders=cat_orders
        )
    fig.update_traces(texttemplate="%{text:.4f}")
    apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

# =========================================================
# [S10] Vista — Órdenes OT
# =========================================================
def view_ordenes(df_in: pd.DataFrame):
    st.subheader("📦 Órdenes OT por usuario y turno")
    cnt = df_in.groupby(["Usuario","Turno"]).size().reset_index(name="CNT")
    if cnt.empty:
        st.info("No hay datos para 'Órdenes OT'."); return

    chart_is_h = (st.session_state.get("chart_type") == "Barra horizontal")
    cat_orders = {"Turno": ["Turno A","Turno B"]}

    if chart_is_h:
        fig = px.bar(
            cnt, x="CNT", y="Usuario", color="Turno", orientation="h",
            text=cnt["CNT"].round(4),
            category_orders=cat_orders
        )
    else:
        fig = px.bar(
            cnt, x="Usuario", y="CNT", color="Turno",
            text=cnt["CNT"].round(4),
            category_orders=cat_orders
        )
    fig.update_traces(texttemplate="%{text:.4f}")
    apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

# =========================================================
# [S11] Vista — Inicio / Fin
# =========================================================
def view_inicio_fin(df_in: pd.DataFrame):
    st.subheader("🕒 Inicio y fin de turnos")
    agg = df_in.groupby(["Usuario","Turno"])["DatetimeOper"].agg(["min","max"]).reset_index()
    agg["min_h"] = agg["min"].dt.strftime("%H:%M")
    agg["max_h"] = agg["max"].dt.strftime("%H:%M")

    # Duración en horas con 4 decimales
    agg["DurH"] = (agg["max"] - agg["min"]).dt.total_seconds() / 3600.0

    # Barras agrupadas por usuario, color por turno, texto con rango
    fig = px.bar(
        agg, x="Usuario", y="DurH", color="Turno",
        text=agg.apply(lambda r: f"{r['min_h']} → {r['max_h']}", axis=1),
        category_orders={"Turno":["Turno A","Turno B"]}
    )
    fig.update_traces(texttemplate="%{text}", hovertemplate="Usuario: %{x}<br>Turno: %{legendgroup}<br>Duración: %{y:.4f} h<extra></extra>")
    fig.update_yaxes(title="Horas")
    apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

# =========================================================
# [S12] Render final
# =========================================================
render_kpis(df_f)
st.divider()
view_tm(df_f)
st.divider()
view_ordenes(df_f)
st.divider()
view_inicio_fin(df_f)
