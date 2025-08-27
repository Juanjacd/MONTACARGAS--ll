# widgets.py
# ---------------------------------------------------------
# Componentes UI reutilizables (Streamlit).
# No hace lecturas/escrituras — solo widgets y helpers.
# ---------------------------------------------------------
from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Iterable, Tuple


# =========================
# Header / título
# =========================
def render_hero(title: str) -> None:
    """
    Título responsivo dentro de una tarjeta.
    style.py define variables CSS (var(--panel), var(--ink), etc.).
    """
    st.markdown(
        f"""
<div style="
  border:1px solid var(--border);background:var(--panel);
  border-radius:14px;padding:14px 16px;margin-bottom:12px;">
  <h1 style="
    margin:0 0 6px 0;
    font-weight:800;
    line-height:1.15;
    color:var(--ink);
    /* Responsivo: entre 22px y 38px según ancho */
    font-size:clamp(22px, 2.2vw + 16px, 38px);
  ">{title}</h1>
</div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Tema / preferencias
# =========================
def sidebar_theme_toggle(*, default: bool = False, key: str = "ui_dark") -> bool:
    """Switch de modo oscuro en el sidebar."""
    st.markdown("---")
    dark = st.checkbox("🌙 Modo oscuro", value=default, key=key)
    return bool(dark)


def sidebar_preferences(
    *,
    palettes: Dict[str, Dict[str, str]],
    default_palette: str,
    default_chart_type: str = "Barra horizontal",
    key_ns: str = "prefs",
) -> Tuple[str, str]:
    """
    Selector de paleta y orientación de barra.
    Devuelve: (pal_name, chart_type)
    """
    with st.expander("⚙️ Preferencias", expanded=False):
        chart_type = st.selectbox(
            "Orientación (Gráfica TM)",
            ["Barra horizontal", "Barra vertical"],
            index=(0 if default_chart_type == "Barra horizontal" else 1),
            key=f"{key_ns}_chart_type",
        )
        pal_name = st.selectbox(
            "🎨 Paleta",
            list(palettes.keys()),
            index=list(palettes.keys()).index(default_palette)
            if default_palette in palettes
            else 0,
            key=f"{key_ns}_palette",
        )

    # Guarda en sesión por comodidad de otras capas
    st.session_state["chart_type"] = chart_type
    st.session_state["pal_name"] = pal_name
    return pal_name, chart_type


# =========================
# Filtros
# =========================
def sidebar_filters(
    df: pd.DataFrame,
    *,
    avail_items: Optional[List[str]] = None,
    default_items: Optional[Iterable[str]] = None,
    key_ns: str = "filters",
) -> Dict[str, object]:
    """
    Renderiza selectores de Usuario, Turno, Rango de fechas e Ítems.
    Devuelve dict con:
      users: List[str]
      turns: List[str]
      date_range: Tuple[date, date]
      items: List[str]
    """
    users_all = sorted(df["Usuario"].dropna().unique().tolist())
    turns_all = ["Turno A", "Turno B"]
    fmin = df["FechaOper"].min().date()
    fmax = df["FechaOper"].max().date()

    sel_users = st.multiselect(
        "Usuarios", users_all, default=[], key=f"{key_ns}_users"
    )
    sel_turns = st.multiselect(
        "Turnos", turns_all, default=[], key=f"{key_ns}_turns"
    )
    sel_range = st.date_input(
        "Rango de fechas", (fmin, fmax), key=f"{key_ns}_range"
    )

    # Ítems disponibles a partir del DF, si no se pasan explícitos
    if avail_items is None:
        avail_items = sorted([x for x in df["ItemExt"].dropna().unique().tolist()])

    default_items = list(default_items) if default_items is not None else []
    sel_items = st.multiselect(
        "Ítems", avail_items, default=default_items, key=f"{key_ns}_items"
    )

    return dict(
        users=sel_users,
        turns=sel_turns,
        date_range=sel_range,
        items=sel_items,
    )


# =========================
# Descarga CSV
# =========================
def sidebar_download_csv(
    df: pd.DataFrame,
    *,
    filename: str = "filtrado_montacargas.csv",
    label: str = "⬇️ Descargar filtrado (CSV)",
    key: str = "dl_csv",
) -> None:
    st.markdown("---")
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# =========================
# Helpers puentes
# =========================
def color_map_from(palettes: Dict[str, Dict[str, str]], pal_name: str) -> Dict[str, str]:
    """Obtiene el mapa de color de la paleta actual con fallback seguro."""
    return palettes.get(pal_name) or next(iter(palettes.values()))


def date_range_to_ts(sel_range, fmin, fmax) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Convierte el rango de date_input a timestamps (inicio-cierre del día).
    """
    if isinstance(sel_range, (list, tuple)) and len(sel_range) == 2:
        start = pd.Timestamp(sel_range[0])
        end = pd.Timestamp(sel_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start = pd.Timestamp(fmin)
        end = pd.Timestamp(fmax) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return start, end
