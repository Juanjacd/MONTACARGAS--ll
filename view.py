# view.py
# ---------------------------------------------------------
# Render de vistas con Streamlit + Plotly (sin I/O).
# Depende de:
#   - style.apply_plot_theme(fig, dark)
#   - rules.TURNOS, rules.LATE_B_CUTOFF
#   - classify.canon_item_from_text, classify.item_ext
# ---------------------------------------------------------

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional, Dict, Iterable

from style import apply_plot_theme
from rules import TURNOS, LATE_B_CUTOFF
from classify import canon_item_from_text, item_ext


# =========================
# Helpers de presentación
# =========================
def render_section_title(txt: str) -> None:
    st.markdown(
        f'<div style="border:1px solid var(--border);'
        f'background:var(--panel);border-radius:12px;'
        f'padding:10px 12px;margin:14px 0 8px 0">'
        f'<h2 style="margin:0;font-weight:700;font-size:18px;'
        f'color:var(--ink)">{txt}</h2></div>',
        unsafe_allow_html=True,
    )


def _short_label(user: str, turno_ab: str) -> str:
    return f"{str(user).strip()}-{turno_ab}"


def _responsive_bar_style(fig, n_categories: int) -> None:
    if n_categories <= 6:
        bargap, bgrp = 0.12, 0.05
    elif n_categories <= 12:
        bargap, bgrp = 0.10, 0.04
    elif n_categories <= 24:
        bargap, bgrp = 0.08, 0.03
    else:
        bargap, bgrp = 0.06, 0.02
    fig.update_layout(bargap=bargap, bargroupgap=bgrp)


def _bold_tickfont(size: int = 12) -> dict:
    # No hay "weight" para ticks en Plotly; usar una familia bold
    return dict(size=size, family="Arial Black, DejaVu Sans, Segoe UI, sans-serif")


# =========================
# KPIs
# =========================
def render_kpis(df_filtered: pd.DataFrame) -> None:
    total = len(df_filtered)
    uniq_users = df_filtered["Usuario"].nunique()
    tA = int((df_filtered["Turno"] == "Turno A").sum())
    tB = int((df_filtered["Turno"] == "Turno B").sum())

    st.markdown(
        f"""
<div style="border:1px solid var(--border);background:var(--panel);
           border-radius:14px;padding:12px 14px;margin-left:10px;color:var(--ink);
           max-width:420px">
  <div style="display:flex;align-items:center;gap:8px;font-weight:800;
              font-size:20px;margin:2px 0 12px 0">📊 Indicadores</div>
  <div style="display:grid;grid-template-columns:1fr;gap:12px">
    <div><div style="color:var(--muted);font-size:13px">Órdenes (filtrado)</div>
         <div style="font-size:32px;font-weight:800">{total:,}</div></div>
    <div><div style="color:var(--muted);font-size:13px">Usuarios únicos</div>
         <div style="font-size:32px;font-weight:800">{uniq_users}</div></div>
    <div><div style="color:var(--muted);font-size:13px">Turno A</div>
         <div style="font-size:32px;font-weight:800">{tA:,}</div></div>
    <div><div style="color:var(--muted);font-size:13px">Turno B</div>
         <div style="font-size:32px;font-weight:800">{tB:,}</div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Vista 1 — TM por usuario/turno
# =========================
def view_tm_por_usuario_turno(
    df_f: pd.DataFrame,
    *,
    dark: bool,
    chart_type: str,
    color_map: Dict[str, str],
    sel_items: Optional[Iterable[str]] = None,
    avail_items: Optional[List[str]] = None,
    threshold_min: int = 15,
) -> None:
    """
    df_f: DataFrame filtrado final con columnas:
        Usuario, Turno, FechaOper, DatetimeOper, ItemExt
    """
    render_section_title(
        "Tiempo Muerto — dos barras por Usuario (Turno A y B), apilado por ítem"
    )

    # Calcular gaps entre registros del mismo día/turno por usuario
    df_g = df_f.sort_values(["Usuario", "DatetimeOper"]).copy()
    df_g["prev_dt"] = df_g.groupby("Usuario")["DatetimeOper"].shift(1)
    df_g["prev_fecha"] = df_g.groupby("Usuario")["FechaOper"].shift(1)
    df_g["prev_turno"] = df_g.groupby("Usuario")["Turno"].shift(1)
    same = (
        df_g["prev_dt"].notna()
        & (df_g["FechaOper"] == df_g["prev_fecha"])
        & (df_g["Turno"] == df_g["prev_turno"])
    )
    df_g = df_g[same].copy()

    exc = [
        TURNOS["Turno A"]["lunch"],
        TURNOS["Turno A"]["fuel"],
        TURNOS["Turno B"]["lunch"],
        TURNOS["Turno B"]["fuel"],
    ]

    def _subtract_window(seg_start, seg_end, win_start, win_end):
        if win_end <= seg_start or win_start >= seg_end:
            return [(seg_start, seg_end)]
        parts = []
        if seg_start < win_start:
            parts.append((seg_start, max(seg_start, win_start)))
        if seg_end > win_end:
            parts.append((min(seg_end, win_end), seg_end))
        return [(s, e) for (s, e) in parts if e > s]

    def _subtract_windows(seg_start, seg_end, date, windows):
        segs = [(seg_start, seg_end)]
        for st_t, en_t in windows:
            st_w = pd.Timestamp.combine(pd.Timestamp(date), st_t)
            en_w = pd.Timestamp.combine(pd.Timestamp(date), en_t)
            new = []
            for s, e in segs:
                new.extend(_subtract_window(s, e, st_w, en_w))
            segs = new
            if not segs:
                break
        return segs

    rows = []
    for _, r in df_g.iterrows():
        start = r["prev_dt"]
        end = r["DatetimeOper"]
        date = r["FechaOper"]
        if pd.isna(start) or pd.isna(end) or pd.isna(date):
            continue
        segs = _subtract_windows(start, end, date, exc)
        if not segs:
            continue
        adj = sum((e - s).total_seconds() / 60.0 for s, e in segs)
        if adj > threshold_min:
            rows.append(
                {
                    "Usuario": r["Usuario"],
                    "Turno": r["Turno"],
                    "ItemExt": r.get("ItemExt"),
                    "AdjMin": adj,
                }
            )

    dead_ext = pd.DataFrame(rows)
    if dead_ext.empty:
        st.info("No se detectó TM > 15 min con el filtro actual.")
        return

    tm_ut = (
        dead_ext.groupby(["Usuario", "Turno", "ItemExt"])["AdjMin"].sum().reset_index()
    )
    if sel_items:
        tm_ut = tm_ut[tm_ut["ItemExt"].isin(list(sel_items))]

    g = tm_ut.copy()
    g["TurnoAB"] = g["Turno"].str.replace("Turno ", "", regex=False)
    g["UsuarioTurnoShort"] = g.apply(
        lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1
    )

    order_users = (
        g.groupby("Usuario")["AdjMin"].sum().sort_values(ascending=False).index.tolist()
    )
    order_axis, present_keys = [], set(g["UsuarioTurnoShort"])
    for u in order_users:
        for ab in ["A", "B"]:
            key = _short_label(u, ab)
            if key in present_keys:
                order_axis.append(key)

    g = g.rename(columns={"AdjMin": "Min"})
    hover_tmpl_h = (
        "Ítem: %{customdata[0]}<br>Minutos TM: %{customdata[1]:.0f}m<extra></extra>"
    )

    is_horizontal = (chart_type or "Barra horizontal") == "Barra horizontal"
    if is_horizontal:
        height = max(320, 22 * len(order_axis) + 110)
        fig = px.bar(
            g,
            x="Min",
            y="UsuarioTurnoShort",
            color="ItemExt",
            orientation="h",
            barmode="stack",
            category_orders={
                "UsuarioTurnoShort": order_axis,
                "ItemExt": (list(sel_items) if sel_items else (avail_items or [])),
            },
            color_discrete_map=color_map,
            custom_data=["ItemExt", "Min"],
            height=height,
        )
        fig.update_traces(hovertemplate=hover_tmpl_h, marker_line_width=0, opacity=0.95)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=order_axis,
            tickfont=_bold_tickfont(12),  # “negrilla”
        )

        totals = g.groupby("UsuarioTurnoShort")["Min"].sum().reindex(order_axis)
        fig.add_trace(
            go.Scatter(
                x=totals.values,
                y=totals.index.tolist(),
                mode="text",
                text=[f"{v:.0f} min" for v in totals.values],
                textposition="middle right",
                textfont=dict(size=12),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        xmax = max(1, float(totals.max()))
        fig.update_xaxes(range=[0, xmax * 1.06], tickfont=_bold_tickfont(12))
        _responsive_bar_style(fig, len(order_axis))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=110), legend_title_text="Ítem")
    else:
        fig = px.bar(
            g,
            x="UsuarioTurnoShort",
            y="Min",
            color="ItemExt",
            barmode="stack",
            category_orders={
                "UsuarioTurnoShort": order_axis,
                "ItemExt": (list(sel_items) if sel_items else (avail_items or [])),
            },
            color_discrete_map=color_map,
            custom_data=["ItemExt", "Min"],
            height=420,
        )
        fig.update_traces(hovertemplate=hover_tmpl_h, marker_line_width=0, opacity=0.95)
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=order_axis,
            tickangle=-30,
            tickfont=_bold_tickfont(11),  # “negrilla”
        )
        totals = g.groupby("UsuarioTurnoShort")["Min"].sum().reindex(order_axis)
        ymax = float(totals.max()) * 1.12
        fig.update_yaxes(range=[0, ymax])
        fig.add_trace(
            go.Bar(
                x=totals.index.tolist(),
                y=totals.values,
                marker_color="rgba(0,0,0,0)",
                showlegend=False,
                hoverinfo="skip",
                text=[f"{v:.0f} min" for v in totals.values],
                textposition="outside",
                textfont=dict(size=11),
            )
        )
        _responsive_bar_style(fig, len(order_axis))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Ítem")

    apply_plot_theme(fig, dark)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Vista 2 — Órdenes OT
# =========================
def _labels_over_bars_consistent(fig, totals: pd.Series) -> None:
    """
    Coloca anotaciones encima de cada barra, con tamaño y desplazamiento coherentes.
    """
    if totals.empty:
        return

    max_val = float(totals.max())
    # Ajuste superior para garantizar espacio homogéneo
    max_digits = len(str(int(max_val)))
    top_pad = 0.12 + 0.015 * max(0, max_digits - 3)  # más dígitos, más espacio
    y_max = max_val * (1 + top_pad)
    fig.update_yaxes(range=[0, y_max], automargin=True)

    # Tamaño de fuente en función de los dígitos del número de cada barra
    annotations = []
    for x_val, y_val in totals.items():
        d = len(str(int(y_val))) if y_val > 0 else 1
        lab_size = max(11, 16 - max(0, d - 3))  # entre 11 y ~16
        yshift = 6 + int(lab_size * 0.9)
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                xref="x",
                yref="y",
                text=f"{int(y_val):,}".replace(",", "."),
                showarrow=False,
                yanchor="bottom",
                yshift=yshift,
                font=dict(size=lab_size),
                align="center",
            )
        )
    prev = list(fig.layout.annotations) if fig.layout.annotations else []
    fig.update_layout(annotations=prev + annotations)


def view_ordenes_ot(
    df_f: pd.DataFrame,
    *,
    dark: bool,
    color_map: Dict[str, str],
    sel_items: Optional[Iterable[str]] = None,
    avail_items: Optional[List[str]] = None,
) -> None:
    render_section_title("Órdenes OT — total de movimientos por usuario y turno")

    cnt = df_f.groupby(["Usuario", "Turno", "ItemExt"]).size().reset_index(name="CNT")
    if sel_items:
        cnt = cnt[cnt["ItemExt"].isin(list(sel_items))]
    if cnt.empty:
        st.info("No hay órdenes en el filtro actual para 'Órdenes OT'.")
        return

    cnt["TurnoAB"] = cnt["Turno"].str.replace("Turno ", "", regex=False)
    cnt["UsuarioTurnoShort"] = cnt.apply(
        lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1
    )

    order_users = (
        cnt.groupby("Usuario")["CNT"].sum().sort_values(ascending=False).index.tolist()
    )
    order_axis, present_keys = [], set(cnt["UsuarioTurnoShort"])
    for u in order_users:
        for ab in ["A", "B"]:
            k = _short_label(u, ab)
            if k in present_keys:
                order_axis.append(k)

    hover_tmpl = (
        "Ítem: %{customdata[0]}<br>Órdenes: %{customdata[1]:.0f}"
        "<br>%{customdata[2]}<extra></extra>"
    )

    fig = px.bar(
        cnt,
        x="UsuarioTurnoShort",
        y="CNT",
        color="ItemExt",
        barmode="stack",
        category_orders={
            "UsuarioTurnoShort": order_axis,
            "ItemExt": (list(sel_items) if sel_items else (avail_items or [])),
        },
        color_discrete_map=color_map,
        custom_data=["ItemExt", "CNT", "UsuarioTurnoShort"],
        height=520,
    )
    fig.update_traces(hovertemplate=hover_tmpl, marker_line_width=0, opacity=0.95)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order_axis,
        tickangle=-30,
        tickfont=_bold_tickfont(11),  # “negrilla”
    )

    totals = cnt.groupby("UsuarioTurnoShort")["CNT"].sum().reindex(order_axis)
    _labels_over_bars_consistent(fig, totals)

    _responsive_bar_style(fig, len(order_axis))
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=80), legend_title_text="Ítem")
    apply_plot_theme(fig, dark)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        render_kpis(df_f)


# =========================
# Vista 3 — Inicio/Fin
# =========================
def _minutes_of_day(ts: pd.Timestamp) -> float:
    t = ts.time()
    return t.hour * 60 + t.minute + t.second / 60.0


def _fmt_hhmm(minutes: float) -> str:
    m = int(round(minutes))
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def _minutes_for_plot(ts: pd.Timestamp, turno: str) -> float:
    m = _minutes_of_day(ts)
    if turno == "Turno B" and ts.time() < LATE_B_CUTOFF:
        m += 24 * 60
    return m


def _most_common(lst: List[str]) -> str:
    lst = [x for x in lst if x and str(x).strip() != ""]
    return pd.Series(lst).mode().iloc[0] if lst else "—"


def _classify_any_row(row) -> str:
    raw = canon_item_from_text(row.get("ItemRaw"))
    if raw:
        return raw
    base = item_ext(row.get("Ubic.proced"), row.get("Ubicación de destino"))
    return base if base else "—"


def view_inicio_fin_turno(df_pre: pd.DataFrame, *, dark: bool) -> None:
    render_section_title(
        "Inicio y fin de turno — hora por Usuario/Turno (promedio o real)"
    )

    d = df_pre.copy()
    if "ItemExt_any" not in d.columns:
        d["ItemExt_any"] = d.apply(_classify_any_row, axis=1)

    recs = []
    for (usr, fecha_op, turno), g in d.sort_values("DatetimeOper").groupby(
        ["Usuario", "FechaOper", "Turno"]
    ):
        if g.empty:
            continue
        g = g.copy()
        g["t_vis"] = g["DatetimeOper"].apply(lambda ts: _minutes_for_plot(ts, turno))
        r_ini = g.loc[g["t_vis"].idxmin()]
        t_ini_vis = float(r_ini["t_vis"])
        it_ini = r_ini["ItemExt_any"]

        lunch_start, _ = TURNOS[turno]["lunch"]
        lunch_mins = lunch_start.hour * 60 + lunch_start.minute
        g_pre_l = g[g["t_vis"] < lunch_mins]
        if not g_pre_l.empty:
            r_al = g_pre_l.loc[g_pre_l["t_vis"].idxmax()]
            t_alim_vis = float(r_al["t_vis"])
            it_alim = r_al["ItemExt_any"]
        else:
            t_alim_vis, it_alim = None, None

        r_cie = g.loc[g["t_vis"].idxmax()]
        t_cie_vis = float(r_cie["t_vis"])
        it_cie = r_cie["ItemExt_any"]

        had_extra = bool(g["IsExtra"].any())
        recs.append(
            dict(
                Usuario=usr,
                Turno=turno,
                FechaOper=fecha_op,
                t_ini=t_ini_vis,
                t_alim=t_alim_vis,
                t_cie=t_cie_vis,
                it_ini=it_ini,
                it_alim=(it_alim if t_alim_vis is not None else "—"),
                it_cie=it_cie,
                extra=had_extra,
            )
        )
    if not recs:
        st.info("No se pudieron calcular hitos con el filtro actual.")
        return

    dd = pd.DataFrame(recs)
    one_day = dd["FechaOper"].nunique() == 1

    agg_rows = []
    for (usr, turno), g in dd.groupby(["Usuario", "Turno"]):
        if one_day:
            r = g.iloc[-1]
            n_dias = 1
            t_ini, t_alim, t_cie = r["t_ini"], r["t_alim"], r["t_cie"]
            it_ini, it_alim, it_cie = r["it_ini"], r["it_alim"], r["it_cie"]
            modo = "Día único"
            extra_days = int(bool(r["extra"]))
            extra_info = "Sí" if bool(r["extra"]) else "No"
        else:
            n_dias = g["FechaOper"].nunique()
            t_ini = g["t_ini"].mean(skipna=True)
            t_alim = g["t_alim"].mean(skipna=True)
            t_cie = g["t_cie"].mean(skipna=True)
            it_ini = _most_common(g["it_ini"].tolist())
            it_alim = _most_common(g["it_alim"].tolist())
            it_cie = _most_common(g["it_cie"].tolist())
            modo = f"Promedio de {n_dias} días"
            extra_days = int(g["extra"].sum())
            extra_info = f"{extra_days} de {n_dias} días con extra"

        agg_rows.append(
            dict(
                Usuario=usr,
                Turno=turno,
                n_dias=n_dias,
                modo=modo,
                t_ini=t_ini,
                t_alim=t_alim,
                t_cie=t_cie,
                it_ini=it_ini,
                it_alim=it_alim,
                it_cie=it_cie,
                extra_days=extra_days,
                extra_info=extra_info,
            )
        )
    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        st.info("No hay agregaciones para mostrar.")
        return

    agg["TurnoAB"] = agg["Turno"].str.replace("Turno ", "", regex=False)
    agg["UsuarioTurnoShort"] = agg.apply(
        lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1
    )
    order_axis = sorted(agg["UsuarioTurnoShort"].unique().tolist())

    rows = []
    for _, r in agg.iterrows():
        t_ini = r["t_ini"]
        t_alim = r["t_alim"] if pd.notna(r["t_alim"]) else None
        base_for_end = t_alim if t_alim is not None else t_ini
        seg_ini = max(0.0, float(t_ini))
        seg_alim = max(0.0, float(t_alim - t_ini)) if t_alim is not None else 0.0
        seg_cie = max(0.0, float(r["t_cie"] - base_for_end))

        rows += [
            dict(
                UsuarioTurnoShort=r["UsuarioTurnoShort"],
                Hito="Inicio",
                Seg=seg_ini,
                Hora=_fmt_hhmm(t_ini),
                Info=r["modo"],
                Item=r["it_ini"],
                Extra=r["extra_info"],
            ),
            dict(
                UsuarioTurnoShort=r["UsuarioTurnoShort"],
                Hito="Antes de alimentación",
                Seg=seg_alim,
                Hora=_fmt_hhmm(t_alim) if t_alim is not None else "—",
                Info=r["modo"],
                Item=(r["it_alim"] if t_alim is not None else "—"),
                Extra=r["extra_info"],
            ),
            dict(
                UsuarioTurnoShort=r["UsuarioTurnoShort"],
                Hito="Antes de cierre",
                Seg=seg_cie,
                Hora=_fmt_hhmm(r["t_cie"]),
                Info=r["modo"],
                Item=r["it_cie"],
                Extra=r["extra_info"],
            ),
        ]
    m = pd.DataFrame(rows)
    top_per_bar = m.groupby("UsuarioTurnoShort")["Seg"].sum().reindex(order_axis)

    has_any_extra = (agg["extra_days"] > 0).any()
    base_max = 27 * 60 if has_any_extra else 24 * 60
    y_max = max(base_max, int(top_per_bar.max() // 60 + 2) * 60)
    ticks = list(range(0, y_max + 1, 60))
    ticktext = [_fmt_hhmm(t) for t in ticks]

    hover_tmpl = (
        "Hito: %{customdata[0]}<br>Hora: %{customdata[1]}<br>%{customdata[2]}"
        "<br>Ítem más común: %{customdata[3]}<br>Horas extra: %{customdata[4]}<extra></extra>"
    )

    fig = px.bar(
        m,
        x="UsuarioTurnoShort",
        y="Seg",
        color="Hito",
        barmode="stack",
        category_orders={
            "UsuarioTurnoShort": order_axis,
            "Hito": ["Inicio", "Antes de alimentación", "Antes de cierre"],
        },
        color_discrete_map={
            "Inicio": "#1F77B4",
            "Antes de alimentación": "#E4572E",
            "Antes de cierre": "#2CA02C",
        },
        custom_data=["Hito", "Hora", "Info", "Item", "Extra"],
        height=480,
    )
    fig.update_traces(hovertemplate=hover_tmpl, marker_line_width=0, opacity=0.96)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order_axis,
        tickangle=-30,
        tickfont=_bold_tickfont(11),  # “negrilla”
    )
    fig.update_yaxes(tickvals=ticks, ticktext=ticktext, title="Hora del día (HH:MM)")
    fig.add_trace(
        go.Scatter(
            x=top_per_bar.index.tolist(),
            y=top_per_bar.values,
            mode="text",
            text=[_fmt_hhmm(v) for v in top_per_bar.values],
            textposition="top center",
            textfont=dict(size=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    _responsive_bar_style(fig, len(order_axis))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=140), legend_title_text="Hito")
    apply_plot_theme(fig, dark)
    st.plotly_chart(fig, use_container_width=True)
