# metrics.py
# ---------------------------------------------------------
# Cálculos y transformaciones numéricas para las vistas:
# - Fecha operativa / bandera de horas extra
# - TM ajustado por ventanas (almuerzo/combustible)
# - Conteos de OT por usuario/turno/ítem
# - Hitos de inicio / pre-alimentación / cierre
# ---------------------------------------------------------

from __future__ import annotations

from collections import Counter
from datetime import time as dtime
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from cfg import THRESH_MIN                   # umbral TM en minutos
from rules import TURNOS, LATE_B_CUTOFF, turno_by_time


# =========================
# Utilidades de tiempo
# =========================

def minutes_of_day(ts: pd.Timestamp) -> float:
    """Minutos desde 00:00 para un timestamp (sin +24h)."""
    t = ts.time()
    return t.hour * 60 + t.minute + t.second / 60.0


def fmt_hhmm(minutes: float) -> str:
    """Formatea minutos decimales -> HH:MM."""
    m = int(round(minutes))
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def minutes_for_plot(ts: pd.Timestamp, turno: str) -> float:
    """
    Minutos visibles para graficar. Para Turno B, si el evento cae antes de
    LATE_B_CUTOFF (ej. 03:00), se suma +24h para que quede a la derecha.
    """
    m = minutes_of_day(ts)
    if turno == "Turno B" and ts.time() < LATE_B_CUTOFF:
        m += 24 * 60
    return m


def most_common(values: Sequence[str]) -> str:
    vals = [x for x in values if x and str(x).strip() != ""]
    return Counter(vals).most_common(1)[0][0] if vals else "—"


# ==========================================
# Fecha operativa + bandera de horas extra
# ==========================================

def apply_oper_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve copia con columnas:
      - Turno (si faltaba)
      - IsExtra (True si Turno B y hora < LATE_B_CUTOFF)
      - FechaOper (ajustada -1 día si IsExtra)
      - DatetimeOper (combine FechaOper + Time)
    Requiere columnas: Fecha (date), Time (datetime.time), Turno (opcional).
    """
    d = df.copy()

    if "Turno" not in d.columns or d["Turno"].isna().any():
        d["Turno"] = d["Time"].apply(turno_by_time)

    extra_mask = (d["Turno"] == "Turno B") & (
        d["Time"].apply(lambda x: x is not None and x < LATE_B_CUTOFF)
    )
    d["IsExtra"] = extra_mask

    d["FechaOper"] = pd.to_datetime(d["Fecha"])
    d.loc[extra_mask, "FechaOper"] = d.loc[extra_mask, "FechaOper"] - pd.Timedelta(days=1)

    d["DatetimeOper"] = d.apply(
        lambda r: pd.Timestamp.combine(
            pd.Timestamp(r["FechaOper"]).date(), r["Time"]
        )
        if (pd.notnull(r["FechaOper"]) and r["Time"] is not None)
        else pd.NaT,
        axis=1,
    )
    return d


# =====================================================
# TM ajustado (descartando ventanas de lunch/combust.)
# =====================================================

def _subtract_window(
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Resta una ventana [win_start, win_end) de un segmento [seg_start, seg_end)."""
    if win_end <= seg_start or win_start >= seg_end:
        return [(seg_start, seg_end)]

    parts: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if seg_start < win_start:
        parts.append((seg_start, max(seg_start, win_start)))
    if seg_end > win_end:
        parts.append((min(seg_end, win_end), seg_end))
    return [(s, e) for (s, e) in parts if e > s]


def _subtract_windows(
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    date: pd.Timestamp,
    windows: Sequence[Tuple[dtime, dtime]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Resta múltiples ventanas (definidas en horas dtime) a un segmento absoluto."""
    segs = [(seg_start, seg_end)]
    for st_t, en_t in windows:
        st_w = pd.Timestamp.combine(pd.Timestamp(date), st_t)
        en_w = pd.Timestamp.combine(pd.Timestamp(date), en_t)
        new: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in segs:
            new.extend(_subtract_window(s, e, st_w, en_w))
        segs = new
        if not segs:
            break
    return segs


def compute_dead_time_rows(df_with_oper: pd.DataFrame, threshold_min: Optional[int] = None) -> pd.DataFrame:
    """
    Calcula gaps (min) entre movimientos del mismo usuario/día/turno,
    restando lunch+combustible. Devuelve filas:
      Usuario, Turno, ItemExt, AdjMin
    Requiere columnas: Usuario, FechaOper, DatetimeOper, Turno, ItemExt
    """
    thr = THRESH_MIN if threshold_min is None else int(threshold_min)

    d = (
        df_with_oper.sort_values(["Usuario", "DatetimeOper"])
        .copy()
    )
    d["prev_dt"] = d.groupby("Usuario")["DatetimeOper"].shift(1)
    d["prev_fecha"] = d.groupby("Usuario")["FechaOper"].shift(1)
    d["prev_turno"] = d.groupby("Usuario")["Turno"].shift(1)

    same_block = (
        d["prev_dt"].notna()
        & (d["FechaOper"] == d["prev_fecha"])
        & (d["Turno"] == d["prev_turno"])
    )
    d = d[same_block].copy()

    rows: List[dict] = []
    for _, r in d.iterrows():
        start = r["prev_dt"]
        end = r["DatetimeOper"]
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue

        turno = r["Turno"]
        date = r["FechaOper"]

        windows = [TURNOS["Turno A"]["lunch"], TURNOS["Turno A"]["fuel"]] if turno == "Turno A" else \
                  [TURNOS["Turno B"]["lunch"], TURNOS["Turno B"]["fuel"]]

        segs = _subtract_windows(start, end, date, windows)
        if not segs:
            continue

        adj = sum((e - s).total_seconds() / 60.0 for s, e in segs)
        if adj > thr:
            rows.append(
                {
                    "Usuario": r["Usuario"],
                    "Turno": turno,
                    "ItemExt": r.get("ItemExt"),
                    "AdjMin": adj,
                }
            )

    return pd.DataFrame(rows)


def summarize_tm_by_user_turno(dead_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa TM ajustado por Usuario/Turno/ItemExt → Min (suma).
    """
    if dead_rows.empty:
        return pd.DataFrame(columns=["Usuario", "Turno", "ItemExt", "Min"])
    g = (
        dead_rows.groupby(["Usuario", "Turno", "ItemExt"])["AdjMin"]
        .sum()
        .reset_index()
        .rename(columns={"AdjMin": "Min"})
    )
    return g


# =========================
# Conteos de Órdenes OT
# =========================

def counts_ot_by_user_turno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve conteos por Usuario/Turno/ItemExt → CNT.
    Requiere columnas: Usuario, Turno, ItemExt.
    """
    if df.empty:
        return pd.DataFrame(columns=["Usuario", "Turno", "ItemExt", "CNT"])
    cnt = (
        df.groupby(["Usuario", "Turno", "ItemExt"])
        .size()
        .reset_index(name="CNT")
    )
    return cnt


# =========================================================
# Hitos de Inicio / Alimentación / Cierre para la vista 3
# =========================================================

def build_inicio_fin_segments(df_oper: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partir de movimientos (con FechaOper/DatetimeOper/Turno/ItemRaw/IsExtra),
    calcula:
      - 'agg' agregado por Usuario/Turno (promedios o último día si solo 1)
      - 'segments' filas listas para gráfico apilado:
            UsuarioTurnoShort, Hito, Seg, Hora, Info, Item, Extra
    NOTA: La columna 'ItemExt_any' (ítem “más representativo” por hito)
          puede obtenerse previamente y venir ya en df_oper; si no existe,
          se usa ItemExt/ItemRaw si están.
    """
    d = df_oper.copy()
    # Elegir la mejor columna de ítem disponible
    item_col = (
        "ItemExt_any"
        if "ItemExt_any" in d.columns
        else "ItemExt"
        if "ItemExt" in d.columns
        else "ItemRaw"
        if "ItemRaw" in d.columns
        else None
    )
    if item_col is None:
        d["__item"] = "—"
        item_col = "__item"

    # Recorrer por día-oper y turno
    recs: List[dict] = []
    for (usr, fecha_op, turno), g in d.sort_values("DatetimeOper").groupby(
        ["Usuario", "FechaOper", "Turno"]
    ):
        if g.empty:
            continue

        g = g.copy()
        g["t_vis"] = g["DatetimeOper"].apply(lambda ts: minutes_for_plot(ts, turno))

        # Inicio (mínimo visible)
        r_ini = g.loc[g["t_vis"].idxmin()]
        t_ini_vis = float(r_ini["t_vis"])
        it_ini = r_ini[item_col]

        # Antes de alimentación (máximo visible < lunch)
        lunch_start, _ = TURNOS[turno]["lunch"]
        lunch_mins = lunch_start.hour * 60 + lunch_start.minute
        g_pre_l = g[g["t_vis"] < lunch_mins]
        if not g_pre_l.empty:
            r_al = g_pre_l.loc[g_pre_l["t_vis"].idxmax()]
            t_alim_vis = float(r_al["t_vis"])
            it_alim = r_al[item_col]
        else:
            t_alim_vis = None
            it_alim = None

        # Cierre (máximo visible del día)
        r_cie = g.loc[g["t_vis"].idxmax()]
        t_cie_vis = float(r_cie["t_vis"])
        it_cie = r_cie[item_col]

        had_extra = bool(g.get("IsExtra", pd.Series(False)).any())

        recs.append(
            {
                "Usuario": usr,
                "Turno": turno,
                "FechaOper": fecha_op,
                "t_ini": t_ini_vis,
                "t_alim": t_alim_vis,
                "t_cie": t_cie_vis,
                "it_ini": it_ini,
                "it_alim": it_alim if t_alim_vis is not None else "—",
                "it_cie": it_cie,
                "extra": had_extra,
            }
        )

    if not recs:
        return pd.DataFrame(), pd.DataFrame()

    dd = pd.DataFrame(recs)
    one_day = dd["FechaOper"].nunique() == 1

    # Agregado por usuario/turno
    agg_rows: List[dict] = []
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
            it_ini = most_common(g["it_ini"].tolist())
            it_alim = most_common(g["it_alim"].tolist())
            it_cie = most_common(g["it_cie"].tolist())
            extra_days = int(g["extra"].sum())
            modo = f"Promedio de {n_dias} días"
            extra_info = f"{extra_days} de {n_dias} días con extra"

        agg_rows.append(
            {
                "Usuario": usr,
                "Turno": turno,
                "n_dias": n_dias,
                "modo": modo,
                "t_ini": t_ini,
                "t_alim": t_alim,
                "t_cie": t_cie,
                "it_ini": it_ini,
                "it_alim": it_alim,
                "it_cie": it_cie,
                "extra_days": extra_days,
                "extra_info": extra_info,
            }
        )

    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Segments para gráfico apilado
    segments: List[dict] = []
    agg["TurnoAB"] = agg["Turno"].str.replace("Turno ", "", regex=False)
    agg["UsuarioTurnoShort"] = agg.apply(
        lambda r: f"{str(r['Usuario']).strip()}-{r['TurnoAB']}", axis=1
    )
    for _, r in agg.iterrows():
        t_ini = r["t_ini"]
        t_alim = r["t_alim"] if pd.notna(r["t_alim"]) else None
        base_for_end = t_alim if t_alim is not None else t_ini

        seg_ini = max(0.0, float(t_ini))
        seg_alim = max(0.0, float(t_alim - t_ini)) if t_alim is not None else 0.0
        seg_cie = max(0.0, float(r["t_cie"] - base_for_end))

        segments += [
            {
                "UsuarioTurnoShort": r["UsuarioTurnoShort"],
                "Hito": "Inicio",
                "Seg": seg_ini,
                "Hora": fmt_hhmm(t_ini),
                "Info": r["modo"],
                "Item": r["it_ini"],
                "Extra": r["extra_info"],
            },
            {
                "UsuarioTurnoShort": r["UsuarioTurnoShort"],
                "Hito": "Antes de alimentación",
                "Seg": seg_alim,
                "Hora": fmt_hhmm(t_alim) if t_alim is not None else "—",
                "Info": r["modo"],
                "Item": r["it_alim"] if t_alim is not None else "—",
                "Extra": r["extra_info"],
            },
            {
                "UsuarioTurnoShort": r["UsuarioTurnoShort"],
                "Hito": "Antes de cierre",
                "Seg": seg_cie,
                "Hora": fmt_hhmm(r["t_cie"]),
                "Info": r["modo"],
                "Item": r["it_cie"],
                "Extra": r["extra_info"],
            },
        ]

    segments_df = pd.DataFrame(segments)
    return agg, segments_df
