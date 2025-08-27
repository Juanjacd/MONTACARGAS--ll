# ========================= data_io.py =========================
# Lectura de Excel, persistencia en SQLite y utilidades de fecha operativa
# =============================================================

from __future__ import annotations
import pandas as pd
import sqlite3
import hashlib
import re
from datetime import time as dtime
from typing import Optional, List, Tuple

from cfg import TURNOS, LATE_B_CUTOFF

TABLE = "ordenes"

# ---------------------- Normalización básica ----------------------

def _norm_nfd_ascii(s: str) -> str:
    import unicodedata
    if s is None: return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()

def _norm_compact(s: str) -> str:
    s = _norm_nfd_ascii(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def pick_col(cols_map: dict, *aliases) -> Optional[str]:
    """Encuentra en df.columns (map normalizado -> original) la mejor coincidencia para cualquier alias."""
    def normkey(x): return _norm_compact(x)
    norm2orig = {normkey(k): v for k, v in cols_map.items()}
    for alias in aliases:
        key = normkey(alias)
        for nk, orig in norm2orig.items():
            if key in nk or nk in key:
                return orig
    return None

# ---------------------- Parsing de hora ----------------------

def to_time(x) -> Optional[dtime]:
    """Convierte varios formatos (HH:MM, Excel serial, Timestamp) a datetime.time."""
    if pd.isna(x):
        return None
    try:
        # pandas/py datetime-like
        if hasattr(x, "hour"):
            return dtime(int(x.hour), int(getattr(x, "minute", 0)), int(getattr(x, "second", 0)))
    except Exception:
        pass
    # Excel serial (número)
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            dtv = pd.to_datetime(x, unit="d", origin="1899-12-30")
            return dtime(int(dtv.hour), int(dtv.minute), int(dtv.second))
        except Exception:
            pass
    # Texto
    s = str(x).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            dtv = pd.to_datetime(s, format=fmt)
            return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv, "second", 0)))
        except Exception:
            continue
    # Parse laxo
    dtv = pd.to_datetime(s, errors="coerce")
    if pd.notnull(dtv):
        return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv, "second", 0)))
    return None

# ---------------------- Turno A/B por hora ----------------------

def turno_by_time(t: Optional[dtime]) -> Optional[str]:
    if t is None:
        return None
    a_start, a_end = TURNOS["Turno A"]["start"], TURNOS["Turno A"]["end"]
    b_start, b_end = TURNOS["Turno B"]["start"], TURNOS["Turno B"]["end"]
    if a_start <= t < a_end:
        return "Turno A"
    if b_start <= t <= b_end:
        return "Turno B"
    # 22:55–23:59 y 00:00–02:59 cuentan como B
    if t < LATE_B_CUTOFF or t > b_end:
        return "Turno B"
    return None

# ---------------------- SQLite: DDL/DML ----------------------

def ensure_db(path: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            id TEXT PRIMARY KEY,
            usuario TEXT, fecha TEXT, time TEXT, turno TEXT, datetime TEXT,
            orden TEXT, ubic_proced TEXT, ubic_destino TEXT, itemraw TEXT
        )
    """)
    # Garantiza columnas requeridas si el esquema cambió
    cur.execute(f"PRAGMA table_info({TABLE})")
    existing_cols = {row[1].lower() for row in cur.fetchall()}
    required = ["id","usuario","fecha","time","turno","datetime","orden","ubic_proced","ubic_destino","itemraw"]
    for col in required:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {col} TEXT")
    con.commit()
    con.close()

def clear_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.execute(f"DELETE FROM {TABLE}")
    con.commit()
    con.close()

def _make_uid(row: pd.Series) -> str:
    dtv = pd.to_datetime(row.get("Datetime"), errors="coerce")
    dtv = pd.NaT if pd.isna(dtv) else dtv.floor("min")
    base = f"{row.get('Usuario','')}|{row.get('Fecha','')}|{str(dtv)}|{str(row.get('Orden') or '')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def upsert_df(path: str, df: pd.DataFrame) -> int:
    """Inserta/actualiza filas en SQLite; devuelve # de filas procesadas."""
    if df.empty:
        return 0
    con = sqlite3.connect(path)
    cur = con.cursor()
    df2 = df.copy()
    df2["Datetime"] = pd.to_datetime(df2["Datetime"]).dt.floor("min")
    df2["id"] = df2.apply(_make_uid, axis=1)

    rows: List[Tuple] = []
    for _, r in df2.iterrows():
        rows.append((
            r["id"],
            r.get("Usuario"),
            str(r.get("Fecha")) if pd.notnull(r.get("Fecha")) else None,
            str(r.get("Time")) if pd.notnull(r.get("Time")) else None,
            r.get("Turno"),
            r.get("Datetime").isoformat() if pd.notnull(r["Datetime"]) else None,
            str(r.get("Orden") or None),
            r.get("Ubic.proced"),
            r.get("Ubicación de destino"),
            r.get("ItemRaw") if "ItemRaw" in df2.columns else None
        ))
    cur.executemany(
        f"""INSERT OR REPLACE INTO {TABLE}
            (id,usuario,fecha,time,turno,datetime,orden,ubic_proced,ubic_destino,itemraw)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows
    )
    con.commit()
    con.close()
    return len(rows)

def read_all(path: str) -> pd.DataFrame:
    """Lee todo el histórico y normaliza nombres de columnas."""
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    except Exception:
        df = pd.DataFrame(columns=[
            "id","usuario","fecha","time","turno","datetime",
            "orden","ubic_proced","ubic_destino","itemraw"
        ])
    con.close()
    if df.empty:
        return df

    df = df.rename(columns={
        "usuario":"Usuario",
        "fecha":"Fecha",
        "time":"TimeStr",
        "turno":"Turno",
        "datetime":"Datetime",
        "orden":"Orden",
        "ubic_proced":"Ubic.proced",
        "ubic_destino":"Ubicación de destino",
        "itemraw":"ItemRaw"
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.date
    # Reconstruir Time desde Datetime cuando sea posible
    if "Time" not in df.columns or df["Datetime"].notna().any():
        df["Time"] = df["Datetime"].dt.time
    return df[[
        "Usuario","Fecha","Time","Turno","Datetime","Orden",
        "Ubic.proced","Ubicación de destino","ItemRaw"
    ]]

# ---------------------- Excel -> DataFrame ----------------------

def load_excel(file, sheet_name: str = "Hoja1") -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas normalizadas:
    ['Usuario','Fecha','Hora','Time','Turno','Datetime','Orden',
     'Ubic.proced','Ubicación de destino','ItemRaw']
    """
    xls = pd.ExcelFile(file)
    if sheet_name not in xls.sheet_names:
        sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Mapa normalizado para búsqueda flexible
    cols = {c.lower().strip(): c for c in df.columns}

    col_usuario = pick_col(cols, "usuario")
    col_fecha   = pick_col(cols, "fecha", "fecha confirmacion")
    col_hora    = pick_col(cols, "hora", "hora confirmacion")
    col_orden   = pick_col(cols, "orden", "numero de orden de transporte", "ot")
    col_uproc   = pick_col(cols, "ubic.proced", "ubic proced", "ubic procedencia", "ubicacion procedencia", "origen", "ubic origen")
    col_udest   = pick_col(cols, "ubicacion de destino", "ubic destino", "destino", "ubic. destino")
    col_item    = pick_col(cols, "item","ítem","movimiento","tipo movimiento","tipo_movimiento",
                           "actividad","proceso","tarea","categoria","clase","operacion",
                           "detalle movimiento","detalle del movimiento","mov","nombre movimiento")

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
    out["Ubic.proced"] = df[col_uproc].astype(str).str.strip() if col_uproc else None
    out["Ubicación de destino"] = df[col_udest].astype(str).str.strip() if col_udest else None
    out["ItemRaw"] = df[col_item].astype(str).str.strip() if col_item else None

    out["Turno"] = out["Time"].apply(turno_by_time)
    out["Datetime"] = out.apply(
        lambda r: pd.Timestamp.combine(r["Fecha"], r["Time"]) if (pd.notnull(r["Fecha"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    out = out.dropna(subset=["Fecha","Time","Datetime"])
    return out[[
        "Usuario","Fecha","Hora","Time","Turno","Datetime",
        "Orden","Ubic.proced","Ubicación de destino","ItemRaw"
    ]]

# ---------------------- Fecha operativa / horas extra ----------------------

def apply_oper_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade FechaOper y DatetimeOper. Las 00:00–02:59 de Turno B
    se asignan al día operativo anterior y se marcan como 'IsExtra'=True.
    Si 'Turno' no existe o tiene NaN, se infiere por la hora.
    """
    d = df.copy()
    if "Turno" not in d.columns or d["Turno"].isna().any():
        d["Turno"] = d["Time"].apply(turno_by_time)

    # Marca horas extra: Turno B entre 00:00 y 02:59
    extra_mask = (d["Turno"] == "Turno B") & (d["Time"].apply(lambda x: x is not None and x < LATE_B_CUTOFF))
    d["IsExtra"] = extra_mask

    # Fecha operativa (mueve 00:00–02:59 del B al día anterior)
    d["FechaOper"] = pd.to_datetime(d["Fecha"])
    d.loc[extra_mask, "FechaOper"] = d.loc[extra_mask, "FechaOper"] - pd.Timedelta(days=1)

    d["DatetimeOper"] = d.apply(
        lambda r: pd.Timestamp.combine(pd.Timestamp(r["FechaOper"]).date(), r["Time"])
        if (pd.notnull(r["FechaOper"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    return d

# ---------------------- Exports ----------------------

__all__ = [
    "ensure_db", "clear_db", "upsert_df", "read_all",
    "load_excel", "apply_oper_day",
    "to_time", "turno_by_time",
    "TABLE"
]
