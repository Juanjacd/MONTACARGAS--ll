# cfg.py — constantes y parámetros de la app (Python 3.9 compatible)
from datetime import time as dtime
from typing import Optional, Dict

# --- App ---
APP_TITLE   = "Análisis de Tiempo Muerto — Montacargas"
APP_TAGLINE = "TM + Órdenes OT + Inicio/Fin (auto horas extra)"
DB_DEFAULT_PATH = "montacargas.db"
DEFAULT_SHEET   = "Hoja1"

# --- Reglas de negocio / horarios ---
THRESH_MIN = 15  # minutos mínimos para considerar TM
TURNOS = {
    "Turno A": {"start": dtime(5, 0),  "end": dtime(13, 55),
                "lunch": (dtime(8, 20), dtime(9, 0)),
                "fuel":  (dtime(9, 0),  dtime(9, 18))},
    "Turno B": {"start": dtime(14, 0), "end": dtime(22, 55),
                "lunch": (dtime(17, 25), dtime(18, 0)),
                "fuel":  (dtime(18, 0),  dtime(18, 18))},
}
LATE_B_CUTOFF = dtime(3, 0)  # Turno B continúa hasta 03:00 del día siguiente

# --- Ítems / categorías ---
ITEM_WAZ    = "WA-ZONE"
ITEM_BODEGA = "BODEGA-INT"
EXT_ITEMS = [
    "CALIDAD-OK", "TRANSFER", "INSPECCIÓN", "CARPA",
    ITEM_WAZ, ITEM_BODEGA, "UBIC.SOBRESTOCK", "REACOM.SOBRESTOCK",
]
ITEMS_HIDDEN = []  # ítems a ocultar opcionalmente en leyendas/filtros

# --- Paletas de color para ítems (Plotly) ---
PALETTES = {
    "Petróleo & Tierra": {
        "CALIDAD-OK":"#2A9D8F","TRANSFER":"#457B9D","INSPECCIÓN":"#3A6B35","CARPA":"#B65E3C",
        ITEM_WAZ:"#C4952B", ITEM_BODEGA:"#6C757D","UBIC.SOBRESTOCK":"#2F855A","REACOM.SOBRESTOCK":"#6D5BD0",
    },
    "Vibrante Tropical": {
        "CALIDAD-OK":"#00B894","TRANSFER":"#0984E3","INSPECCIÓN":"#6C5CE7","CARPA":"#E17055",
        ITEM_WAZ:"#FDCB6E", ITEM_BODEGA:"#636E72","UBIC.SOBRESTOCK":"#00CEC9","REACOM.SOBRESTOCK":"#A29BFE",
    },
    "Neón Pro": {
        "CALIDAD-OK":"#00E676","TRANSFER":"#2979FF","INSPECCIÓN":"#00B0FF","CARPA":"#FF5252",
        ITEM_WAZ:"#FFD54F", ITEM_BODEGA:"#90A4AE","UBIC.SOBRESTOCK":"#69F0AE","REACOM.SOBRESTOCK":"#7C4DFF",
    },
}
DEFAULT_PALETTE_NAME = "Petróleo & Tierra"

def get_palette(name: Optional[str]) -> Dict[str, str]:
    return PALETTES.get(name or DEFAULT_PALETTE_NAME, PALETTES[DEFAULT_PALETTE_NAME])

__all__ = [
    "APP_TITLE","APP_TAGLINE","DB_DEFAULT_PATH","DEFAULT_SHEET",
    "THRESH_MIN","TURNOS","LATE_B_CUTOFF",
    "ITEM_WAZ","ITEM_BODEGA","EXT_ITEMS","ITEMS_HIDDEN",
    "PALETTES","DEFAULT_PALETTE_NAME","get_palette",
]
