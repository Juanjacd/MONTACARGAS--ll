# ========================= classify.py =========================
# Normalización de texto y clasificación de movimientos/ítems
# ===============================================================

from __future__ import annotations
from typing import Optional, List
import re
import unicodedata

from cfg import ITEM_WAZ, ITEM_BODEGA

# ---------------------- Helpers de normalización ----------------------

def _norm_nfd_ascii(s: str) -> str:
    if s is None: 
        return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()

def _norm_compact(s: str) -> str:
    s = _norm_nfd_ascii(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def _norm_tokens(s: str) -> List[str]:
    s = _norm_nfd_ascii(s).lower()
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if t]

def _contains_any(txt: str, patterns: List[str]) -> bool:
    return any(p in txt for p in patterns)

def _norm_label(s: str) -> str:
    s = _norm_nfd_ascii(str(s or "")).lower()
    s = re.sub(r"[^a-z0-9\s\-_/\.]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ---------------------- Reglas de clasificación ----------------------

CANON_MAP = {
    "calidad ok": "CALIDAD-OK", "calidad-ok": "CALIDAD-OK",
    "transfer": "TRANSFER",
    "inspeccion": "INSPECCIÓN", "inspección": "INSPECCIÓN", "inspection": "INSPECCIÓN",
    "inspec": "INSPECCIÓN", "insp": "INSPECCIÓN", "insp.": "INSPECCIÓN",
    "carpa": "CARPA", "carpas": "CARPA",
    "wa zone": "WA-ZONE", "wa-zone": "WA-ZONE", "wazone": "WA-ZONE",
    "bodega int": "BODEGA-INT", "bodega-int": "BODEGA-INT", "movimiento en bodega": "BODEGA-INT",
    "ubic.sobrestock": "UBIC.SOBRESTOCK", "ubicacion sobrestock": "UBIC.SOBRESTOCK",
    "reacom.sobrestock": "REACOM.SOBRESTOCK", "reacomodacion sobrestock": "REACOM.SOBRESTOCK",
}

def has_003(s: str) -> bool:
    s = str(s or "").upper().strip()
    return s.startswith("003") or bool(re.search(r'(^|[^0-9])003([^0-9]|$)', s))

def looks_like_slot_code(s: str) -> bool:
    s = str(s or "").upper().strip()
    if not s or s.startswith("003"):
        return False
    return bool(re.fullmatch(r'[0-9A-Z][0-9A-Z/-]*', s))

def _item_base(ubic_proced: str, ubic_dest: str) -> Optional[str]:
    up = str(ubic_proced or "")
    ud = str(ubic_dest or "")
    both_txt = f"{up} | {ud}"
    compact = _norm_compact(both_txt)
    toks = set(_norm_tokens(both_txt))

    if "wazone" in compact or ("wa" in toks and "zone" in toks) or "zonav" in compact:
        return ITEM_WAZ
    if _contains_any(compact, ["transfer","traslado","trasl","transferen","trasfer","transfe","transf"]):
        return "TRANSFER"
    if _contains_any(compact, ["inspeccion","inspection","inspec","insp","insp."]):
        return "INSPECCIÓN"
    if _contains_any(compact, ["carpa","carpas"]):
        return "CARPA"
    if ("calidad" in toks and "ok" in toks) or "calidadok" in compact or "okcalidad" in compact:
        return "CALIDAD-OK"
    return None

def canon_item_from_text(s: str) -> Optional[str]:
    t = _norm_label(s)
    if not t:
        return None
    if t in CANON_MAP:
        return CANON_MAP[t]
    for k, v in CANON_MAP.items():
        if t == k or t.replace("-", " ") == k or k in t:
            return v
    return None

def item_ext(ubic_proced: str, ubic_dest: str) -> Optional[str]:
    up = str(ubic_proced or "").strip()
    ud = str(ubic_dest or "").strip()
    up003 = has_003(up)
    ud003 = has_003(ud)

    if up003 and ud003:
        return "REACOM.SOBRESTOCK"
    if up003 and not ud003:
        return "REACOM.SOBRESTOCK"
    if ud003 and not up003:
        return "UBIC.SOBRESTOCK"

    base = _item_base(up, ud)
    if base:
        return base

    if looks_like_slot_code(up) and looks_like_slot_code(ud):
        return "BODEGA-INT"

    return None

# ---------------------- Wrappers usados desde app ----------------------

def classify_any(row) -> Optional[str]:
    """Devuelve ítem extendido o None."""
    raw = canon_item_from_text(row.get("ItemRaw"))
    if raw:
        return raw
    return item_ext(row.get("Ubic.proced"), row.get("Ubicación de destino"))

def classify_any_row(row) -> str:
    """Devuelve ítem extendido o '—' (para vistas donde se requiere texto)."""
    it = classify_any(row)
    return it if it else "—"

# ---------------------- Exports ----------------------

__all__ = [
    "canon_item_from_text",
    "item_ext",
    "classify_any",
    "classify_any_row",
    "has_003",
    "looks_like_slot_code",
    "CANON_MAP",
]
