# rules.py
# ---------------------------------------------------------
# Reglas de negocio relacionadas con turnos/horarios.
# Solo constantes y helpers puros (sin dependencias UI).
# ---------------------------------------------------------

from __future__ import annotations
from datetime import time as dtime
from typing import List, Tuple, Optional


# Límite para considerar eventos de Turno B que
# ocurren pasada la medianoche (00:00–03:00 aprox.)
LATE_B_CUTOFF: dtime = dtime(3, 0)

# Definición de turnos y ventanas a excluir en TM
TURNOS = {
    "Turno A": {
        "start": dtime(5, 0),
        "end": dtime(13, 55),
        "lunch": (dtime(8, 20), dtime(9, 0)),
        "fuel":  (dtime(9, 0),  dtime(9, 18)),
    },
    "Turno B": {
        "start": dtime(14, 0),
        "end": dtime(22, 55),
        "lunch": (dtime(17, 25), dtime(18, 0)),
        "fuel":  (dtime(18, 0),  dtime(18, 18)),
    },
}


def normalize_turno_label(s: Optional[str]) -> Optional[str]:
    """
    Normaliza etiquetas de turno a 'Turno A' / 'Turno B'.
    Acepta variantes como 'A', 'B', 'turno a', etc.
    """
    if s is None:
        return None
    t = str(s).strip().lower().replace("_", " ").replace("-", " ")
    if t in {"a", "turno a", "turnoa"}:
        return "Turno A"
    if t in {"b", "turno b", "turnob"}:
        return "Turno B"
    return None


def turno_by_time(t: Optional[dtime]) -> Optional[str]:
    """
    Clasifica una hora del día en 'Turno A', 'Turno B' o None
    según las reglas de negocio. Los eventos entre 00:00 y LATE_B_CUTOFF
    se consideran del Turno B (horas extra).
    """
    if t is None:
        return None

    a_start, a_end = TURNOS["Turno A"]["start"], TURNOS["Turno A"]["end"]
    b_start, b_end = TURNOS["Turno B"]["start"], TURNOS["Turno B"]["end"]

    if a_start <= t < a_end:
        return "Turno A"
    if b_start <= t <= b_end:
        return "Turno B"
    if t < LATE_B_CUTOFF or t > b_end:
        return "Turno B"  # Turno B extendido (post-00:00 o > 22:55)
    return None


def windows_for_turn(turno: str) -> List[Tuple[dtime, dtime]]:
    """
    Devuelve las ventanas (lunch, fuel) para el turno dado.
    """
    t = normalize_turno_label(turno) or turno
    if t not in TURNOS:
        return []
    return [TURNOS[t]["lunch"], TURNOS[t]["fuel"]]


__all__ = [
    "LATE_B_CUTOFF",
    "TURNOS",
    "normalize_turno_label",
    "turno_by_time",
    "windows_for_turn",
]
