"""Repetición espaciada adaptada a conversación.

Vocabulario (Leitner): una palabra no "sube de caja" por verla, sino cuando el usuario la USA.
    día 1  presentada         → caja 0, vuelve mañana
    día 2  reutilizada/pedida → si la usa cuando se le provoca: caja +1
    día 5  otro contexto      → …
    día 10 ¿la usa sola?      → uso espontáneo: caja +2
    fallo (se le provoca y no sale / la usa mal) → vuelve a caja 1 (mañana)
    dominada: caja ≥ 5 y ≥ 2 usos espontáneos → se recicla cada 45 días

Gramática: dominio 0..1 por estructura (media móvil de la precisión observada en conversación).
"""

from __future__ import annotations

DAY = 86400.0
INTERVALS_DAYS = (1, 1, 3, 5, 10, 21, 45)  # por caja 0..6


def interval_for_box(box: int) -> float:
    return INTERVALS_DAYS[min(max(box, 0), len(INTERVALS_DAYS) - 1)] * DAY


def apply_vocab_event(item: dict, event: str, now: float) -> dict:
    """Devuelve el item actualizado. `event`: introduced | exposed | elicited | used_prompted |
    used_spontaneous | failed."""
    it = dict(item)
    if event == "introduced":
        it["exposures"] = it.get("exposures", 0) + 1
        if it.get("status") in (None, "new"):
            it["box"] = 0
            it["next_due"] = now + interval_for_box(0)
    elif event == "exposed":
        it["exposures"] = it.get("exposures", 0) + 1
    elif event == "elicited":
        it["elicitations"] = it.get("elicitations", 0) + 1
    elif event in ("used_prompted", "used_spontaneous"):
        spontaneous = event == "used_spontaneous"
        key = "spontaneous_uses" if spontaneous else "prompted_uses"
        it[key] = it.get(key, 0) + 1
        it["box"] = min(it.get("box", 0) + (2 if spontaneous else 1), len(INTERVALS_DAYS) - 1)
        it["last_used"] = now
        it["next_due"] = now + interval_for_box(it["box"])
        it["status"] = "active" if it["box"] >= 3 else "learning"
        if it["box"] >= 5 and it.get("spontaneous_uses", 0) >= 2:
            it["status"] = "mastered"
    elif event == "failed":
        it["failures"] = it.get("failures", 0) + 1
        it["box"] = 1
        it["next_due"] = now + interval_for_box(1)
        it["status"] = "learning"
    else:
        raise ValueError(f"evento de vocabulario desconocido: {event}")
    return it


def update_mastery(mastery: float, correct: int, major: int, minor: int) -> float:
    """Dominio nuevo de una estructura tras una sesión."""
    weighted_errors = major + 0.4 * minor
    observations = correct + weighted_errors
    if observations <= 0:
        return mastery
    accuracy = correct / observations
    weight = min(0.15 + 0.08 * observations, 0.5)  # más evidencia → más peso
    return round(max(0.0, min(1.0, (1 - weight) * mastery + weight * accuracy)), 3)


def review_interval_days(mastery: float) -> int:
    if mastery < 0.5:
        return 1
    if mastery < 0.7:
        return 3
    if mastery < 0.85:
        return 7
    return 14
