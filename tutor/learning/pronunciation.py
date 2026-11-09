"""Pronunciación sin un modelo de fonemas: dos señales de la propia transcripción.

  - Palabras con baja confianza de Whisper en lo que dijo el alumno (pueden ser ruido: por eso cuentan más las que se
    repiten en varias clases).
  - «Dila tú»: la frase objetivo se conoce; si Whisper oyó otra palabra parecida en su lugar ("thought" → "taught"),
    casi seguro es pronunciación.

Para cada palabra se da un consejo concreto con los fallos típicos de hispanohablantes.
"""

from __future__ import annotations

import difflib
import re

_SKIP = {"the", "and", "you", "are", "was", "but", "for", "not", "with", "that", "this", "have", "has", "had", "its",
         "it's", "i'm", "uh", "um", "hmm", "okay", "yeah", "mmm"}

# Palabras con letras mudas o pronunciación muy distinta a como se escriben.
_SPECIAL = {
    "comfortable": "Se dice «CÓMF-ter-bl», en tres sílabas: la «or» casi no suena.",
    "vegetable": "Se dice «VÉCH-te-bl», en tres sílabas.",
    "chocolate": "Se dice «CHÓK-let», en dos sílabas.",
    "interesting": "Se dice «ÍN-tres-ting», con el acento al principio.",
    "different": "Se dice «DÍF-rent», en dos sílabas.",
    "every": "Se dice «ÉV-ri», en dos sílabas.",
    "business": "Se dice «BÍZ-nes», en dos sílabas.",
    "wednesday": "Se dice «UÉNZ-dei»: la primera «d» no suena.",
    "island": "La «s» no suena: «ÁI-land».",
    "hour": "La «h» no suena: «ÁUer».",
    "honest": "La «h» no suena: «ÓN-est».",
    "walk": "La «l» no suena: «uok».",
    "talk": "La «l» no suena: «tok».",
    "could": "La «l» no suena: «kud».",
    "would": "La «l» no suena: «wud».",
    "should": "La «l» no suena: «shud».",
    "know": "La «k» no suena: «nou».",
    "knowledge": "La «k» no suena: «NÓL-ich».",
    "listen": "La «t» no suena: «LÍS-en».",
    "people": "Se dice «PÍ-pol», con «i» larga.",
    "work": "La «or» es una vocal neutra, como «uérk» con los labios redondos; no «uork».",
    "world": "Como «work»: «uérld», la «r» y la «l» juntas al final.",
    "focus": "Se dice «FÓU-kus», no «fo-kus».",
    "develop": "El acento va en la segunda sílaba: «di-VÉL-op».",
    "developer": "El acento va en la segunda sílaba: «di-VÉL-o-per».",
    "schedule": "Se dice «SKÉ-yul» (EE. UU.).",
    "colleague": "Se dice «KÓ-lig», en dos sílabas.",
    "meeting": "La «ee» es una «i» larga: «MIII-ting».",
}


def candidate_word(word: str) -> str:
    """La palabra normalizada si merece seguirse (no muletillas, números ni palabras de 1-2 letras)."""
    w = word.lower().strip(".,!?;:\"'()…-").replace("’", "'")
    if len(w) < 3 or w in _SKIP or not re.fullmatch(r"[a-z']+", w):
        return ""
    return w


def misheard_pairs(missing: list[str], extra: list[str]) -> list[tuple[str, str]]:
    """«Dila tú»: palabras de la frase objetivo que Whisper oyó como otra parecida (probable pronunciación)."""
    pairs = []
    for t in missing:
        best = max(extra, key=lambda e: difflib.SequenceMatcher(a=t, b=e).ratio(), default="")
        if best and best != t and difflib.SequenceMatcher(a=t, b=best).ratio() >= 0.5 and candidate_word(t):
            pairs.append((t, best))
    return pairs


def pron_tip(word: str) -> str:
    """Consejo en español para pronunciar la palabra (fallos típicos de hispanohablantes)."""
    w = word.lower()
    if w in _SPECIAL:
        return _SPECIAL[w]
    tips: list[str] = []
    if re.match(r"s[ptkcmnlw]", w):
        tips.append(f"Empieza directamente con «s», sin «e» delante: «s-{w[1:]}», no «es{w[1:]}».")
    if "th" in w:
        tips.append("«th»: pon la punta de la lengua entre los dientes y sopla (no «t» ni «d»).")
    if re.search(r"[^aeiou]ed$", w) and len(w) > 4:
        if re.search(r"[td]ed$", w):
            tips.append("Termina en «-id»: la «-ed» se oye como una sílaba más.")
        elif re.search(r"(p|k|ss|sh|ch|f|x|ck)ed$", w):
            tips.append("La «-ed» suena «t» y no añade sílaba: «" + w[:-2] + "t».")
        else:
            tips.append("La «-ed» suena «d» y no añade sílaba: «" + w[:-2] + "d».")
    if "v" in w:
        tips.append("«v»: los dientes de arriba tocan el labio de abajo (no es «b»).")
    if w.startswith("h") and w not in ("hour", "honest", "honour", "honor"):
        tips.append("«h» inicial: como una «j» muy suave, sólo aire.")
    if re.search(r"(ee|ea)", w):
        tips.append("«ee/ea» es una «i» larga y tensa.")
    elif re.search(r"[^aeiou]i[^aeiou]", w):
        tips.append("La «i» corta es más relajada, entre «i» y «e» (ship ≠ sheep).")
    if "sh" in w:
        tips.append("«sh»: como cuando pides silencio («shhh»).")
    if re.search(r"(^j|dg|ge$)", w):
        tips.append("«j/dg»: como «ch» pero con voz («dch»), no como la jota española.")
    if re.search(r"[^aeiou][st]s$|ts$|ks$|ps$", w):
        tips.append("No te comas las consonantes del final: se oyen todas.")
    if re.search(r"w", w) and not tips:
        tips.append("«w»: labios redondos, como una «u» rápida («uó-»).")
    return " ".join(tips[:2]) or "Escúchala con 🔊 y repítela despacio, marcando cada sonido del final."
