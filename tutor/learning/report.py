"""Informe de una clase con audio: un HTML autónomo (sin servidor) con lo que dijo el alumno, la versión corregida
—con su audio— y el porqué, la pronunciación a practicar y las cifras de la clase. Para guardarlo o escucharlo después.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import wave
from typing import Callable

import numpy as np

MAX_CLIPS = 12
SR = 24000


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def report_items(summary: dict) -> list[dict]:
    """Frases corregidas de la clase (sin repetir), las correcciones primero."""
    items, seen = [], set()
    for fb in summary.get("feedback") or []:
        for s in fb["feedback"].get("suggestions", []):
            text = (s.get("sentence") or s.get("better") or "").strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            items.append({"kind": s.get("kind", ""), "said": s.get("you_said") or "", "better": s.get("better", ""),
                          "sentence": text, "why_es": s.get("why_es", ""), "learner": fb.get("text", "")})
    items.sort(key=lambda i: i["kind"] != "correction")
    return items


def build_report(summary: dict, started_at: float, tts: Callable[[str], np.ndarray]) -> str:
    e = html.escape
    clips: dict[str, str] = {}

    def audio(text: str) -> str:
        if not text or (text not in clips and len(clips) >= MAX_CLIPS):
            return ""
        if text not in clips:
            clips[text] = base64.b64encode(wav_bytes(tts(text))).decode()
        return f'<audio controls preload="none" src="data:audio/wav;base64,{clips[text]}"></audio>'

    date = dt.datetime.fromtimestamp(started_at).strftime("%d/%m/%Y %H:%M")
    rp = summary.get("roleplay")
    unit = summary.get("unit") or {}
    title = rp["title_es"] if rp else unit.get("title_es") or summary.get("next_lesson") or "Clase"
    flu = summary.get("fluency") or {}
    parts = [f"<h1>{e(title)}</h1><p class='muted'>Clase {e(str(summary.get('lesson_number', '')))} · {e(date)} · "
             f"nivel {e(str(summary.get('level', '')))}</p>"]
    kpis = [(flu.get("speaking_wpm"), "palabras/min"), (flu.get("mean_turn_words"), "palabras por respuesta"),
            (summary.get("fluency_index"), "índice de fluidez")]
    if unit:
        kpis.append((f"{unit.get('score', 0)}%", "dominio de la unidad"))
    parts.append("<div class='kpis'>" + "".join(f"<div><b>{e(str(v if v is not None else '–'))}</b>"
                                               f"<span>{e(lbl)}</span></div>" for v, lbl in kpis) + "</div>")
    if rp:
        parts.append("<h2>Objetivos del role-play</h2><ul>" + "".join(
            f"<li>{'✅' if g['ok'] else '⬜'} {e(g['es'])}</li>" for g in rp["goals"]) + "</ul>")
    items = report_items(summary)
    if items:
        parts.append("<h2>Tus frases, corregidas</h2><p class='muted'>Escucha cada una y repítela en voz alta.</p>")
        for i in items:
            said = f"<p class='said'>Dijiste: “{e(i['learner'])}”</p>" if i["learner"] and i["kind"] != "richer_answer" \
                else ""
            label = {"correction": "Corrección", "more_natural": "Más natural", "richer_answer": "Respuesta más completa"}
            parts.append(f"<section><div class='kind'>{label.get(i['kind'], '')}</div>{said}"
                         f"<p class='better'>{e(i['sentence'])}</p>{audio(i['sentence'])}"
                         + (f"<p>{e(i['why_es'])}</p>" if i["why_es"] else "") + "</section>")
    pron = summary.get("pronunciation") or []
    if pron:
        parts.append("<h2>Pronunciación para practicar</h2>")
        for p in pron:
            heard = f" <span class='muted'>(se entendió «{e(p['heard'])}»)</span>" if p.get("heard") else ""
            parts.append(f"<section><p class='better'>{e(p['word'])}{heard}</p>{audio(p['word'])}"
                         f"<p>{e(p['tip'])}</p></section>")
    if summary.get("useful_expression"):
        parts.append(f"<h2>Expresión útil</h2><section><p class='better'>{e(summary['useful_expression'])}</p>"
                     f"{audio(summary['useful_expression'])}<p class='muted'>{e(summary.get('useful_expression_es', ''))}"
                     "</p></section>")
    for key, head in (("well_done", "Bien hecho"), ("improve", "Para mejorar")):
        if summary.get(key):
            parts.append(f"<h2>{head}</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in summary[key]) + "</ul>")
    if summary.get("tomorrow"):
        parts.append(f"<h2>Próxima clase</h2><p>{e(summary['tomorrow'])}</p>")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(title)} · informe</title>
<style>
:root {{ --bg:#f6f7fb; --card:#fff; --ink:#1d2433; --muted:#6b7385; --accent:#3b82f6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11141b; --card:#1a1f2a; --ink:#e8ecf3; --muted:#9aa3b5; }} }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.5 -apple-system, system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:720px; margin:0 auto; }}
h1 {{ margin:0 0 4px; }} h2 {{ margin:28px 0 10px; font-size:19px; }}
.muted {{ color:var(--muted); }}
section {{ background:var(--card); border-radius:12px; padding:12px 14px; margin:10px 0; }}
.kind {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--accent); }}
.said {{ color:var(--muted); margin:4px 0; }} .better {{ font-weight:600; font-size:17px; margin:4px 0 8px; }}
audio {{ width:100%; height:36px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin:16px 0; }}
.kpis div {{ background:var(--card); border-radius:12px; padding:10px 12px; }}
.kpis b {{ display:block; font-size:20px; }} .kpis span {{ color:var(--muted); font-size:13px; }}
</style></head><body><main>{''.join(parts)}</main></body></html>"""
