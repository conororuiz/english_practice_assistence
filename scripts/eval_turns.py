"""Banco de pruebas de la detección de final de turno con el pipeline REAL (Silero → Whisper → Smart Turn → detector).

Genera respuestas de "alumno" con Kokoro (voz distinta a la del tutor) con pausas de pensamiento, muletillas,
autocorrecciones y habla lenta, y simula el streaming en tiempo real: el resultado del análisis de cada pausa
sólo llega al detector cuando ha transcurrido el tiempo de cómputo real (como ocurre en producción).

Métricas:
  - cortes prematuros: el sistema da el turno por terminado antes de que el alumno acabe (lo peor)
  - latencia de fin de turno: desde que el alumno termina de verdad hasta que el sistema lo detecta
  - ayudas (STUCK) y finales perdidos

También acepta grabaciones reales:  --wav fichero.wav --end 7.4 [--tutor "pregunta"]

Uso:  uv run python scripts/eval_turns.py [--multiplier 1.0] [--p50 0.9 --p90 2.2]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.audio.turn_detector import FRAME_S, Ev, PatienceProfile, TurnDetector  # noqa: E402
from tutor.audio.vad import FRAME_SAMPLES, SileroVAD  # noqa: E402
from tutor.config import settings  # noqa: E402
from tutor.session.models import Models  # noqa: E402

# (pregunta del tutor, [("say", texto) | ("pause", segundos)], tipo)
SCENARIOS = [
    ("What made it so busy?", [("say", "We had a deadline for a new project.")], "fluida"),
    ("What did you do yesterday?", [("say", "Yesterday I went to"), ("pause", 1.8), ("say", "umm"), ("pause", 2.0),
                                     ("say", "I went to the office because"), ("pause", 2.0),
                                     ("say", "I needed to finish a report.")], "pausas de pensamiento"),
    ("Why is the project late?", [("say", "I think the main problem is that"), ("pause", 3.0),
                                   ("say", "we don't have enough people in the team.")], "pausa larga construyendo"),
    ("Have you ever been to London?", [("say", "Yes, I have.")], "respuesta corta"),
    ("What do you do in your free time?", [("say", "I like reading,"), ("pause", 1.2), ("say", "cooking,"),
                                           ("pause", 1.4), ("say", "and going to the gym.")], "enumeración"),
    ("When is the deadline?", [("say", "Well, let me think."), ("pause", 2.5), ("say", "Probably in two weeks.")],
     "muletilla 'let me think'"),
    ("Where did you travel last week?", [("say", "Last week I go,"), ("pause", 1.0),
                                         ("say", "I went to London for work.")], "autocorrección"),
    ("Tell me about your job.", [("say", "My job is"), ("pause", 1.5), ("say", "about"), ("pause", 1.5),
                                 ("say", "managing projects"), ("pause", 2.0), ("say", "for a big company.")], "habla lenta"),
    ("How was your weekend?", [("say", "It was great. I visited my parents and we had lunch together.")], "fluida larga"),
    ("What would you do if you had more time?", [("say", "If I had more time, I would"), ("pause", 2.2),
                                                 ("say", "uh"), ("pause", 1.5),
                                                 ("say", "learn to play the guitar.")], "condicional con pausas"),
]
LEARNER_VOICE = "am_michael"


def synth(models: Models, pieces: list[tuple[str, object]], speed: float) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    chunks = [rng.normal(0, 0.002, 8000).astype(np.float32)]  # 0.5 s antes de empezar
    for kind, val in pieces:
        if kind == "say":
            audio, sr = models.tts._kokoro.create(str(val), voice=LEARNER_VOICE, speed=speed, lang="en-us")
            x = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
            chunks.append(x)
        else:
            chunks.append(rng.normal(0, 0.002, int(float(val) * 16000)).astype(np.float32))
    speech = np.concatenate(chunks)
    true_end = len(speech) / 16000
    tail = rng.normal(0, 0.002, int(14 * 16000)).astype(np.float32)  # silencio final largo
    return np.concatenate([speech, tail]), true_end


FILLER_PIECES = {"umm", "uh", "um", "erm"}


def _to16k(audio: np.ndarray, sr: int) -> np.ndarray:
    return np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)


_ALIGNER = None


def _aligner():
    """faster-whisper small.en sólo para localizar las palabras del audio sintético (marcas de tiempo por palabra);
    la transcripción que se evalúa es la del motor configurado (models.stt)."""
    global _ALIGNER
    if _ALIGNER is None:
        from faster_whisper import WhisperModel

        _ALIGNER = WhisperModel(str(settings.models_dir / "faster-whisper-small.en"), device="cpu", compute_type="int8")
    return _ALIGNER


def synth_natural(models: Models, pieces: list[tuple[str, object]], speed: float) -> tuple[np.ndarray, float]:
    """Sintetiza la intervención completa de una vez (prosodia de continuación realista) y abre las pausas en
    los límites de palabra correspondientes (localizados con las marcas de tiempo de Whisper). Las muletillas
    se sintetizan aparte y se colocan dentro de la pausa."""
    rng = np.random.default_rng(0)
    noise = lambda sec: rng.normal(0, 0.002, int(sec * 16000)).astype(np.float32)  # noqa: E731
    words_text = [str(v) for k, v in pieces if k == "say" and str(v).lower() not in FILLER_PIECES]
    audio, sr = models.tts._kokoro.create(" ".join(words_text), voice=LEARNER_VOICE, speed=speed, lang="en-us")
    full = _to16k(audio, sr)
    segs, _ = _aligner().transcribe(full, language="en", word_timestamps=True, beam_size=1)
    wts = [w for sg in segs for w in (sg.words or [])]
    # puntos de corte: tras la última palabra de cada trozo seguido de una pausa
    cuts, n_words, inserts = [], 0, []
    current_insert: list[np.ndarray] = []
    for k, v in pieces:
        if k == "say" and str(v).lower() not in FILLER_PIECES:
            if current_insert:
                inserts.append((n_words, current_insert))
                current_insert = []
            n_words += len(str(v).split())
        elif k == "say":
            fa, fsr = models.tts._kokoro.create(str(v), voice=LEARNER_VOICE, speed=speed * 0.8, lang="en-us")
            current_insert.append(_to16k(fa, fsr))
        else:
            current_insert.append(noise(float(v)))
    out, pos = [noise(0.5)], 0
    for after_words, parts in inserts:
        if after_words <= 0 or after_words > len(wts):
            continue
        w_end = wts[after_words - 1].end
        w_next = wts[after_words].start if after_words < len(wts) else w_end
        cut = int(((w_end + w_next) / 2) * 16000)
        out.append(full[pos:cut])
        out.extend(parts)
        pos = cut
    out.append(full[pos:])
    speech = np.concatenate(out)
    # fin real = último instante con energía de voz
    env = np.abs(speech) > 0.02
    true_end = (np.nonzero(env)[0][-1] + 1) / 16000 if env.any() else len(speech) / 16000
    return np.concatenate([speech, noise(14)]), float(true_end)


def simulate(models: Models, audio: np.ndarray, tutor: str, profile: PatienceProfile, stop_at_end: bool = True) -> dict:
    vad = SileroVAD(models.vad_session)
    det = TurnDetector(profile)
    det.start_listening(0.0, tutor)
    frames = audio[: len(audio) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    turn: list[np.ndarray] = []
    texts: list[str] = []
    pending: list[tuple[float, float, str, int]] = []  # (llega_en, smart_turn, texto, nº de pausa)
    pause_seq = 0
    capturing, t = False, 0.0
    events, compute = [], []
    ends: list[tuple[float, str]] = []
    for idx, f in enumerate(frames):
        t += FRAME_S
        if capturing:
            turn.append(f)
        while pending and pending[0][0] <= t:
            _, st, txt, seq = pending.pop(0)
            det.set_analysis(st, txt, latest=seq == pause_seq)
        for ev in det.process(vad(f), t):
            events.append((ev.value, round(t, 2)))
            if ev == Ev.SPEECH_START and not capturing:  # como en la lección: ~0.6 s de audio previo
                capturing, turn = True, list(frames[max(0, idx - 18):idx + 1])
            elif ev == Ev.PAUSE:
                # Como en la lección: se transcribe el turno entero hasta esta pausa.
                whole = np.concatenate(turn)
                pause_seq += 1
                t0 = time.perf_counter()
                st = models.smart_turn.probability_complete(whole)
                tr = models.stt.transcribe(whole, context=tutor)
                dt = time.perf_counter() - t0
                compute.append(dt)
                texts = [tr.text] if tr.text else []
                ready_at = max(t + dt, pending[-1][0] if pending else 0)
                pending.append((ready_at, st, " ".join(texts), pause_seq))
            elif ev in (Ev.END_OF_TURN, Ev.STUCK):
                ends.append((t, ev.value))
                if stop_at_end:
                    break
        if ends and stop_at_end:
            break
    return {"ends": ends, "events": events, "text": " ".join(texts), "compute": compute}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=float, default=PatienceProfile().multiplier)
    ap.add_argument("--p50", type=float, default=PatienceProfile().pause_p50)
    ap.add_argument("--p90", type=float, default=PatienceProfile().pause_p90)
    ap.add_argument("--speed", type=float, default=0.9, help="velocidad de habla del alumno sintético")
    ap.add_argument("--prosody", choices=["natural", "independent"], default="natural",
                    help="natural: frase sintetizada entera y cortada; independent: cada trozo sintetizado por separado")
    ap.add_argument("--grace", type=float, default=1.2,
                    help="tiempo típico desde el fin de turno hasta que suena el tutor (si el alumno retoma antes, "
                         "el sistema reabre el turno sin que se note)")
    ap.add_argument("--wav", help="grabación real (16 kHz mono) en lugar de los escenarios sintéticos")
    ap.add_argument("--end", type=float, help="segundo en que el hablante termina de verdad (para --wav)")
    ap.add_argument("--tutor", default="Tell me about it.")
    args = ap.parse_args()

    models = Models.load(settings)
    cases = []
    if args.wav:
        import soundfile as sf

        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
        audio = np.concatenate([audio, np.zeros(14 * 16000, np.float32)])
        cases.append(("grabación", args.tutor, audio, args.end or len(audio) / 16000 - 14))
    else:
        for tutor, pieces, kind in SCENARIOS:
            fn = synth_natural if args.prosody == "natural" else synth
            audio, true_end = fn(models, pieces, args.speed)
            cases.append((kind, tutor, audio, true_end))

    latencies, cutoffs, recovered, stuck, missed = [], 0, 0, 0, 0
    print(f"{'escenario':28} {'resultado':12} {'latencia':>9}  transcripción")
    for kind, tutor, audio, true_end in cases:
        prof = PatienceProfile(pause_p50=args.p50, pause_p90=args.p90, multiplier=args.multiplier)
        r = simulate(models, audio, tutor, prof, stop_at_end=False)
        ends = r["ends"]
        if not ends:
            missed += 1
            verdict, lat = "PERDIDO", float("nan")
        else:
            first, (final_t, final_kind) = ends[0][0], ends[-1]
            lat = final_t - true_end
            resumed = [t for e, t in r["events"] if e == "resumed"]
            if first < true_end - 0.05:
                in_grace = all(any(0 <= rt - et <= args.grace for rt in resumed) for et, _ in ends[:-1])
                if in_grace and final_t >= true_end - 0.05:
                    recovered += 1
                    verdict = "recuperado"
                else:
                    cutoffs += 1
                    verdict, lat = "CORTE!", first - true_end
            else:
                verdict = "ayuda" if final_kind == "stuck" else "ok"
            if verdict != "CORTE!":
                stuck += final_kind == "stuck"
                latencies.append(lat)
        print(f"{kind:28} {verdict:12} {lat:8.2f}s  {r['text'][:70]}")
    n = len(cases)
    print("\nResumen")
    print(f"  cortes prematuros: {cutoffs}/{n}   recuperados en la ventana de gracia: {recovered}/{n}   "
          f"finales perdidos: {missed}/{n}   ayudas (stuck): {stuck}/{n}")
    if latencies:
        print(f"  latencia de fin de turno: mediana {statistics.median(latencies):.2f}s · máx {max(latencies):.2f}s")


if __name__ == "__main__":
    main()
