"""Orquestador de una clase de 15 minutos con el profesor.

Flujo por trama de audio (32 ms):  micrófono → Silero VAD → TurnDetector → eventos
  PAUSE       → transcribir el segmento (Whisper) + Smart Turn sobre el turno → completitud
  SPECULATE   → empezar a generar la respuesta (LLM + TTS) sin reproducirla todavía
  END_OF_TURN → liberar la respuesta (o generarla) · la evaluación del turno corre en paralelo
  RESUMED     → el usuario no había terminado: se descarta la respuesta y el turno sigue abierto
  BARGE_IN    → el usuario habla encima del tutor: se corta el audio y se le cede la palabra

La clase sigue la unidad actual de la ruta de aprendizaje, en pasos (tiempo + avance real):
  warmup   ~1 min    saludo (y repaso rápido de una unidad dominada)
  teach    ~2-3 min  presentación: tema de hoy, regla en español, ejemplos, palabras nuevas (un paso por turno)
  drill    ~4-5 min  ejercicios guiados, uno por turno; si falla la estructura, lo repite una vez
  produce  ~4 min    conversación real / role-play usando la estructura y las palabras
  review   ~1.5 min  repaso de los errores y palabras de hoy
  wrapup   ~1 min    cierre. Fin garantizado ~15:00.
El paso nunca retrocede. Si la presentación o los ejercicios terminan antes, se pasa al siguiente paso.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..ai.conversation import ConversationAgent, SpokenFilter, visible_text
from ..ai.prompts import drill_note, produce_reminder, teach_steps
from ..ai.coach import Coach, worth_coaching
from ..ai.evaluator import Evaluator, asks_for_help, rule_based_evaluation, term_used
from ..ai.llm import LLM
from ..ai.summarizer import Summarizer, build_summary_input
from ..audio.tts import OUTPUT_SR
from ..audio.turn_detector import FRAME_S, Ev, TurnDetector
from ..audio.vad import FRAME_SAMPLES, SileroVAD
from ..config import Settings
from ..learning.db import Database
from ..learning.engine import LearningEngine, LiveVocabLog
from ..learning.fluency import turn_metrics
from ..learning.memory import UserMemory
from ..learning.planner import plan_next_lesson
from ..learning.route import ROUTE_BY_KEY, drill_asked, error_is_unit_related
from .models import Models

log = logging.getLogger(__name__)

PREROLL_FRAMES = 19  # ~0.6 s de audio previo al inicio detectado de la voz
GRACE_HOLD_S = 0.8  # si el fin de turno es dudoso, el tutor no suena antes de este margen (el alumno puede retomar)
AUDIO_CHUNK_SAMPLES = OUTPUT_SR // 2  # trozos de 0.5 s hacia el navegador
PHASE_ORDER = ("warmup", "teach", "drill", "produce", "review", "wrapup")
# Límites de tiempo de cada paso (fracción de la clase): el paso cambia al llegar aquí aunque no haya terminado.
PHASE_END = {"warmup": 0.07, "teach": 0.27, "drill": 0.55, "produce": 0.82, "review": 0.93}


class Transport(Protocol):
    async def send_json(self, data: dict) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...


@dataclass
class Reply:
    id: int
    request: list[dict]
    phase: str
    user_turn: dict | None
    speculative_text: str | None = None
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    audible: bool = False
    audible_at: float = 0.0
    finished: bool = False
    audio_seconds: float = 0.0
    spoken: list[str] = field(default_factory=list)
    raw: str = ""
    final: bool = False  # despedida: al terminar de sonar, fin de la lección
    rescue: bool = False  # el alumno no entendió: se permite traducir la pregunta al español


def _to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class LessonSession:
    def __init__(self, settings: Settings, models: Models, db: Database, llm: LLM, transport: Transport,
                 use_llm_for_analysis: bool = True, user_id: int = 1, local_llm: bool = False):
        self.s = settings
        self.models = models
        self.db = db
        self.t = transport
        self.memory = UserMemory(db, user_id)
        self.plan = plan_next_lesson(self.memory)
        self.snapshot = self.memory.snapshot()
        self.agent = ConversationAgent(llm, self.plan, self.snapshot, settings.lesson_seconds)
        self.evaluator = Evaluator(llm if use_llm_for_analysis else None, errors_only=local_llm)
        self.coach = Coach(llm if use_llm_for_analysis else None)
        # Con un modelo local pequeño el resumen es determinista: sólo hechos medidos, nada inventado.
        self.summarizer = Summarizer(llm if use_llm_for_analysis and not local_llm else None, settings.summary_effort)
        self.engine = LearningEngine(self.memory)
        self.detector = TurnDetector(self.memory.patience())
        self.vad = SileroVAD(models.vad_session)
        self.vocab = LiveVocabLog()
        self.session_id = db.create_session(user_id, self.plan.lesson_number, self.plan.to_dict())

        self.t0 = 0.0
        self.audio_time = 0.0  # reloj del flujo de audio (s) — base temporal del detector de turnos
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=PREROLL_FRAMES)
        self.capturing = False
        self.turn_audio: list[np.ndarray] = []
        # Transcripción del turno: se transcribe todo lo dicho desde `commit_frame` en cada pausa; si el turno
        # pasa de ~25 s, lo anterior a la última pausa se congela en `committed_text` (Whisper ve 30 s como máximo).
        self.commit_frame = 0
        self.committed_text = ""
        self.committed_unclear: list[str] = []
        self.tail_text = ""
        self.tail_unclear: list[str] = []
        self.last_pause_frame = 0
        self._analysis_tail: asyncio.Task | None = None
        self._pause_seq = 0

        self.reply: Reply | None = None
        self._reply_ids = 0
        self.tutor_speaking = False
        self.turn_idx = 0
        self.user_turns: list[dict] = []
        self.eval_tasks: list[asyncio.Task] = []
        self.coach_tasks: list[asyncio.Task] = []
        self.corrections_given: list[dict] = []
        self.last_fix_turn = -10
        self.next_notes: list[str] = []
        self.pending_focus: list[str] = []
        # Pasos de la clase
        self.unit = ROUTE_BY_KEY.get((self.plan.unit or {}).get("key", ""))
        self.phase_floor = 0  # índice en PHASE_ORDER: el paso nunca retrocede
        self.warmup_replies = 0
        self.teach = teach_steps(self.plan)
        self.teach_step = 0
        self.drills = list(self.plan.drills)
        self.drill_pos = 0
        self.drill_is_asked = False  # el ejercicio drills[drill_pos] ya se ha planteado y el alumno debe responderlo
        self.drill_retry = False  # el alumno está repitiendo el ejercicio tras una corrección
        self.drill_ask_tries = 0
        self.talk = list(self.plan.talk)
        self.talk_pos = 0  # pregunta de conversación que toca plantear
        self.talk_follow_ups = 0  # repreguntas sobre la pregunta actual
        self.review_pos = 0
        self.review_retry = False
        self.ending = False
        self.finished = False
        self._clock_task: asyncio.Task | None = None
        self.summary: dict | None = None

    # ------------------------------------------------------------------ ciclo de vida
    def elapsed(self) -> float:
        return time.monotonic() - self.t0 if self.t0 else 0.0

    async def start(self) -> None:
        self.t0 = time.monotonic()
        u = self.plan.unit or {}
        await self.t.send_json({"type": "lesson", "plan": {
            "lesson_number": self.plan.lesson_number, "week": self.plan.week, "title": self.plan.title,
            "domain": self.plan.domain, "weekly_goal": self.plan.weekly_goal,
            "grammar": [g["name"] for g in self.plan.target_grammar],
            "new_vocab": self.plan.new_vocab, "review_vocab": self.plan.review_vocab,
            "unit": {"number": self.plan.unit_number, "count": self.plan.unit_count, "lesson": self.plan.unit_lesson,
                     "title": u.get("title"), "title_es": u.get("title_es"), "goal": u.get("goal"),
                     "form": u.get("form"), "examples": u.get("examples"),
                     "score": self.plan.unit_status.get("score", 0)} if u else None,
        }, "level": self.snapshot["level"], "total": self.s.lesson_seconds})
        self._clock_task = _spawn(self._clock_loop())
        self._start_reply("(The learner has joined the voice lesson and is ready.)", user_turn=None)

    async def stop(self, reason: str = "client") -> None:
        if not self.finished:
            await self.finish(reason)

    # ------------------------------------------------------------------ audio entrante
    async def on_audio(self, pcm16: bytes) -> None:
        if self.finished:
            return
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, x])
        while self._pending.shape[0] >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            await self._on_frame(frame)

    async def _on_frame(self, frame: np.ndarray) -> None:
        prob = self.vad(frame)
        self.audio_time += FRAME_S
        self._preroll.append(frame)
        if self.capturing:
            self.turn_audio.append(frame)
        for ev in self.detector.process(prob, self.audio_time, tutor_speaking=self.tutor_speaking):
            await self._handle(ev)

    async def _handle(self, ev: Ev) -> None:
        if ev == Ev.SPEECH_START:
            if not self.capturing:
                self.capturing = True
                self.turn_audio = list(self._preroll)
                self.commit_frame = self.last_pause_frame = 0
                self.committed_text, self.tail_text = "", ""
                self.committed_unclear, self.tail_unclear = [], []
            else:
                self._cancel_speculation()
            await self.t.send_json({"type": "state", "state": "user_speaking"})
        elif ev == Ev.PAUSE:
            self._pause_seq += 1
            self._analysis_tail = _spawn(self._analyze(len(self.turn_audio), self._analysis_tail, self._pause_seq))
            await self.t.send_json({"type": "state", "state": "user_pause"})
        elif ev == Ev.SPECULATE:
            text = self._turn_text()
            if text and not self.ending:
                self._start_reply(text, user_turn=None, speculative=True)
        elif ev in (Ev.END_OF_TURN, Ev.STUCK):
            await self._end_of_turn(stuck=ev == Ev.STUCK)
        elif ev == Ev.RESUMED:
            self._cancel_reply(reason="resumed")
            await self.t.send_json({"type": "state", "state": "user_speaking"})
        elif ev == Ev.BARGE_IN:
            await self._barge_in()
        elif ev == Ev.NO_RESPONSE:
            if not self.ending and (self.reply is None or self.reply.finished):
                self.next_notes.append("[The learner hasn't started answering for a while. Say something like 'Take your "
                                       "time' and rephrase your last question more simply, maybe giving an example.]")
                self.detector.stop_listening()
                self._start_reply("(silence — no answer yet)", user_turn=None)

    async def _analyze(self, upto: int, prev: asyncio.Task | None, seq: int) -> None:
        """Transcribe el turno hasta la pausa `upto` y estima su completitud. Encadenado para mantener el orden;
        si mientras esperaba ya empezó otra pausa, se salta (la siguiente transcripción cubre este audio)."""
        if prev is not None:
            try:
                await prev
            except Exception:  # noqa: BLE001
                pass
        if seq != self._pause_seq or not self.capturing:
            return
        loop = asyncio.get_running_loop()
        if (upto - self.commit_frame) * FRAME_S > 25.0 and self.last_pause_frame > self.commit_frame:
            self.committed_text = self._turn_text()
            self.committed_unclear = self.committed_unclear + self.tail_unclear
            self.commit_frame, self.tail_text, self.tail_unclear = self.last_pause_frame, "", []
        frames = self.turn_audio[self.commit_frame:upto]
        audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        whole = np.concatenate(self.turn_audio[:upto]) if upto else audio
        context = (self.agent.last_tutor_text + " " + self.committed_text)[-400:]

        def work():
            st_p = self.models.smart_turn.probability_complete(whole)
            tr = self.models.stt.transcribe(audio, context=context)
            return st_p, tr

        st_p, tr = await loop.run_in_executor(self.models.stt_pool, work)
        if not self.capturing:
            return
        self.last_pause_frame = upto
        self.tail_text, self.tail_unclear = tr.text, list(tr.low_confidence_words)
        text = self._turn_text()
        self.detector.set_analysis(st_p, text, latest=seq == self._pause_seq)
        await self.t.send_json({"type": "user_partial", "text": text})
        await self.t.send_json({"type": "turn_debug", "smart_turn": round(st_p, 2), "text_p": self.detector._text_p,
                                "completeness": round(self.detector.completeness, 2),
                                "required_silence": round(self.detector.required_silence(), 2),
                                "multiplier": round(self.detector.profile.multiplier, 2)})

    def _turn_text(self) -> str:
        return " ".join(t for t in (self.committed_text, self.tail_text) if t).strip()

    def _turn_unclear(self) -> list[str]:
        return self.committed_unclear + self.tail_unclear

    # ------------------------------------------------------------------ fin de turno del usuario
    async def _end_of_turn(self, stuck: bool) -> None:
        text = self._turn_text()
        st = self.detector.stats
        if not re.search(r"[a-zA-Z]{2,}", text):  # ruido / tos / nada transcrito: seguimos escuchando
            self.capturing = False
            self.detector.start_listening(self.audio_time, self.agent.last_tutor_text)
            await self.t.send_json({"type": "state", "state": "listening"})
            return
        total = (self.audio_time - st.final_silence) - st.started_at
        metrics = turn_metrics(text, st.speech_time, total, st.onset_latency, list(st.pauses), stuck, self._turn_unclear())
        user_turn = {"text": text, "audio": metrics.to_dict(), "t_start": st.started_at, "stuck": stuck,
                     "completeness": round(st.completeness, 2), "final_silence": round(st.final_silence, 2)}
        log.info("🎤 %s  (silencio final %.2fs, c=%.2f%s)", text, st.final_silence, st.completeness, ", atascado" if stuck else "")
        await self.t.send_json({"type": "user_final", "text": text, "stuck": stuck,
                                "final_silence": round(st.final_silence, 2), "completeness": round(st.completeness, 2)})
        await self.t.send_json({"type": "state", "state": "thinking"})
        hold = 0.0 if st.completeness >= 0.85 else GRACE_HOLD_S
        r = self.reply
        if r and not r.finished and r.speculative_text == text and r.user_turn is None:
            r.user_turn = user_turn  # la respuesta especulativa coincide: se libera sin esperar al LLM
            r.speculative_text = None
            self._open_gate(r, hold)
        else:
            self._cancel_reply(reason="replace")
            self._start_reply(text, user_turn=user_turn, hold=hold)

    @staticmethod
    def _open_gate(r: Reply, hold: float) -> None:
        if hold > 0:
            asyncio.get_running_loop().call_later(hold, r.gate.set)
        else:
            r.gate.set()

    # ------------------------------------------------------------------ respuestas del tutor
    def _phase(self) -> str:
        e = self.elapsed() / self.s.lesson_seconds
        if e >= PHASE_END["review"]:
            phase = "wrapup"
        elif e >= PHASE_END["produce"]:
            flowing = self.user_turns and self.user_turns[-1]["audio"]["words"] >= 20
            phase = "produce" if e < PHASE_END["produce"] + 0.04 and flowing and self.agent.last_phase == "produce" \
                else "review"  # la conversación va bien: se alarga un poco antes del repaso
        elif e >= PHASE_END["drill"] or self.drill_pos >= len(self.drills) or (
                self.drill_pos == len(self.drills) - 1 and self.drill_is_asked and not self.drill_retry):
            phase = "produce"  # (o el alumno está respondiendo al último ejercicio: la respuesta ya abre la conversación)
        elif e >= PHASE_END["teach"] or self.teach_step >= len(self.teach):
            phase = "drill"
        elif e >= PHASE_END["warmup"] or self.warmup_replies >= (2 if self.plan.review_unit else 1):
            phase = "teach"
        else:
            phase = "warmup"
        # nunca se vuelve a un paso anterior; y un paso sin contenido se salta
        idx = max(PHASE_ORDER.index(phase), self.phase_floor)
        if PHASE_ORDER[idx] == "teach" and self.teach_step >= len(self.teach):
            idx += 1
        if PHASE_ORDER[idx] == "drill" and self.drill_pos >= len(self.drills):
            idx += 1
        return PHASE_ORDER[idx]

    def _unit_error(self, e: dict) -> bool:
        return self.unit is not None and error_is_unit_related(self.unit, e)

    def _drill_results(self) -> tuple[int, int]:
        """(ejercicios bien, ejercicios respondidos) con las evaluaciones disponibles."""
        done = [t for t in self.user_turns if t.get("drill") and t.get("eval")]
        ok = sum(1 for t in done if not any(self._unit_error(e) for e in t["eval"].get("errors", [])))
        return ok, len(done)

    def _practice_items(self) -> list[str]:
        unit_items, other, seen = [], [], set()
        for t in self.user_turns:
            for e in (t.get("eval") or {}).get("errors", []):
                key = e["original"].lower()
                if key in seen:
                    continue
                if self._unit_error(e):
                    seen.add(key)
                    unit_items.append(f"Today's structure: \"{e['original']}\" → \"{e['corrected']}\"")
                elif e["severity"] == "major":
                    seen.add(key)
                    other.append(f"Error: \"{e['original']}\" → \"{e['corrected']}\" ({e['skill_key']})")
        items = (unit_items[:2] + other)[:3]
        for term in self.vocab.introduced:
            if term not in self.vocab.used and len(items) < 5:
                items.append(f"Today's word not yet used by the learner: '{term}'")
        if not unit_items and self.unit:
            items.append(f"One more quick question that needs today's structure ({self.unit.form.split(' · ')[0]})")
        return items

    def _quick_errors(self, text: str) -> list[dict]:
        """Comprobación instantánea con reglas (la evaluación con el LLM llega después): errores seguros."""
        if not text or asks_for_help(text):
            return []
        return [e.model_dump() for e in rule_based_evaluation(text, self.plan).errors]

    def _step_notes(self, phase: str, user_text: str) -> list[str]:
        """Instrucción concreta del paso actual (presentación / ejercicios). No modifica el estado."""
        if phase == "teach" and self.teach_step < len(self.teach):
            prefix = "React briefly to what they just said, then: " if self.teach_step == 0 and self.user_turns else ""
            return [f"[Presentation, part {self.teach_step + 1} of {len(self.teach)}. {prefix}{self.teach[self.teach_step]}]"]
        if phase == "drill" and self.drill_pos < len(self.drills):
            drill = self.drills[self.drill_pos]
            nxt = self.drills[self.drill_pos + 1] if self.drill_pos + 1 < len(self.drills) else None
            if self.drill_is_asked and asks_for_help(user_text):
                return [f"[The learner didn't understand the exercise \"{drill}\". Explain it again more simply, with an "
                        "example answer, and ask them to try.]"]
            found = ""
            if self.drill_is_asked:
                errs = sorted(self._quick_errors(user_text), key=lambda e: not self._unit_error(e))
                if errs:
                    found = f"\"{errs[0]['original']}\" → \"{errs[0]['corrected']}\" ({errs[0]['explanation_es']})"
            note = drill_note(drill, nxt, self.drill_is_asked, self.drill_retry, self.drill_pos + 1, len(self.drills),
                              found)
            if not self.drill_is_asked and self.agent.last_phase != "drill":
                note = ("[Make a short transition: \"Now let's practise.\" Then react briefly to what they just said and "
                        + note[1:])
            return [note]
        return []

    def _advance_steps(self, r: Reply, spoken: str) -> None:
        """Tras una respuesta del profesor: avanza la presentación o el ejercicio según lo que realmente dijo."""
        if r.phase == "warmup":
            self.warmup_replies += 1
        elif r.phase == "teach":
            self.teach_step += 1
        elif r.phase == "drill" and self.drill_pos < len(self.drills):
            last_user = self.user_turns[-1]["text"] if self.user_turns and r.user_turn is not None else ""
            drill = self.drills[self.drill_pos]
            nxt = self.drills[self.drill_pos + 1] if self.drill_pos + 1 < len(self.drills) else None
            if not self.drill_is_asked:
                self.drill_ask_tries += 1
                if drill_asked(drill, spoken) or self.drill_ask_tries >= 2:
                    self.drill_is_asked, self.drill_ask_tries = True, 0
            elif last_user and asks_for_help(last_user):
                pass  # se explicó otra vez el mismo ejercicio
            elif self.drill_retry or nxt is None or drill_asked(nxt, spoken):
                self.drill_pos += 1
                self.drill_retry = False
                self.drill_is_asked = nxt is not None  # la nota pedía plantear el siguiente en esta misma respuesta
            else:
                self.drill_retry = True  # el profesor corrigió y pidió repetir
        elif r.phase == "produce" and self.talk_pos < len(self.talk):
            if drill_asked(self.talk[self.talk_pos], spoken):
                self.talk_pos += 1
                self.talk_follow_ups = 0
            else:
                self.talk_follow_ups += 1
        elif r.phase == "review":
            last = self.user_turns[-1]["text"] if self.user_turns and r.user_turn is not None else ""
            wrong = any(self._unit_error(e) for e in self._quick_errors(last))
            if wrong and not self.review_retry:
                self.review_retry = True  # se pidió repetir: el ítem no avanza
            else:
                self.review_retry = False
                self.review_pos += 1
        if r.phase != "drill":
            self.drill_is_asked = False
            self.drill_retry = False

    def _turn_notes(self, user_turn: dict | None, phase: str) -> list[str]:
        notes = list(self.next_notes)
        self.next_notes.clear()
        if user_turn:
            a = user_turn["audio"]
            longest = max(a["pauses"]) if a["pauses"] else 0
            notes.append(f"[Learner turn: {a['words']} words, {len(a['pauses'])} pauses (longest {longest:.1f}s)"
                         + (f", answered after {a['onset_latency']:.1f}s" if a.get("onset_latency") else "") + "]")
            if user_turn.get("stuck"):
                notes.append("[The learner seems stuck mid-sentence. Help warmly: offer a sentence starter, a word they "
                             "may need, or a simpler question. Very short.]")
            if asks_for_help(user_turn.get("text", "")):
                notes.append(f"[The learner did NOT understand you or asked you to repeat (\"{user_turn['text'][:120]}\"). "
                             "Don't react as if they had answered. Say it again in simpler words with ONE short question "
                             "(maybe an example answer); you may add one short Spanish translation of it in <es>…</es>.]")
            if a.get("unclear_words"):
                notes.append(f"[Speech recognition was unsure about: {', '.join(a['unclear_words'][:5])}. Don't guess "
                             "or suggest replacements for these words; if the meaning isn't clear, ask what they meant.]")
        if self.turn_idx - self.last_fix_turn <= 1:
            notes.append("[You corrected something in your previous turn: let this turn flow unless an error blocks understanding.]")
        elif self.pending_focus and phase in ("produce", "review") and (
                self.plan.correction_intensity != "light" or any("Today's structure" in f for f in self.pending_focus)):
            notes.append("[From the learner's recent answers (not corrected yet): " + "; ".join(self.pending_focus[:2])
                         + ". Recast ONE of them briefly if it fits naturally.]")
            self.pending_focus.clear()
        if phase in ("drill", "produce"):
            missing = [v["term"] for v in self.plan.new_vocab if v["term"].lower() not in self.vocab.introduced]
            if missing:
                notes.append(f"[Today's words still to introduce (teach one when it fits): {', '.join(missing)}]")
        review_error = False
        if phase in ("produce", "review") and user_turn:
            errs = self._quick_errors(user_turn.get("text", ""))
            unit_errs = [e for e in errs if self._unit_error(e)]
            if unit_errs and phase == "review" and not self.review_retry:
                e = unit_errs[0]
                review_error = True
                notes.append(f"[Their answer has a mistake with today's structure: \"{e['original']}\" → "
                             f"\"{e['corrected']}\". Do NOT praise it. Say the correct sentence and ask them to say it "
                             "again.]")
            elif unit_errs:
                e = unit_errs[0]
                notes.append(f"[Their answer has a mistake with today's structure: \"{e['original']}\" → "
                             f"\"{e['corrected']}\". Do NOT say it was perfect; recast it briefly and continue.]")
            elif errs and self.turn_idx - self.last_fix_turn > 1:
                e = errs[0]
                notes.append(f"[Their answer has a clear mistake: \"{e['original']}\" → \"{e['corrected']}\". "
                             "Recast it briefly if it fits, then continue.]")
        # Al pasar de un paso a otro (presentación, inicio de los ejercicios) también se revisa la última respuesta.
        if user_turn and (phase == "teach" or (phase == "drill" and not self.drill_is_asked)):
            errs = self._quick_errors(user_turn.get("text", ""))
            notes.append(f"[Their last answer has a mistake: \"{errs[0]['original']}\" → \"{errs[0]['corrected']}\" "
                         f"(<es>{errs[0]['explanation_es']}</es>). Do NOT praise it as correct; give the correct form "
                         "briefly.]" if errs else
                         "[No clear mistake detected in their answer: praise it briefly, don't invent a correction.]")
        if phase == "produce":
            left = (PHASE_END["produce"] * self.s.lesson_seconds - self.elapsed()) / 60
            question = self.talk[self.talk_pos] if self.talk_pos < len(self.talk) else None
            if self.agent.last_phase != "produce":
                if question:
                    notes.append(f"[Start the conversation with this question: \"{question}\"]")
            else:
                notes.append(produce_reminder(self.plan, question, self.talk_follow_ups, left))
        if phase == "review" and not review_error:
            items = self._practice_items()
            if items:
                notes.append(f"[Review item for this turn (react to their last answer first): "
                             f"{items[self.review_pos % len(items)]}]")
            unused = [t for t in self.vocab.introduced if t not in self.vocab.used]
            if unused:
                notes.append(f"[Today's words the learner hasn't used yet: {', '.join(unused[:4])}. Ask a question "
                             "that makes them use one.]")
        notes.extend(self._step_notes(phase, (user_turn or {}).get("text", "")))
        return notes

    def _start_reply(self, user_text: str, user_turn: dict | None, speculative: bool = False, hold: float = 0.0) -> None:
        phase = self._phase()
        if phase == "wrapup":
            ok, tried = self._drill_results()
            self.agent.progress_hint = f" (today they got {ok} of {tried} exercises right)" if tried else ""
        if speculative:
            notes = self._peek_notes(phase)
        else:
            notes = self._turn_notes(user_turn, phase)
        request = self.agent.build_request(user_text, self.elapsed(), phase, notes, self._practice_items())
        self._reply_ids += 1
        r = Reply(self._reply_ids, request, phase, user_turn, speculative_text=user_text if speculative else None)
        r.final = phase == "wrapup"
        r.rescue = asks_for_help(user_text)
        if not speculative:
            self._open_gate(r, hold)
        self.reply = r
        r.task = _spawn(self._run_reply(r))

    def _peek_notes(self, phase: str) -> list[str]:
        """Notas para una respuesta especulativa (sin consumir las colas: se reconstruyen si se confirma)."""
        st = self.detector.stats
        provisional = {"text": self._turn_text(), "audio": {"words": len(self._turn_text().split()), "pauses": list(st.pauses),
                                 "onset_latency": st.onset_latency, "unclear_words": self._turn_unclear()}, "stuck": False}
        saved_next, saved_focus = list(self.next_notes), list(self.pending_focus)
        notes = self._turn_notes(provisional, phase)
        self.next_notes, self.pending_focus = saved_next, saved_focus
        return notes

    async def _run_reply(self, r: Reply) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        events: list = []

        async def produce() -> None:
            nonlocal events
            try:
                async for kind, val in self.agent.generate(r.request):
                    if kind == "sentence":
                        await queue.put(val)  # type: ignore[arg-type]
                    else:
                        r.raw, events = val  # type: ignore[misc]
            finally:
                await queue.put(None)

        producer = _spawn(produce())
        spoken_filter = SpokenFilter(self.plan.all_target_terms, allow_question=r.rescue,
                                     teaching=r.phase == "teach")
        try:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                sentence = spoken_filter(sentence)
                if not re.search(r"\w", visible_text(sentence)):
                    continue
                audio = await loop.run_in_executor(self.models.tts_pool, self.models.tts.synthesize, sentence)
                await r.gate.wait()
                if not r.audible:
                    await self._on_audible(r)
                log.info("🧑‍🏫 %s", visible_text(sentence))
                await self.t.send_json({"type": "tutor_text", "reply_id": r.id, "text": visible_text(sentence)})
                for i in range(0, len(audio), AUDIO_CHUNK_SAMPLES):
                    await self.t.send_bytes(struct.pack("<I", r.id) + _to_pcm16(audio[i:i + AUDIO_CHUNK_SAMPLES]))
                r.audio_seconds += len(audio) / OUTPUT_SR
                r.spoken.append(sentence)
            await producer
            await r.gate.wait()
            if not r.audible:
                await self._on_audible(r)
            r.finished = True
            self._commit_reply(r, events, interrupted=False)
            await self.t.send_json({"type": "reply_end", "reply_id": r.id, "audio_seconds": round(r.audio_seconds, 2),
                                    "final": r.final})
        except asyncio.CancelledError:
            producer.cancel()
            raise
        except Exception:
            log.exception("Fallo generando la respuesta del tutor")
            producer.cancel()
            r.finished = True
            if not r.audible:
                await self._on_audible(r)
            # Recuperación: una frase local para no dejar al usuario en silencio.
            fallback = "Sorry, I lost my train of thought for a second. Could you say that again?"
            audio = await loop.run_in_executor(self.models.tts_pool, self.models.tts.synthesize, fallback)
            await self.t.send_json({"type": "tutor_text", "reply_id": r.id, "text": fallback})
            await self.t.send_bytes(struct.pack("<I", r.id) + _to_pcm16(audio))
            r.audio_seconds += len(audio) / OUTPUT_SR
            self.agent.commit(r.request, fallback, r.phase, use_provider_content=False)
            await self.t.send_json({"type": "reply_end", "reply_id": r.id, "audio_seconds": round(r.audio_seconds, 2),
                                    "final": r.final})

    async def _on_audible(self, r: Reply) -> None:
        """Punto de no retorno: la respuesta empieza a sonar. Se confirma el turno del usuario."""
        r.audible = True
        r.audible_at = time.monotonic()
        self.phase_floor = max(self.phase_floor, PHASE_ORDER.index(r.phase))
        self.tutor_speaking = True
        self.detector.stop_listening()
        self.capturing = False
        if r.user_turn is not None:
            self._commit_user_turn(r.user_turn)
        if r.final:
            self.ending = True
        await self.t.send_json({"type": "state", "state": "tutor_speaking", "phase": r.phase})

    def _commit_user_turn(self, ut: dict) -> None:
        if self.drill_is_asked and self.drill_pos < len(self.drills):
            ut["drill"] = self.drills[self.drill_pos]  # respuesta a un ejercicio guiado
        self.turn_idx += 1
        ut["idx"] = self.turn_idx
        ut["turn_id"] = self.db.add_turn(self.session_id, self.turn_idx, "user", ut["text"], self._phase(),
                                         ut["t_start"], ut["t_start"] + ut["audio"]["total_time"], ut["audio"])
        self.user_turns.append(ut)
        self.detector.learn_from_turn()
        tutor_text = self.agent.last_tutor_text
        self.eval_tasks.append(_spawn(self._evaluate(ut, tutor_text)))

    async def _evaluate(self, ut: dict, tutor_text: str) -> None:
        ev = await self.evaluator.evaluate(tutor_text, ut["text"], self.plan, self.plan.recurring_mistakes,
                                           unclear=ut["audio"].get("unclear_words") or [])
        d = ev.model_dump()
        ut["eval"] = d
        self.db.set_turn_eval(ut["turn_id"], d)
        recent_tutor = " ".join(m["content"] if isinstance(m["content"], str) else " ".join(
            b.get("text", "") for b in m["content"] if isinstance(b, dict)) for m in self.agent.messages[-4:]
            if m["role"] == "assistant")
        targets = set(self.plan.all_target_terms) | set(self.vocab.introduced)
        reported = {u["term"].lower(): u["correct"] for u in d["target_vocab_used"]}
        for term in targets:
            t = term.lower()
            if t in reported or term_used(t, ut["text"]):
                if t in self.vocab.introduced or t in {v["term"].lower() for v in self.plan.review_vocab}:
                    self.vocab.use(t, ut["idx"], recent_tutor, correct=reported.get(t, True))
        recurring = {m["original"].lower() for m in self.plan.recurring_mistakes}
        target_skills = {g["key"] for g in self.plan.target_grammar}
        for e in d["errors"]:
            if ut.get("drill"):
                continue  # en los ejercicios el profesor ya corrige la respuesta en el momento
            if self._unit_error(e):
                self.pending_focus.append(f"Today's structure: \"{e['original']}\" → \"{e['corrected']}\"")
            elif e["severity"] == "major" and (e["original"].lower() in recurring or e["skill_key"] in target_skills
                                               or self.memory.pattern_count(e["original"], e["corrected"]) >= 1):
                self.pending_focus.append(f"\"{e['original']}\" → \"{e['corrected']}\"")
        await self.t.send_json({"type": "turn_eval", "idx": ut["idx"],
                                "errors": [{"original": e["original"], "corrected": e["corrected"], "severity": e["severity"]}
                                           for e in d["errors"]]})
        if worth_coaching(ut["text"]) and d["understood_question"] and not d["likely_asr_issue"]:
            self.coach_tasks.append(_spawn(self._coach(ut, tutor_text)))

    async def _coach(self, ut: dict, tutor_text: str) -> None:
        """Análisis de la respuesta para la pantalla (no se dice en voz alta: no interrumpe la clase)."""
        fb = (await self.coach.feedback(tutor_text, ut["text"], ut["eval"]["errors"], self.plan)).model_dump()
        ut["feedback"] = fb
        self.db.set_turn_eval(ut["turn_id"], {**ut["eval"], "feedback": fb})
        await self.t.send_json({"type": "turn_feedback", "idx": ut["idx"], "text": ut["text"], "feedback": fb})

    def _detect_vocab(self, spoken: str, tagged: set[str]) -> None:
        """Vocabulario objetivo que el tutor ha dicho (aunque el modelo no haya usado las etiquetas):
        palabra nueva → presentada; palabra dentro de una pregunta → intento de provocarla."""
        meanings = {v["term"].lower(): v["meaning_es"] for v in self.plan.new_vocab + self.plan.review_vocab}
        questions = [q for q in re.split(r"(?<=[.!?])\s+", spoken) if q.strip().endswith("?")]
        for term, meaning in meanings.items():
            if term in tagged or not term_used(term, spoken):
                continue
            if term not in self.vocab.introduced and term in {v["term"].lower() for v in self.plan.new_vocab}:
                self.vocab.introduce(term, meaning)
            if any(term_used(term, q) for q in questions):
                self.vocab.elicit(term, self.turn_idx)

    def _commit_reply(self, r: Reply, events: list, interrupted: bool) -> None:
        spoken = " ".join(r.spoken)
        # Al historial (y a la base de datos) va lo que de verdad sonó: si el filtro quitó algo (p. ej. traducciones
        # al español de cada pregunta), el modelo no debe verlo en su historial o lo seguirá repitiendo.
        filtered = visible_text(spoken) != visible_text(r.raw)
        text = spoken if interrupted or (filtered and spoken) else r.raw
        self.agent.commit(r.request, text, r.phase, use_provider_content=not interrupted and not filtered)
        self.db.add_turn(self.session_id, self.turn_idx, "tutor", visible_text(text), r.phase,
                         r.audible_at - self.t0 if r.audible_at else self.elapsed(), self.elapsed())
        for e in events:
            if e.kind == "vocab" and e.attrs.get("term"):
                self.vocab.introduce(e.attrs["term"], e.attrs.get("es"))
            elif e.kind == "elicit" and e.attrs.get("term"):
                self.vocab.elicit(e.attrs["term"], self.turn_idx)
            elif e.kind == "fix":
                self.corrections_given.append(e.attrs)
                self.last_fix_turn = self.turn_idx
        tagged = {e.attrs.get("term", "").lower() for e in events if e.kind in ("vocab", "elicit")}
        self._detect_vocab(visible_text(" ".join(r.spoken)), tagged)
        if r.audible or not interrupted:
            self._advance_steps(r, visible_text(" ".join(r.spoken) or r.raw))
        if re.search(r"more natural|you can say|we say|instead of|better to say", visible_text(" ".join(r.spoken)), re.I) \
                and not any(e.kind == "fix" for e in events):
            self.last_fix_turn = self.turn_idx  # corrección sin etiqueta: también cuenta para dosificar

    def _cancel_speculation(self) -> None:
        if self.reply and self.reply.speculative_text is not None and not self.reply.audible:
            self._cancel_reply(reason="speculation")

    def _cancel_reply(self, reason: str) -> None:
        r = self.reply
        if r and not r.finished and not r.audible and r.task:
            r.task.cancel()
            log.debug("Respuesta %s cancelada (%s)", r.id, reason)
            self.reply = None

    async def _barge_in(self) -> None:
        r = self.reply
        if r is None or self.ending:
            return
        quick = r.audible_at and time.monotonic() - r.audible_at < 2.0 and r.user_turn is not None
        if r.task and not r.task.done():
            r.task.cancel()
        if not r.finished:
            r.finished = True
            self._commit_reply(r, [], interrupted=True)
        heard = visible_text(" ".join(r.spoken))
        self.next_notes.append(f"[The learner interrupted you. They probably only heard: \"{heard[:200]}\". "
                               "Respond to what they say now.]")
        if quick:  # interrumpió justo al empezar → probablemente no había terminado: más paciencia
            self.detector.profile.observe_endpoint(True, 0.0, 0.0)
        await self.t.send_json({"type": "stop_audio", "reply_id": r.id})
        self.tutor_speaking = False
        self.detector.start_listening(self.audio_time - 0.3, self.agent.last_tutor_text)

    async def on_playback_done(self, reply_id: int) -> None:
        r = self.reply
        if r is None or r.id != reply_id or not r.finished:
            return
        await self._after_playback(r)

    async def _after_playback(self, r: Reply) -> None:
        if not self.tutor_speaking:
            return
        self.tutor_speaking = False
        if r.final:
            await self.finish("completed")
            return
        self.detector.start_listening(self.audio_time, self.agent.last_tutor_text)
        await self.t.send_json({"type": "state", "state": "listening"})

    # ------------------------------------------------------------------ reloj
    async def _clock_loop(self) -> None:
        total = self.s.lesson_seconds
        while not self.finished:
            await asyncio.sleep(0.5)
            e = self.elapsed()
            await self.t.send_json({"type": "clock", "elapsed": round(e, 1), "total": total, "phase": self._phase()})
            r = self.reply
            # Respaldo si el navegador no confirma el fin de la reproducción.
            if r and r.finished and self.tutor_speaking and r.audible_at and \
                    time.monotonic() > r.audible_at + r.audio_seconds + 4.0:
                await self._after_playback(r)
            # Cierre proactivo: casi sin tiempo y el usuario no está hablando.
            idle = self.detector.listening and not self.detector.in_turn and not self.tutor_speaking
            if not self.ending and e >= total * 0.95 and idle:
                self.ending = True
                self.detector.stop_listening()
                self._start_reply("(Time is almost up.)", user_turn=None)
            # Límite duro: nunca más de ~25 s por encima de la duración de la clase.
            if e >= total + 25 and not self.finished:
                await self._hard_stop()
                return

    async def _hard_stop(self) -> None:
        if self.reply and self.reply.task and not self.reply.task.done():
            self.reply.task.cancel()
        await self.t.send_json({"type": "stop_audio", "reply_id": self.reply.id if self.reply else 0})
        bye = "We're out of time for today. Great job! Your summary is on the screen."
        audio = await asyncio.get_running_loop().run_in_executor(self.models.tts_pool, self.models.tts.synthesize, bye)
        self._reply_ids += 1
        await self.t.send_json({"type": "tutor_text", "reply_id": self._reply_ids, "text": bye})
        await self.t.send_bytes(struct.pack("<I", self._reply_ids) + _to_pcm16(audio))
        await self.finish("completed")

    # ------------------------------------------------------------------ cierre y aprendizaje
    def _commit_pending_turn(self) -> None:
        """El alumno pulsó «Terminar» con un turno ya transcrito sin respuesta: se guarda y evalúa igualmente."""
        r = self.reply
        text = self._turn_text()
        if not self.capturing or not re.search(r"[a-zA-Z]{2,}", text) or (r and r.audible and r.user_turn is not None):
            return
        st = self.detector.stats
        total = max(self.audio_time - st.started_at, 0.1)
        speech = st.speech_time or total
        metrics = turn_metrics(text, speech, total, st.onset_latency, list(st.pauses), False, self._turn_unclear())
        self.capturing = False
        self._commit_user_turn({"text": text, "audio": metrics.to_dict(), "t_start": st.started_at, "stuck": False,
                                "completeness": round(st.completeness, 2), "final_silence": 0.0})

    async def finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        if self._clock_task and asyncio.current_task() is not self._clock_task:
            self._clock_task.cancel()
        if self.reply and self.reply.task and not self.reply.task.done() and asyncio.current_task() is not self.reply.task:
            self.reply.task.cancel()
        await self.t.send_json({"type": "state", "state": "summarizing"})
        self._commit_pending_turn()
        if self.eval_tasks:
            _, pending = await asyncio.wait(self.eval_tasks, timeout=10)
            for task in pending:
                task.cancel()
            for ut in self.user_turns:  # evaluaciones que no llegaron a tiempo: reglas locales (instantáneas)
                if "eval" not in ut:
                    ut["eval"] = rule_based_evaluation(ut["text"], self.plan).model_dump()
                    self.db.set_turn_eval(ut["turn_id"], ut["eval"])
        if self.coach_tasks:  # el último análisis suele seguir en marcha; no se espera más de 15 s
            _, pending = await asyncio.wait(self.coach_tasks, timeout=15)
            for task in pending:
                task.cancel()
        feedback = [{"idx": ut["idx"], "text": ut["text"], "feedback": ut["feedback"]}
                    for ut in self.user_turns if ut.get("feedback")]
        self.memory.save_patience(self.detector.profile)
        if len(self.user_turns) < 2:
            self.db.finish_session(self.session_id, {}, {}, status="aborted")
            self.summary = {"aborted": True, "reason": reason, "feedback": feedback}
            await self.t.send_json({"type": "summary", "summary": self.summary})
            return
        lesson_seconds = min(self.elapsed(), self.s.lesson_seconds) or self.s.lesson_seconds
        metrics = self.engine.finalize(self.session_id, self.plan, self.user_turns, self.vocab, lesson_seconds)
        metrics["duration_s"] = round(self.elapsed(), 1)
        metrics["patience"] = self.detector.profile.to_dict()
        # Se marca como completada ANTES de planificar la siguiente (el planner cuenta las lecciones completadas).
        self.db.finish_session(self.session_id, metrics, {})
        snapshot = self.memory.snapshot()
        next_plan = plan_next_lesson(self.memory)
        data = build_summary_input(metrics, self.plan, next_plan, snapshot, self.corrections_given)
        summary = await self.summarizer.summarize(data, self.plan)
        self.engine.apply_summary_profile(summary.strengths, summary.weaknesses)
        self.summary = {**summary.model_dump(), "level": metrics["level"], "fluency_index": metrics["fluency_index"],
                        "fluency_trend": metrics["fluency_trend"], "lesson_number": self.plan.lesson_number,
                        "fluency": metrics["fluency"], "duration_s": metrics["duration_s"], "reason": reason,
                        "next_lesson": next_plan.title, "unit": metrics.get("unit"),
                        "next_unit": {"title": next_plan.title, "number": next_plan.unit_number,
                                      "lesson": next_plan.unit_lesson},
                        "feedback": feedback}
        self.db.finish_session(self.session_id, metrics, self.summary)
        await self.t.send_json({"type": "summary", "summary": self.summary})


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)

    def _done(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            log.error("Tarea fallida", exc_info=t.exception())

    task.add_done_callback(_done)
    return task
