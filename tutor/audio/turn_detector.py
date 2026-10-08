"""Detección de final de turno con "patience window" adaptativa.

Lógica pura (sin modelos ni E/S) para poder probarla de forma determinista. Combina:

  1. VAD (Silero) por trama de 32 ms  → ¿hay voz ahora? ¿cuánto dura el silencio actual?
  2. Smart Turn (audio)               → entonación / prosodia: ¿suena a frase terminada?
  3. Heurística textual               → muletillas finales ("umm", "let me think"), palabras que dejan
                                         la frase abierta ("because", "to", "the"), respuestas cortas.
  4. Contexto                         → ¿la pregunta del tutor pedía una respuesta larga o un sí/no?
  5. Historial del usuario (perfil)   → cuánto suelen durar SUS pausas internas (p50/p90 aprendidos)
                                         y un multiplicador que se corrige cuando le interrumpimos.
  6. Historial reciente del turno     → si ya hizo una pausa de 2 s y siguió, esperamos al menos eso.
  7. Velocidad de habla               → los hablantes lentos reciben más margen.

Silencio requerido = interpolación entre una espera "rápida" (frase claramente terminada) y una
"lenta" (frase claramente abierta) según la probabilidad combinada de completitud `c`:

    required = fast + (1 - c)^1.3 · (slow - fast)

Nunca se cierra el turno mientras la transcripción del último segmento no esté lista. Si la frase está
claramente abierta (muletilla final, "because", "to the"…) no se cierra nunca por silencio normal: sólo al
llegar a un techo (1.5 × la espera lenta, ≤ 12 s) se emite STUCK y el tutor ofrece ayuda en vez de
"contestar" a una frase a medias.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

FRAME_S = 0.032

# --- Umbrales del VAD --------------------------------------------------------
SPEECH_ON = 0.5
SPEECH_OFF = 0.35
ON_FRAMES = 3  # 96 ms de voz para considerar que empieza a hablar
OFF_FRAMES = 8  # 256 ms de silencio para considerar que hay una pausa
BARGE_IN_PROB = 0.8
BARGE_IN_FRAMES = 8  # 256 ms de voz clara mientras habla el tutor = interrupción

# --- Heurística textual ------------------------------------------------------
FILLER_ENDINGS = (
    "um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "mm", "eh", "ah",
    "you know", "i mean", "let me think", "let me see",
    "how do you say", "how can i say", "what's the word", "how do i say",
)
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "mm", "eh", "ah"}
OPEN_ENDINGS = {
    "and", "but", "or", "so", "because", "cause", "cuz", "if", "when", "while", "that", "which",
    "who", "where", "to", "the", "a", "an", "of", "with", "for", "in", "on", "at", "by", "my",
    "your", "his", "her", "our", "their", "is", "are", "was", "were", "am", "be", "have", "has",
    "had", "will", "would", "can", "could", "should", "must", "i", "we", "you", "they", "he",
    "she", "about", "from", "than", "very", "really", "more", "then", "also", "not", "don't",
    "didn't", "doesn't", "i'm", "it's", "there", "this", "some", "like", "into", "after", "before",
}
_PRONOUNS = {"i", "you", "we", "they", "he", "she", "it", "there", "that"}
_AUX = {"have", "has", "had", "am", "is", "are", "was", "were", "will", "would", "can", "could", "should", "must",
        "do", "does", "did", "not", "don't", "didn't", "doesn't"}
SHORT_COMPLETE = {
    "yes", "no", "yeah", "yep", "nope", "sure", "exactly", "of course", "i think so", "not really",
    "maybe", "definitely", "absolutely", "right", "okay", "ok", "thanks", "thank you", "i don't know",
    "not yet", "never", "sometimes", "always", "great", "good", "fine", "nothing",
}
_CLOSING_PHRASES = ("thank you", "thanks", "i don't know", "i don't understand", "i think so", "of course",
                    "can you repeat", "say that again", "what do you mean", "i'm not sure", "me too", "you too")
_LONG_ANSWER_CUES = re.compile(
    r"^(tell me|describe|explain|why|how|what do you think|what would you|can you tell|walk me through|"
    r"what happened|imagine|talk about|what did)",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^\w' ]+", " ", text.lower()).strip()


def text_completeness(text: str, expects_long_answer: bool = False) -> float:
    """Probabilidad heurística (0..1) de que el texto sea una intervención terminada."""
    raw = text.strip()
    if not raw:
        return 0.3
    norm = _norm(raw)
    words = norm.split()
    if not words:
        return 0.3
    tail = " " + " ".join(words[-3:])
    if any(tail.endswith(" " + f) for f in FILLER_ENDINGS):
        return 0.05
    if raw.endswith(("...", "…", "-", "—", ",")):
        return 0.15
    if words[-1] in _AUX and len(words) >= 2 and words[-2] in _PRONOUNS | {"not"}:
        # Respuesta corta con auxiliar: "Yes, I have." / "No, it isn't." / "I think it is."
        return 0.75
    if any(tail.endswith(" " + p) for p in _CLOSING_PHRASES):
        # "…thank you" / "…I don't know" terminan en palabra "abierta" (you, know) pero cierran la frase
        return 0.85
    if raw.endswith("?") and len(words) >= 2:
        # Pregunta al tutor ("Can you repeat?", "What does it mean?"): espera respuesta ya
        return 0.9
    if words[-1] in OPEN_ENDINGS:
        return 0.15
    if norm in SHORT_COMPLETE:
        return 0.55 if expects_long_answer else 0.85
    score = 0.7
    if raw.endswith(("?", "!", ".")):
        score += 0.1
    if expects_long_answer and len(words) < 5:
        score -= 0.25
    if len(words) >= 8:
        score += 0.05
    return max(0.0, min(1.0, score))


def expects_long_answer(tutor_text: str) -> bool:
    """¿La última pregunta del tutor invita a una respuesta elaborada?"""
    questions = [q.strip() for q in re.split(r"(?<=[?.!])\s+", tutor_text) if q.strip().endswith("?")]
    q = questions[-1] if questions else tutor_text.strip()
    return bool(_LONG_ANSWER_CUES.search(q))


def combine(smart_turn_p: float | None, text_p: float | None) -> float:
    """Fusiona la señal acústica (entonación) y la textual."""
    if smart_turn_p is None and text_p is None:
        return 0.4
    if text_p is None:
        return smart_turn_p  # type: ignore[return-value]
    if smart_turn_p is None:
        return text_p
    c = 0.5 * smart_turn_p + 0.5 * text_p
    if text_p <= 0.15:  # señal textual fuerte de frase abierta: manda el texto
        c = min(c, 0.3)
    if smart_turn_p >= 0.85 and text_p >= 0.7:
        c = max(c, 0.9)
    return c


# --- Perfil de paciencia aprendido ---------------------------------------------


@dataclass
class PatienceProfile:
    """Lo que el sistema ha aprendido sobre el ritmo del usuario. Se persiste entre sesiones."""

    pause_p50: float = 0.9  # duración típica de una pausa interna (s)
    pause_p90: float = 2.2  # pausas largas de "estoy construyendo la frase"
    onset_p50: float = 1.5  # tiempo típico antes de empezar a responder
    multiplier: float = 1.15  # B1: empezamos siendo pacientes; se ajusta con el feedback
    pause_samples: int = 0
    false_endpoints: int = 0
    endpoints: int = 0

    def observe_pause(self, d: float) -> None:
        """Pausa interna que terminó con el usuario retomando la palabra (= no había terminado)."""
        d = min(d, 12.0)
        self.pause_samples += 1
        lr = max(0.03, 1.0 / (self.pause_samples + 4))
        # Estimación estocástica de cuantiles (Robbins-Monro), escalada a segundos.
        self.pause_p50 += lr * 1.2 * (0.5 - (d < self.pause_p50))
        self.pause_p90 += lr * 3.0 * (0.9 - (d < self.pause_p90))
        self.pause_p50 = min(max(self.pause_p50, 0.3), 3.0)
        self.pause_p90 = min(max(self.pause_p90, self.pause_p50 + 0.3), 7.0)

    def observe_onset(self, d: float) -> None:
        d = min(d, 20.0)
        self.onset_p50 += 0.1 * (d - self.onset_p50)

    def observe_endpoint(self, false_endpoint: bool, waited: float, completeness: float) -> None:
        """Ajuste por feedback: si le cortamos, más paciencia; si esperamos de más con frase clara, menos."""
        self.endpoints += 1
        if false_endpoint:
            self.false_endpoints += 1
            self.multiplier = min(self.multiplier * 1.08, 1.8)
        elif completeness >= 0.8 and waited > 1.8:
            self.multiplier = max(self.multiplier * 0.985, 0.75)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PatienceProfile":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --- Máquina de estados ------------------------------------------------------


class Ev(str, Enum):
    SPEECH_START = "speech_start"  # empieza a hablar (inicio de turno o reanudación tras pausa)
    PAUSE = "pause"  # empieza un silencio: transcribir el segmento + Smart Turn
    SPECULATE = "speculate"  # alta probabilidad de fin: se puede empezar a generar la respuesta
    END_OF_TURN = "end_of_turn"
    STUCK = "stuck"  # silencio muy largo con la frase a medias: ofrecer ayuda
    NO_RESPONSE = "no_response"  # el usuario no empieza a responder
    BARGE_IN = "barge_in"  # el usuario habla encima del tutor
    RESUMED = "resumed"  # falso final: siguió hablando antes de que sonara la respuesta


@dataclass
class TurnStats:
    started_at: float = 0.0
    speech_time: float = 0.0
    pauses: list[float] = field(default_factory=list)  # pausas internas (que se reanudaron)
    onset_latency: float | None = None
    final_silence: float = 0.0
    completeness: float = 0.0
    stuck: bool = False


class TurnDetector:
    def __init__(self, profile: PatienceProfile):
        self.profile = profile
        self.listening = False  # False mientras habla el tutor
        self._reset_turn()
        self._listen_since = 0.0
        self._nudges = 0

    # --- Control externo -------------------------------------------------
    def start_listening(self, now: float, tutor_text: str = "") -> None:
        """El tutor terminó de hablar (audio reproducido): empieza la ventana del usuario."""
        self.listening = True
        self._listen_since = now
        self._nudges = 0
        self.expects_long = expects_long_answer(tutor_text)
        self._reset_turn()

    def stop_listening(self) -> None:
        """El tutor empieza a sonar: fin de la ventana de gracia; a partir de aquí sólo hay barge-in."""
        self.listening = False
        self.grace = False
        self._ended = False
        self._voice_frames = 0

    def resume_turn(self, now: float) -> None:
        """Falso final: el usuario siguió hablando después de que cerráramos el turno. Reabrimos."""
        self.listening = True
        self.in_turn = True
        self._ended = False
        self.false_endpoint = True
        start = now - ON_FRAMES * FRAME_S
        if self._pause_start is not None:
            self.stats.pauses.append(max(0.0, start - self._pause_start))
        self._pause_start = None
        self._speaking = True
        self._seg_start = start
        self._silence_frames = 0

    def set_analysis(self, smart_turn_p: float | None, text_so_far: str | None, latest: bool = True) -> None:
        """Resultados del análisis de un segmento (llegan de forma asíncrona). `latest=False` si desde entonces
        ya empezó otra pausa: se actualiza el texto pero el turno no se puede cerrar hasta el análisis nuevo."""
        if smart_turn_p is not None:
            self._st_p = smart_turn_p
        if text_so_far is not None:
            self._text = text_so_far
            self._text_p = text_completeness(text_so_far, self.expects_long)
            if latest:
                self._text_ready = True

    # --- Cálculos --------------------------------------------------------
    @property
    def completeness(self) -> float:
        return combine(self._st_p, self._text_p)

    def _windows(self) -> tuple[float, float]:
        p = self.profile
        fast = min(max(p.pause_p50 * 0.9 + 0.35, 0.6), 1.6) * p.multiplier
        slow = min(max(p.pause_p90 * 1.6, 2.5), 7.0) * p.multiplier
        return fast, slow

    @property
    def open_sentence(self) -> bool:
        """La transcripción termina en muletilla o en una palabra que deja la frase abierta."""
        return self._text_p is not None and self._text_p <= 0.15

    def required_silence(self) -> float:
        if self.open_sentence:
            # Frase claramente a medias: no "contestamos" nunca; si el silencio llega al techo, ofrecemos ayuda.
            return self.hard_cap()
        fast, slow = self._windows()
        c = self.completeness
        req = fast + (1.0 - c) ** 1.3 * (slow - fast)
        if c < 0.7 and self.stats.pauses:  # ya ha hecho pausas largas en este turno y siguió
            req = max(req, min(max(self.stats.pauses) * 1.1, slow))
        wpm = self._turn_wpm()
        if wpm is not None and wpm < 80:  # habla despacio → más margen
            req *= 1.15
        return req

    def hard_cap(self) -> float:
        return min(self._windows()[1] * 1.5, 12.0)

    def _turn_wpm(self) -> float | None:
        if not self._text or self.stats.speech_time < 2.0:
            return None
        return len(self._text.split()) / (self.stats.speech_time / 60.0)

    # --- Bucle por trama -------------------------------------------------
    def process(self, prob: float, now: float, tutor_speaking: bool = False) -> list[Ev]:
        events: list[Ev] = []
        if tutor_speaking:
            self._voice_frames = self._voice_frames + 1 if prob >= BARGE_IN_PROB else 0
            if self._voice_frames >= BARGE_IN_FRAMES:
                self._voice_frames = 0
                events.append(Ev.BARGE_IN)
            return events
        if self._ended:
            # Ventana de gracia (el tutor está "pensando", aún no ha sonado nada): si vuelve a hablar,
            # nos hemos precipitado → se reabre el mismo turno.
            self._voice_frames = self._voice_frames + 1 if prob >= SPEECH_ON else 0
            if self.grace and self._voice_frames >= ON_FRAMES:
                self.resume_turn(now)
                events.append(Ev.RESUMED)
            return events
        if not self.listening:
            return events

        is_voice = prob >= SPEECH_ON
        is_silence = prob < SPEECH_OFF
        self._voice_frames = self._voice_frames + 1 if is_voice else 0
        self._silence_frames = self._silence_frames + 1 if is_silence else 0

        if not self.in_turn:
            if self._voice_frames >= ON_FRAMES:
                self.in_turn = True
                self._speaking = True
                self.stats.started_at = now - ON_FRAMES * FRAME_S
                self.stats.onset_latency = self.stats.started_at - self._listen_since
                self._seg_start = self.stats.started_at
                events.append(Ev.SPEECH_START)
            else:
                waited = now - self._listen_since
                limit = max(9.0, self.profile.onset_p50 * 3) + self._nudges * 12.0
                if waited > limit and self._nudges < 2:
                    self._nudges += 1
                    events.append(Ev.NO_RESPONSE)
            return events

        if self._speaking:
            if self._silence_frames >= OFF_FRAMES:
                self._speaking = False
                self._pause_start = now - OFF_FRAMES * FRAME_S
                self.stats.speech_time += self._pause_start - self._seg_start
                self._text_ready = False  # hay un segmento nuevo pendiente de transcribir
                self._speculated = False
                events.append(Ev.PAUSE)
            return events

        # En pausa
        if self._voice_frames >= ON_FRAMES:
            assert self._pause_start is not None
            self.stats.pauses.append(now - ON_FRAMES * FRAME_S - self._pause_start)
            self._speaking = True
            self._seg_start = now - ON_FRAMES * FRAME_S
            self._pause_start = None
            events.append(Ev.SPEECH_START)
            return events

        assert self._pause_start is not None
        silence = now - self._pause_start
        if not self._text_ready:
            return events
        required = self.required_silence()
        c = self.completeness
        if not self._speculated and c >= 0.7 and silence >= min(0.35, required * 0.5):
            self._speculated = True
            events.append(Ev.SPECULATE)
        if silence >= required:
            stuck = self.open_sentence
            self._finish(silence, c, stuck=stuck)
            events.append(Ev.STUCK if stuck else Ev.END_OF_TURN)
        return events

    def _finish(self, silence: float, c: float, stuck: bool) -> None:
        self._ended = True
        self.grace = True
        self._voice_frames = 0
        self.stats.final_silence = silence
        self.stats.completeness = c
        self.stats.stuck = stuck

    def _reset_turn(self) -> None:
        self.in_turn = False
        self._speaking = False
        self._ended = False
        self._voice_frames = 0
        self._silence_frames = 0
        self._pause_start: float | None = None
        self._seg_start = 0.0
        self._st_p: float | None = None
        self._text_p: float | None = None
        self._text = ""
        self._text_ready = True
        self._speculated = False
        self.grace = False
        self.false_endpoint = False
        self.stats = TurnStats()
        if not hasattr(self, "expects_long"):
            self.expects_long = False

    def learn_from_turn(self, false_endpoint: bool | None = None) -> None:
        """Actualiza el perfil de paciencia con lo observado en el turno."""
        for d in self.stats.pauses:
            if d >= 0.25:
                self.profile.observe_pause(d)
        if self.stats.onset_latency is not None:
            self.profile.observe_onset(self.stats.onset_latency)
        fe = self.false_endpoint if false_endpoint is None else false_endpoint
        self.profile.observe_endpoint(fe, self.stats.final_silence, self.stats.completeness)
