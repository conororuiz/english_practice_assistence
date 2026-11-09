"""Speech-to-Text local.

Dos motores con la misma interfaz:
- **MLX** (Apple Silicon, por defecto): Whisper large-v3-turbo cuantizado a 4 bits en la GPU. Mucho más preciso
  con acento español que small.en y con la misma latencia (~1 s por llamada) y memoria (~0.5 GB).
- **faster-whisper** (CTranslate2): con GPU NVIDIA usable, large-v3-turbo en float16 (CUDA); si no, small.en int8
  en CPU. Se decide solo al arrancar (`resolve_stt`); si CUDA falla al cargar, vuelve a la CPU.

La lección transcribe el turno ENTERO hasta la pausa actual (no trozo a trozo): Whisper procesa siempre una
ventana de 30 s, así que cuesta lo mismo y con el contexto completo acierta mucho más.
"""

from __future__ import annotations

import glob
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Frases típicas que Whisper "inventa" sobre silencio o ruido.
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "shit", "oh",
                   "subtitles by the amara.org community", "thank you very much", "thanks"}
_TRAILING_HALLUCINATION = re.compile(  # una frase aparte al final: "… X. Thank you."
    r"(?<=[.!?…])\s+(thank you( very much| so much| for watching)?|thanks( for watching)?|bye( bye)?)[.!?…]*\s*$",
    re.I)
# small.en tiende a "limpiar" las muletillas; un prompt inicial con disfluencias le empuja a transcribirlas (para
# medir fluidez). Con large-v3-turbo ese prompt empeora la precisión con acento español ("I understand the
# question" en vez de "I don't understand…", "Yes, I enjoyed it"…): ahí sólo se da la pregunta del tutor.
_DISFLUENT_PROMPT = "Umm, let me think... uh, I mean, I went to the, uh, office."


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    unclear: list[str] = field(default_factory=list)

    @property
    def low_confidence_words(self) -> list[str]:
        if self.unclear:
            return self.unclear
        return [w.text.strip() for w in self.words if w.probability < 0.4 and len(w.text.strip()) > 2]


def collapse_repetitions(text: str, max_repeats: int = 2) -> str:
    """Bucles de repetición ("¡Vamos! ¡Vamos! ¡Vamos!…", "I'm not a fan of the I'm not a fan of the…"):
    una misma secuencia de 1-6 palabras repetida más de `max_repeats` veces seguidas se deja en una sola."""
    words = text.split()
    norm = [re.sub(r"[^\w']", "", w.lower()) for w in words]
    out: list[str] = []
    out_norm: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for n in range(1, 7):
            unit = norm[i:i + n]
            if len(unit) < n or not any(unit):
                break
            reps = 1
            while norm[i + reps * n:i + (reps + 1) * n] == unit:
                reps += 1
            if reps > max_repeats:
                out.extend(words[i:i + n])
                out_norm.extend(unit)
                i += reps * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            out_norm.append(norm[i])
            i += 1
    return " ".join(out)


def clean_transcript(text: str) -> str:
    """Quita alucinaciones típicas de Whisper conservando lo que el alumno dijo de verdad."""
    t = text.replace("�", " ").replace("’", "'").replace("‘", "'")
    # Whisper a veces imita las comillas tipográficas del contexto y cierra frases con un apóstrofo: "I work' Can…"
    t = re.sub(r"(?<=[a-rt-z])'(?=\s+[A-Z]|\s*$)", ".", t)
    t = re.sub(r"(?<!\w)'|(?<=[a-rt-z])'(?!\w)", "", t)  # comillas sueltas: "' can you repeat? 'I work"
    t = re.sub(r"[¡¿]", "", t)  # puntuación española: sólo aparece en alucinaciones (el modelo transcribe inglés)
    t = collapse_repetitions(t)
    # "…Yes, I enjoy it. Thank you." — el "Thank you" final sobre el ruido de cola es la alucinación más
    # frecuente; en mitad de una lección casi nunca es real y quitarlo no cuesta nada pedagógicamente.
    prev = None
    while prev != t:
        prev = t
        t = _TRAILING_HALLUCINATION.sub("", t).rstrip()
    t = re.sub(r"\s{2,}", " ", t).strip()
    if t.lower().strip(" .!?,") in _HALLUCINATIONS:
        return ""
    return t


def _prompt(context: str, disfluent: bool = True) -> str:
    # Sin comillas: Whisper imita la puntuación del contexto ("I work ' can you repeat? 'iwork.")
    context = re.sub(r"[“”\"‘]|(?<!\w)['’]|['’](?!\w)", "", context).replace("’", "'")
    return ((_DISFLUENT_PROMPT + " " if disfluent else "") + context)[-600:].strip() or None


MLX_TURBO = "whisper-large-v3-turbo-q4-mlx"
CT2_TURBO = "faster-whisper-large-v3-turbo"
CT2_SMALL = "faster-whisper-small.en"


def _preload_cuda_libs() -> None:
    """cuBLAS/cuDNN de los paquetes pip nvidia-* (extra `cuda`): CTranslate2 los busca por nombre en el sistema, así
    que se cargan antes a mano (sin tener que exportar LD_LIBRARY_PATH)."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        if sys.platform == "win32":
            for d in glob.glob(os.path.join(base, "*", "bin")):
                os.add_dll_directory(d)
            continue
        import ctypes

        libs = sorted(glob.glob(os.path.join(base, "*", "lib", "lib*.so*")))
        # primero las que otras necesitan (cublasLt antes que cublas, cudnn base antes que sus motores)
        libs.sort(key=lambda p: (0 if "cublasLt" in p else 1 if "cublas" in p else 2 if "libcudnn.so" in p else 3))
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_cuda: bool | None = None


def cuda_available() -> bool:
    """Hay GPU NVIDIA con driver y CTranslate2 la ve (no comprueba aún cuBLAS/cuDNN: eso lo prueba la carga)."""
    global _cuda
    if _cuda is None:
        _cuda = False
        if platform.system() != "Darwin":
            try:
                _preload_cuda_libs()
                import ctranslate2

                _cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception as e:  # noqa: BLE001
                log.info("CUDA no disponible para Whisper: %s", e)
    return _cuda


class SpeechToText:
    """faster-whisper (CTranslate2): GPU NVIDIA (float16) si la hay, si no CPU int8. TUTOR_STT_DEVICE=cpu|cuda|auto."""

    def __init__(self, model_dir: str, threads: int = 4, device: str = "auto"):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        elif device == "cuda":
            cuda_available()  # carga cuBLAS/cuDNN de pip si están
        self.device, self.name = device, Path(model_dir).name
        compute = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_dir, device=device, compute_type=compute, cpu_threads=threads)
        if device == "cuda":  # sin cuBLAS/cuDNN el fallo salta en la primera inferencia, no al cargar: probarla ya
            list(self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
        # El prompt con muletillas ayuda a small.en pero empeora large-v3-turbo (ver _DISFLUENT_PROMPT).
        self._disfluent = "large" not in Path(model_dir).name

    def transcribe(self, audio: np.ndarray, context: str = "") -> Transcript:
        """Transcribe voz (float32, 16 kHz). `context` = pregunta del tutor / texto previo del turno."""
        if audio.shape[0] < 1600:  # <100 ms: ruido
            return Transcript(text="")
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=[0.0, 0.2, 0.4],  # si sale un bucle de repetición (compression ratio alto), reintenta
            compression_ratio_threshold=2.2,
            initial_prompt=_prompt(context, disfluent=self._disfluent),
            condition_on_previous_text=False,
            vad_filter=False,  # ya segmentamos con Silero
            word_timestamps=True,
            no_speech_threshold=0.6,
        )
        words: list[Word] = []
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
                continue
            texts.append(seg.text.strip())
            logprobs.append(seg.avg_logprob)
            for w in seg.words or []:
                words.append(Word(w.word, w.start, w.end, w.probability))
        text = " ".join(t for t in texts if t)
        # Whisper a veces "inventa" una o dos palabras sobre el ruido final: se recortan las finales muy dudosas.
        trimmed = 0
        while words and trimmed < 2 and words[-1].probability < 0.3:
            words.pop()
            trimmed += 1
        if trimmed:
            text = "".join(w.text for w in words).strip()
        text = clean_transcript(text)
        if not text:
            return Transcript(text="")
        kept = {re.sub(r"[^\w']", "", w.lower()) for w in text.split()}
        words = [w for w in words if re.sub(r"[^\w']", "", w.text.strip().lower()) in kept]
        return Transcript(text=text, words=words, avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0)


class MLXSpeechToText:
    """Whisper en la GPU de Apple Silicon con MLX. Sin marcas de tiempo por palabra: duplican la latencia y
    la confianza se toma por segmento."""

    def __init__(self, model_dir: str):
        import mlx_whisper

        self._mlx = mlx_whisper
        self._path = model_dir

    def transcribe(self, audio: np.ndarray, context: str = "") -> Transcript:
        if audio.shape[0] < 1600:
            return Transcript(text="")
        r = self._mlx.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self._path,
            language="en",
            temperature=(0.0, 0.2, 0.4),
            compression_ratio_threshold=2.2,
            logprob_threshold=None,  # el reintento sólo por bucles de repetición: cada uno cuesta ~1 s
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=_prompt(context, disfluent=False),
            verbose=None,
        )
        texts, logprobs, unclear = [], [], []
        for seg in r.get("segments", []):
            if seg["no_speech_prob"] > 0.6 and seg["avg_logprob"] < -1.0:
                continue
            seg_text = seg["text"].strip()
            texts.append(seg_text)
            logprobs.append(seg["avg_logprob"])
            if seg["avg_logprob"] < -0.9:  # segmento dudoso: sus palabras largas cuentan como "poco claras"
                unclear.extend(w for w in re.findall(r"[A-Za-z']{4,}", seg_text))
        text = clean_transcript(" ".join(t for t in texts if t))
        if not text:
            return Transcript(text="")
        unclear = [w for w in unclear if w.lower() in text.lower()][:8]
        return Transcript(text=text, avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0, unclear=unclear)


def resolve_stt(models_dir: Path, name: str = "auto", device: str = "auto") -> tuple[str, str]:
    """(modelo, dispositivo) según el equipo, sin configurar nada: Apple Silicon → turbo MLX; GPU NVIDIA → turbo
    CTranslate2 en CUDA; si no → small.en en CPU. Un TUTOR_STT_MODEL / TUTOR_STT_DEVICE explícito manda."""
    if name != "auto":
        return name, device
    if platform.system() == "Darwin" and platform.machine() == "arm64" and (models_dir / MLX_TURBO).exists():
        return MLX_TURBO, device
    if device != "cpu" and (models_dir / CT2_TURBO).exists() and (device == "cuda" or cuda_available()):
        return CT2_TURBO, "cuda"
    return CT2_SMALL, "cpu" if device == "auto" else device


def load_stt(model_dir: Path, threads: int = 4, device: str = "auto"):
    """Elige el motor por el formato del modelo: weights.npz/safetensors → MLX; model.bin → faster-whisper."""
    if (model_dir / "weights.npz").exists() or (model_dir / "weights.safetensors").exists():
        try:
            return MLXSpeechToText(str(model_dir))
        except ImportError:
            raise RuntimeError(f"{model_dir.name} necesita mlx-whisper (sólo Apple Silicon). "
                               "Usa TUTOR_STT_MODEL=faster-whisper-small.en en otros equipos.") from None
    try:
        stt = SpeechToText(str(model_dir), threads=threads, device=device)
    except Exception as e:  # noqa: BLE001 — p. ej. falta cuBLAS/cuDNN: seguir en CPU antes que no arrancar
        if device == "cpu":
            raise
        small = model_dir.parent / CT2_SMALL  # turbo en CPU sería lento: small.en si está
        fallback = small if small.exists() else model_dir
        hint = " Para la GPU: uv sync --extra cuda" if re.search(r"cublas|cudnn|\.so|dll", str(e), re.I) else ""
        log.warning("Whisper en GPU falló (%s); uso %s en CPU.%s", e, fallback.name, hint)
        stt = SpeechToText(str(fallback), threads=threads, device="cpu")
    log.info("Reconocimiento de voz: %s en %s", stt.name, stt.device)
    return stt
