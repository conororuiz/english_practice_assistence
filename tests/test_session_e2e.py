"""Lección completa de extremo a extremo con los modelos locales reales (VAD, Whisper, Smart Turn, Kokoro)
y el tutor simulado (sin red). La lección se acorta a 60 s para que el test sea rápido; la lógica de fases
es proporcional a la duración."""

import asyncio
import dataclasses
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tutor.ai.mock import MockLLM
from tutor.config import settings
from tutor.learning.db import Database

pytestmark = pytest.mark.models
MODELS_OK = (settings.models_dir / settings.tts_model).exists()

ANSWERS = [
    "I'm fine, thanks. I'm a bit tired today.",
    "It was quite busy. Yesterday I go to a meeting with my manager.",
    "We had a deadline for a new project.",
    "The deadline is next Friday, so I have went to the office early.",
    "No, never. It's my first big project.",
    "I learned to organise my time better.",
    "I usually go for a walk or watch a series.",
]


@pytest.fixture(scope="module")
def models():
    if not MODELS_OK:
        pytest.skip("modelos no descargados")
    from tutor.session.models import Models

    return Models.load(settings)


def to16k(audio, sr):
    return np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)


class FakeClient:
    """Imita al navegador: envía audio del micrófono continuamente y confirma la reproducción."""

    def __init__(self, models, speed=2.0):
        self.models = models
        self.speed = speed  # el audio se envía más rápido que el tiempo real (el detector usa el reloj del audio)
        self.json: list[dict] = []
        self.audio_bytes = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.utterances = [to16k(*models.tts._kokoro.create(a, voice="am_michael", speed=0.95, lang="en-us"))
                           for a in ANSWERS]
        self.next_answer = 0
        self.session = None
        self.done = asyncio.Event()

    async def send_json(self, data):
        self.json.append(data)
        if data["type"] == "reply_end":
            asyncio.get_running_loop().create_task(self._play(data))
        if data["type"] == "summary":
            self.done.set()

    async def send_bytes(self, data):
        self.audio_bytes += len(data) - 4

    async def _play(self, data):
        await asyncio.sleep(data["audio_seconds"] / 4)  # "reproducción" acelerada
        await self.session.on_playback_done(data["reply_id"])
        if not data["final"]:
            await asyncio.sleep(0.3)
            await self.queue.put(self.utterances[self.next_answer % len(self.utterances)])
            self.next_answer += 1

    async def mic_loop(self):
        rng = np.random.default_rng(1)
        while not self.done.is_set():
            try:
                utt = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                utt = None
            chunk = rng.normal(0, 0.002, 512).astype(np.float32)
            frames = [chunk] if utt is None else [utt[i:i + 512] for i in range(0, len(utt) - 511, 512)]
            for f in frames:
                pcm = (np.clip(f, -1, 1) * 32767).astype("<i2").tobytes()
                await self.session.on_audio(pcm)
                await asyncio.sleep(0.032 / self.speed)


async def test_full_lesson(models, tmp_path: Path):
    from tutor.session.lesson import LessonSession

    s = dataclasses.replace(settings, lesson_seconds=60, data_dir=tmp_path)
    db = Database(tmp_path / "t.db")
    client = FakeClient(models)
    session = LessonSession(s, models, db, MockLLM(), client, use_llm_for_analysis=False)
    client.session = session
    await session.start()
    mic = asyncio.create_task(client.mic_loop())
    await asyncio.wait_for(client.done.wait(), timeout=120)
    mic.cancel()

    kinds = [m["type"] for m in client.json]
    phases = {m["phase"] for m in client.json if m["type"] == "clock"}
    finals = [m for m in client.json if m["type"] == "user_final"]
    summary = next(m for m in client.json if m["type"] == "summary")["summary"]

    assert {"warmup", "main", "practice", "wrapup"} <= phases
    assert len(finals) >= 3, finals
    assert client.audio_bytes > 24000 * 2 * 5  # el tutor habló (>5 s de audio)
    assert "aborted" not in summary
    assert summary["duration_s"] <= 60 + 25  # nunca se alarga más allá del margen duro
    assert summary["next_lesson"] != session.plan.title  # la siguiente lección es otra (planificada tras completar)
    row = db.one("SELECT status, metrics_json FROM sessions")
    assert row["status"] == "completed"
    metrics = json.loads(row["metrics_json"])
    assert metrics["fluency"]["turns"] >= 3
    # Los errores sembrados en las respuestas se detectan (evaluador local) y quedan guardados
    errs = {r["pattern_key"] for r in db.q("SELECT pattern_key FROM errors")}
    assert any("go" in e for e in errs), errs
    # La paciencia aprendida se persiste
    assert json.loads(db.one("SELECT patience_json FROM users")["patience_json"])["endpoints"] >= 3
    # El tutor presentó vocabulario con la etiqueta silenciosa y el usuario lo usó → seguimiento SRS
    assert db.one("SELECT * FROM vocab WHERE term='deadline'") is not None
    assert "turn_eval" in kinds


async def test_barge_in_stops_tutor(models, tmp_path: Path):
    from tutor.session.lesson import LessonSession

    s = dataclasses.replace(settings, lesson_seconds=600, data_dir=tmp_path)
    db = Database(tmp_path / "t.db")
    client = FakeClient(models)
    session = LessonSession(s, models, db, MockLLM(), client, use_llm_for_analysis=False)
    client.session = session
    client._play = lambda data: asyncio.sleep(0)  # el "navegador" nunca termina de reproducir
    await session.start()
    for _ in range(200):  # esperar a que el tutor empiece a sonar
        await asyncio.sleep(0.05)
        if session.tutor_speaking:
            break
    assert session.tutor_speaking
    utt = client.utterances[0]
    for i in range(0, len(utt) - 511, 512):
        await session.on_audio((utt[i:i + 512] * 32767).astype("<i2").tobytes())
    assert any(m["type"] == "stop_audio" for m in client.json)
    assert not session.tutor_speaking
    await session.stop("test")
