"""Configuración centralizada. Todo se puede sobrescribir con variables de entorno (o un fichero .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Carga KEY=VALUE de .env sin dependencias externas (no pisa variables ya definidas)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _default_stt() -> str:
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "whisper-large-v3-turbo-q4-mlx"
    return "faster-whisper-small.en"


@dataclass(frozen=True)
class Settings:
    # --- Servidor ---
    host: str = _env("TUTOR_HOST", "127.0.0.1")
    port: int = int(_env("TUTOR_PORT", "8765"))
    data_dir: Path = Path(_env("TUTOR_DATA_DIR", str(ROOT / "data")))
    models_dir: Path = Path(_env("TUTOR_MODELS_DIR", str(ROOT / "models")))

    # --- LLM ---
    # "auto": Ollama local (gratis) si está disponible; si no, la API de Claude si hay clave; si no, "mock".
    # "ollama" | "anthropic" | "mock" fuerzan un proveedor ("mock" = tutor guionizado, sin IA).
    llm_provider: str = _env("TUTOR_LLM", "auto")
    ollama_url: str = _env("TUTOR_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = _env("TUTOR_OLLAMA_MODEL", "gemma3:4b")
    chat_model: str = _env("TUTOR_CHAT_MODEL", "claude-opus-5-5")
    chat_effort: str = _env("TUTOR_CHAT_EFFORT", "low")
    eval_model: str = _env("TUTOR_EVAL_MODEL", "claude-opus-5-5")
    eval_effort: str = _env("TUTOR_EVAL_EFFORT", "low")
    summary_effort: str = _env("TUTOR_SUMMARY_EFFORT", "medium")

    # --- Speech (local) ---
    # Apple Silicon: Whisper large-v3-turbo (4 bits, MLX/GPU). Otros equipos: TUTOR_STT_MODEL=faster-whisper-small.en
    stt_model: str = _env("TUTOR_STT_MODEL", _default_stt())
    stt_threads: int = int(_env("TUTOR_STT_THREADS", "4"))
    tts_model: str = _env("TUTOR_TTS_MODEL", "kokoro-v1.0.onnx")
    tts_voice_en: str = _env("TUTOR_VOICE_EN", "af_heart")
    tts_voice_es: str = _env("TUTOR_VOICE_ES", "ef_dora")
    tts_speed: float = float(_env("TUTOR_TTS_SPEED", "0.95"))
    vad_model: str = "silero_vad.onnx"
    smart_turn_model: str = "smart-turn-v3.2-cpu.onnx"

    # --- Lección ---
    lesson_seconds: int = int(_env("TUTOR_LESSON_SECONDS", "900"))
    # Límite de RAM del proceso (el requisito es 8 GB en total; el backend debe quedarse muy por debajo).
    ram_budget_mb: int = int(_env("TUTOR_RAM_BUDGET_MB", "3500"))
    store_audio: bool = _env("TUTOR_STORE_AUDIO", "0") == "1"

    extra: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tutor.db"


settings = Settings()
