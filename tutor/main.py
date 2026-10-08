"""Servidor: FastAPI + WebSocket de audio. Sirve también la interfaz web (mínima)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ai.llm import AnthropicLLM
from .ai.mock import MockLLM
from .ai.ollama_llm import OllamaLLM, OllamaServer
from .config import ROOT, settings
from .learning.db import Database
from .learning.fluency import fluency_index
from .learning.memory import UserMemory
from .learning.planner import plan_next_lesson
from .learning.route import ROUTE
from .session.lesson import LessonSession
from .session.models import Models, rss_mb

logging.basicConfig(level=os.environ.get("TUTOR_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tutor")
WEB = ROOT / "web"


class State:
    models: Models | None = None
    db: Database | None = None
    llm = None
    llm_mode = "mock"
    lesson_lock = asyncio.Lock()
    ollama_server: OllamaServer | None = None


state = State()


async def _make_llm():
    provider = settings.llm_provider
    has_creds = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if provider in ("auto", "ollama"):
        state.ollama_server = OllamaServer(settings.ollama_url)
        await state.ollama_server.ensure_running()
        if await OllamaLLM.available(settings.ollama_url, settings.ollama_model):
            llm = OllamaLLM(settings.ollama_model, settings.ollama_url)
            try:
                await llm.warmup()
            except Exception as e:  # noqa: BLE001
                log.warning("No se pudo precargar el modelo local: %s", e)
            return llm, "ollama"
        log.warning("Ollama no disponible o falta el modelo %s (ollama pull %s)", settings.ollama_model, settings.ollama_model)
    if provider in ("auto", "anthropic") and has_creds:
        return AnthropicLLM(settings.chat_model, settings.chat_effort, settings.eval_model, settings.eval_effort), "anthropic"
    if provider != "mock":
        log.warning("Sin modelo de lenguaje disponible: se usa el tutor simulado (modo demo).")
    return MockLLM(), "mock"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.db = Database(settings.db_path)
    state.llm, state.llm_mode = await _make_llm()
    state.models = await asyncio.get_running_loop().run_in_executor(None, Models.load, settings)
    if rss_mb() > settings.ram_budget_mb:
        log.warning("RSS %.0f MB supera el presupuesto configurado (%d MB)", rss_mb(), settings.ram_budget_mb)
    log.info("Listo en http://%s:%d · LLM=%s", settings.host, settings.port, state.llm_mode)
    try:
        yield
    finally:
        # Al cerrar la app: el modelo sale de la memoria y, si la app arrancó Ollama, Ollama se detiene.
        if isinstance(state.llm, OllamaLLM):
            await state.llm.unload()
        if state.ollama_server:
            state.ollama_server.stop()


app = FastAPI(title="English Voice Tutor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": state.models is not None, "llm": state.llm_mode, "rss_mb": round(rss_mb()),
            "ram_budget_mb": settings.ram_budget_mb, "lesson_active": state.lesson_lock.locked()}


@app.get("/api/profile")
async def profile():
    assert state.db is not None
    mem = UserMemory(state.db)
    sessions = state.db.recent_sessions(mem.user_id, limit=30)
    history = [{
        "lesson": s["lesson_number"], "date": s["started_at"], "title": (s.get("plan") or {}).get("title"),
        "fluency_index": (s.get("metrics") or {}).get("fluency_index"),
        "fluency": (s.get("metrics") or {}).get("fluency"),
        "errors": {k: v for k, v in ((s.get("metrics") or {}).get("errors") or {}).items() if k != "list"},
    } for s in reversed(sessions)]
    vocab = mem.vocab_overview()
    return {"snapshot": mem.snapshot(), "history": history,
            "vocab": [{k: v[k] for k in ("term", "meaning_es", "status", "box", "prompted_uses", "spontaneous_uses")}
                      for v in vocab["items"][:60]],
            "next_plan": plan_next_lesson(mem).to_dict(),
            "route": _route(mem),
            "last_summary": sessions[0]["summary"] if sessions else None}


def _route(mem: UserMemory) -> list[dict]:
    progress = mem.all_unit_progress()
    current = mem.current_unit_key()
    out = []
    for i, u in enumerate(ROUTE):
        p = progress.get(u.key) or {}
        status = "mastered" if p.get("status") == "mastered" else "current" if u.key == current else "locked"
        out.append({"number": i + 1, "key": u.key, "title": u.title, "title_es": u.title_es, "status": status,
                    "score": round(p.get("score", 0)), "lessons": p.get("lessons", 0)})
    return out


class WsTransport:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.closed = False

    async def send_json(self, data: dict) -> None:
        if not self.closed:
            try:
                await self.ws.send_text(json.dumps(data, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                self.closed = True

    async def send_bytes(self, data: bytes) -> None:
        if not self.closed:
            try:
                await self.ws.send_bytes(data)
            except Exception:  # noqa: BLE001
                self.closed = True


@app.websocket("/ws/lesson")
async def lesson_ws(ws: WebSocket):
    origin = ws.headers.get("origin", "")
    host = ws.headers.get("host", "")
    if origin and host and origin.split("://", 1)[-1] != host:
        await ws.close(code=1008)  # sólo la propia interfaz puede abrir la lección (evita CSWSH)
        return
    await ws.accept()
    if state.lesson_lock.locked():
        await ws.send_text(json.dumps({"type": "error", "message": "Ya hay una lección en curso."}))
        await ws.close()
        return
    async with state.lesson_lock:
        transport = WsTransport(ws)
        assert state.models and state.db
        session = LessonSession(settings, state.models, state.db, state.llm, transport,
                                use_llm_for_analysis=state.llm_mode != "mock", local_llm=state.llm_mode == "ollama")
        await transport.send_json({"type": "hello", "llm": state.llm_mode})
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await session.on_audio(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "start" and not session.t0:
                        await session.start()
                    elif kind == "playback_done":
                        await session.on_playback_done(int(data.get("reply_id", 0)))
                    elif kind == "end":
                        await session.stop("user_ended")
                if session.finished and session.summary is not None:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            transport.closed = transport.closed or False
            if not session.finished:
                await session.stop("disconnected")
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass


def run() -> None:
    import uvicorn

    uvicorn.run("tutor.main:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
