"""Proveedor LLM local y gratuito con Ollama (modelo pequeño cuantizado, ~3-4 GB de RAM).

Un solo modelo hace conversación, evaluación y resumen. Como sólo hay una inferencia a la vez, la
conversación tiene prioridad: si el tutor tiene que responder mientras se evalúa un turno, la evaluación
se cancela y se reintenta después (la evaluación nunca retrasa la voz del tutor).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from typing import AsyncIterator, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", num_ctx: int = 8192):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(120.0, connect=3.0))
        self.last_assistant_content: str | None = None
        self._think_param = True  # se desactiva si el modelo no admite el parámetro "think"
        self._chats = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._eval_task: asyncio.Task | None = None
        self._preempted = False
        self._eval_lock = asyncio.Lock()

    @staticmethod
    async def available(base_url: str, model: str) -> bool:
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as c:
                tags = (await c.get("/api/tags")).json()
            names = {m["name"] for m in tags.get("models", [])}
            return model in names or f"{model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False

    def _body(self, messages: list[dict], stream: bool, fmt: dict | None = None, temperature: float = 0.7) -> dict:
        body = {"model": self.model, "messages": messages, "stream": stream, "keep_alive": "30m",
                "options": {"num_ctx": self.num_ctx, "temperature": temperature}}
        if self._think_param:
            body["think"] = False  # sin razonamiento oculto: latencia mínima
        if fmt is not None:
            body["format"] = fmt
        return body

    async def _post(self, body: dict, stream: bool):
        resp = await self.client.send(self.client.build_request("POST", "/api/chat", json=body), stream=stream)
        if resp.status_code == 400 and "think" in body:
            text = (await resp.aread()).decode(errors="ignore")
            await resp.aclose()
            if "think" in text:
                self._think_param = False
                body.pop("think")
                return await self._post(body, stream)
            raise RuntimeError(f"Ollama 400: {text[:200]}")
        if resp.status_code >= 400:
            text = (await resp.aread()).decode(errors="ignore")
            await resp.aclose()
            raise RuntimeError(f"Ollama {resp.status_code}: {text[:200]}")
        return resp

    async def unload(self) -> None:
        """Saca el modelo de la memoria inmediatamente (keep_alive=0)."""
        try:
            await self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=10.0)
        except httpx.HTTPError:
            pass
        await self.client.aclose()

    async def warmup(self) -> None:
        """Carga el modelo en memoria antes de la primera lección."""
        resp = await self._post(self._body([{"role": "user", "content": "Hi"}], stream=False) | {"options": {
            "num_ctx": self.num_ctx, "num_predict": 1}}, stream=False)
        await resp.aclose()

    # ------------------------------------------------------------------ conversación (prioritaria)
    async def stream_chat(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        self.last_assistant_content = None
        self._chats += 1
        self._idle.clear()
        if self._eval_task and not self._eval_task.done():
            self._preempted = True
            self._eval_task.cancel()
        try:
            msgs = [{"role": "system", "content": system}] + [_flatten(m) for m in messages]
            resp = await self._post(self._body(msgs, stream=True), stream=True)
            parts: list[str] = []
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        yield chunk
                    if data.get("done"):
                        break
            finally:
                await resp.aclose()
            self.last_assistant_content = "".join(parts) or None
        finally:
            self._chats -= 1
            if self._chats == 0:
                self._idle.set()

    # ------------------------------------------------------------------ salida estructurada (en segundo plano)
    async def parse(self, system: str, user: str, schema: type[T], effort: str | None = None) -> T:
        async with self._eval_lock:
            for _ in range(6):  # reintentos si la conversación la interrumpe
                await self._idle.wait()
                self._preempted = False
                self._eval_task = asyncio.create_task(self._parse_once(system, user, schema))
                try:
                    return await self._eval_task
                except asyncio.CancelledError:
                    if self._preempted:
                        continue
                    raise
            raise TimeoutError("evaluación interrumpida demasiadas veces")

    async def _parse_once(self, system: str, user: str, schema: type[T]) -> T:
        msgs = [{"role": "system", "content": system + "\nAnswer ONLY with JSON that matches the schema."},
                {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(2):
            resp = await self._post(self._body(msgs, stream=False, fmt=schema.model_json_schema(), temperature=0.1),
                                    stream=False)
            data = json.loads(await resp.aread())
            await resp.aclose()
            try:
                return schema.model_validate_json(data["message"]["content"])
            except ValidationError as e:
                last_err = e
        raise ValueError(f"JSON no válido del modelo local: {last_err}")


def _flatten(m: dict) -> dict:
    content = m["content"]
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return {"role": m["role"], "content": content}


class OllamaServer:
    """Arranca `ollama serve` como proceso hijo si no hay ya uno en marcha, y lo detiene al cerrar la app.
    Si Ollama ya estaba funcionando (lo arrancó el usuario o `brew services`), no se toca: sólo se descarga el modelo."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.proc: subprocess.Popen | None = None

    async def _up(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=1.0) as c:
                return (await c.get("/api/version")).status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_running(self) -> bool:
        if await self._up():
            return True
        binary = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
        if not os.path.exists(binary):
            return False
        host = self.base_url.split("://", 1)[-1]
        env = {**os.environ, "OLLAMA_HOST": host, "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0",
               "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"}
        self.proc = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            await asyncio.sleep(0.25)
            if await self._up():
                log.info("Ollama arrancado por la app (pid %d); se detendrá al cerrarla", self.proc.pid)
                return True
        self.stop()
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            log.info("Ollama detenido")
        self.proc = None
