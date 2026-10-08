# English Voice Tutor

Profesor de inglés **por voz** para hispanohablantes con nivel ~B1: abres la app y recibes una **clase de 15 minutos**
con tu profesor particular (sin pulsar botones). Cada clase enseña un tema concreto de una **ruta de aprendizaje**
(p. ej. presente simple con he/she + 8 palabras útiles) y no se pasa al siguiente tema hasta que lo dominas.

- Clase guiada en 6 pasos: saludo → explicación (regla en español, ejemplos, palabras nuevas) → ejercicios guiados
  uno a uno (si fallas la estructura, te corrige y lo repites) → conversación real usando lo aprendido → repaso de
  tus errores de hoy → cierre. Pantalla con el tema del día y el paso en el que estás.
- Ruta de 20 unidades (presente simple → frecuencia → presente continuo → pasado → … → condicionales, opiniones,
  preguntas). Una unidad se da por dominada con ≥2 clases, ≥12 usos correctos, ≥75 % de acierto (≥80 % en la
  última clase) y 6 de sus 8 palabras usadas. Las unidades dominadas se repasan unos días después.

- Conversación natural; el tutor habla en inglés y usa el español sólo para explicar errores, palabras nuevas o
  rescatarte si te pierdes (con voz española).
- **Detección de final de turno adaptativa**: distingue pausas de pensamiento ("umm…", "because…") del final real,
  y aprende cuánto tardas tú.
- Correcciones dosificadas (fluidez primero), práctica de tus errores al final y resumen corto.
- Memoria de progreso: gramática, errores recurrentes, vocabulario con repetición espaciada, métricas de fluidez y
  una ruta de lecciones que cambia según tu rendimiento.

Diseño completo, decisiones y resultados medidos: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Requisitos

- macOS / Linux con Python 3.12 y [uv](https://docs.astral.sh/uv/) (Windows debería funcionar; no probado).
- ~4,5 GB de disco para modelos. RAM medida del sistema completo durante una lección (backend + Gemma + Chrome):
  5,8–7,3 GB de pico. Para dejar más margen: `TUTOR_OLLAMA_MODEL=qwen3.5:4b` (~1,1 GB menos, conversación algo peor).
- Ollama sólo se ejecuta mientras la app está abierta: la app lo arranca (con atención flash y caché en 8 bits para
  gastar menos memoria) y, al cerrarla con Ctrl+C, descarga el modelo y lo detiene. Si ya tenías Ollama en marcha
  por tu cuenta, la app no lo para; sólo libera el modelo de la memoria.
- **Gratis y 100 % local** por defecto: el tutor es un modelo de lenguaje local (Gemma 3 4B) servido con
  [Ollama](https://ollama.com). Opcional: la API de Claude (de pago, aparte de la suscripción de Claude) para una
  conversación y una evaluación de más calidad. Sin ninguno de los dos arranca en **modo demo** (tutor guionizado).
- Chrome, Edge o Safari recientes. **Recomendado: auriculares** (evita que el micrófono capte al tutor).

## Puesta en marcha

```bash
cd ~/english-voice-tutor
brew install ollama                  # servidor de modelos local (una vez; la app lo arranca y lo para sola)
ollama pull gemma3:4b                # el tutor (3,3 GB, una vez)
uv sync                              # dependencias
./scripts/download_models.sh         # modelos de voz (~830 MB; +464 MB en Apple Silicon)
uv run python -m tutor.main          # → http://127.0.0.1:8765
```

Abre http://127.0.0.1:8765, pulsa **Empezar clase** y habla. `CC` muestra/oculta los subtítulos;
**Alt+D** muestra el panel de detección de turno (completitud, silencio requerido, multiplicador de paciencia).
La pestaña **Progreso** muestra tu ruta (unidades dominadas, la actual con su % de dominio), nivel, errores
recurrentes, vocabulario, gramática y la próxima clase.

## Pruebas

```bash
uv run pytest -m "not models"                 # rápidas (lógica): detector, aprendizaje, planner…
uv run pytest                                 # + lección completa con los modelos reales (~75 s)
uv run python scripts/eval_turns.py           # banco de fin de turno (cortes prematuros, latencia)
uv run python scripts/eval_turns.py --wav mi_grabacion.wav --end 7.4   # con una grabación tuya
uv run python scripts/measure_memory.py       # RAM con modelos y pico bajo carga
uv run python scripts/compare_llm.py gemma3:4b qwen3.5:4b   # compara modelos locales como tutor
uv run python scripts/simulate_class.py --classes 3   # clases completas sólo texto: Gemma de profesor y de alumno
TUTOR_LESSON_SECONDS=60 uv run python -m tutor.main   # y en otra terminal:
uv run python scripts/ws_smoke.py             # lección de punta a punta por WebSocket
uv run --with playwright python scripts/browser_e2e.py   # la interfaz real en Chrome headless
```

## Estructura

```
tutor/
  audio/      vad.py · stt.py · tts.py · smart_turn.py · turn_detector.py (núcleo de la detección de turno)
  ai/         conversation.py · prompts.py · evaluator.py · summarizer.py · llm.py · mock.py · schemas.py
  learning/   route.py (ruta de 20 unidades y criterios de dominio) · planner.py · engine.py · db.py · memory.py
              srs.py · fluency.py · curriculum.py (contextos de conversación)
  session/    lesson.py (orquestador de la clase: pasos, ejercicios, repaso) · models.py (carga de modelos)
  main.py     servidor FastAPI + WebSocket
web/          interfaz mínima (index.html, app.js, mic-worklet.js, style.css)
scripts/      descarga de modelos, banco de turnos, medición de RAM, prueba WebSocket
docs/         ARCHITECTURE.md
```

## Configuración (`.env`)

| Variable | Por defecto | |
|---|---|---|
| `TUTOR_LLM` | `auto` | `auto` = Ollama si está; si no, Claude si hay clave; si no, demo. O fuerza `ollama` / `anthropic` / `mock` |
| `TUTOR_OLLAMA_MODEL` | `gemma3:4b` | Modelo local (alternativa ligera: `qwen3.5:4b`) |
| `ANTHROPIC_API_KEY` | — | Opcional (API de pago de Claude) |
| `TUTOR_CHAT_MODEL` / `TUTOR_CHAT_EFFORT` | `claude-opus-5-5` / `low` | Modelo y esfuerzo de la conversación |
| `TUTOR_EVAL_MODEL` / `TUTOR_EVAL_EFFORT` | `claude-opus-5-5` / `low` | Evaluación de cada turno |
| `TUTOR_VOICE_EN` / `TUTOR_VOICE_ES` | `af_heart` / `ef_dora` | Voces Kokoro |
| `TUTOR_TTS_SPEED` | `0.95` | Velocidad del tutor |
| `TUTOR_LESSON_SECONDS` | `900` | Duración de la clase (los pasos son proporcionales) |
| `TUTOR_DATA_DIR` | `./data` | Base de datos SQLite del progreso |

## Licencias de los modelos

Silero VAD (MIT) · Smart Turn v3 (BSD-2) · Whisper / faster-whisper / mlx-whisper (MIT) · Kokoro-82M (Apache-2.0) ·
Gemma 3 (Gemma Terms of Use) · Qwen 3.5 (Apache-2.0).
