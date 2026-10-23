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

- **Windows 10/11, Linux (x86-64 o ARM64) o macOS**. Python 3.12 lo instala `uv` automáticamente.
- ~4,5 GB de disco para modelos. RAM medida del sistema completo durante una lección (backend + Qwen + Chrome):
  ~4,7–6,2 GB de pico estimados (medido con Gemma: 5,8–7,3 GB; Qwen ocupa ~1,1 GB menos). Recomendado **8 GB o más**. No hace falta tarjeta gráfica; si Ollama encuentra una, la usa.
- **Gratis y 100 % local** por defecto: el tutor es un modelo de lenguaje local (Qwen 3.5 4B) servido con
  [Ollama](https://ollama.com). Opcional: la API de Claude (de pago, aparte de la suscripción de Claude) para una
  conversación y una evaluación de más calidad. Sin ninguno de los dos arranca en **modo demo** (tutor guionizado).
- Chrome, Edge o Safari recientes. **Recomendado: auriculares** (evita que el micrófono capte al tutor).
- Reconocimiento de voz: en Mac con Apple Silicon usa Whisper large-v3-turbo (MLX); en Windows, Linux y Mac Intel
  usa Whisper small.en en CPU (algo menos preciso con acento español).

## Instalación paso a paso

Los pasos son los mismos en los tres sistemas: instalar **Git**, **uv** y **Ollama**, descargar el código, el
modelo del tutor y los modelos de voz, y arrancar. Todo se hace una sola vez salvo el último paso.

### Windows 10/11

Abre **PowerShell** (menú Inicio → escribe «PowerShell») y ejecuta, uno por uno:

```powershell
# 1. Herramientas (si winget no existe, instala «App Installer» desde Microsoft Store)
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Ollama.Ollama -e
```

**Cierra PowerShell y ábrelo de nuevo** para que reconozca los comandos nuevos. Después:

```powershell
# 2. Código
cd $HOME
git clone https://github.com/conororuiz/english_practice_assistence.git english-voice-tutor
cd english-voice-tutor

# 3. Modelo del tutor (3,3 GB)
ollama pull qwen3.5:4b

# 4. Dependencias de Python y modelos de voz (~830 MB)
uv sync
uv run python scripts/download_models.py

# 5. Arrancar
uv run python -m tutor.main
```

Notas de Windows:
- Ollama se instala como aplicación con icono en la bandeja y queda siempre en marcha; la app lo usa tal cual.
- La primera vez Windows puede preguntar por el **firewall**: basta con permitir el acceso en redes privadas
  (el servidor sólo escucha en `127.0.0.1`, tu propio equipo).
- Sin winget: descarga los instaladores de [git-scm.com](https://git-scm.com/download/win),
  [ollama.com/download](https://ollama.com/download) y, para uv, ejecuta en PowerShell
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.

### Linux (Ubuntu/Debian; en otras distros cambia `apt` por tu gestor)

```bash
# 1. Herramientas
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh         # uv
curl -fsSL https://ollama.com/install.sh | sh            # Ollama
source ~/.bashrc                                         # o abre una terminal nueva para tener `uv` en el PATH

# 2. Código
cd ~
git clone https://github.com/conororuiz/english_practice_assistence.git english-voice-tutor
cd english-voice-tutor

# 3. Modelo del tutor (3,3 GB)
ollama pull qwen3.5:4b

# 4. Dependencias de Python y modelos de voz (~830 MB)
uv sync
uv run python scripts/download_models.py

# 5. Arrancar
uv run python -m tutor.main
```

Notas de Linux:
- El instalador de Ollama lo deja como servicio de systemd siempre activo. Si prefieres que sólo funcione mientras
  usas la app: `sudo systemctl disable --now ollama` (la app lo arrancará y parará sola).
- Si `ollama pull` dice que no puede conectar, arranca el servicio: `sudo systemctl start ollama`.

### macOS

```bash
# 1. Herramientas (con Homebrew: https://brew.sh)
brew install git uv ollama

# 2. Código
cd ~
git clone https://github.com/conororuiz/english_practice_assistence.git english-voice-tutor
cd english-voice-tutor

# 3. Modelo del tutor (3,3 GB). Ollama tiene que estar en marcha para descargar:
ollama serve &                    # o abre la app Ollama
ollama pull qwen3.5:4b

# 4. Dependencias de Python y modelos de voz (~830 MB; +464 MB en Apple Silicon)
uv sync
uv run python scripts/download_models.py

# 5. Arrancar
uv run python -m tutor.main
```

En macOS, si Ollama no estaba en marcha, la app lo arranca (con atención flash y caché en 8 bits para gastar menos
memoria) y, al cerrarla con Ctrl+C, descarga el modelo y lo detiene. Si ya lo tenías en marcha, no lo para; sólo
libera el modelo de la memoria. (Lo mismo vale para Windows y Linux si Ollama no está ya funcionando.)

### Usar la app

Con la app arrancada, abre **http://127.0.0.1:8765** en Chrome o Edge, pulsa **Empezar clase**, acepta el permiso
del micrófono y habla. Para cerrarla, `Ctrl+C` en la terminal. Las siguientes veces sólo hace falta:

```bash
cd ~/english-voice-tutor          # Windows: cd $HOME\english-voice-tutor
uv run python -m tutor.main
```

`CC` muestra/oculta los subtítulos; **Alt+D** muestra el panel de detección de turno (completitud, silencio
requerido, multiplicador de paciencia). La pestaña **Progreso** muestra tu ruta (unidades dominadas, la actual con
su % de dominio), nivel, errores recurrentes, vocabulario, gramática y la próxima clase.

Para actualizar a la última versión: `git pull`, `uv sync` y `uv run python scripts/download_models.py`.

### Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `uv`, `ollama` o `git` «no se reconoce como comando» | Cierra y vuelve a abrir la terminal tras instalarlos |
| La app arranca en **modo demo** | Ollama no responde o falta el modelo: `ollama list` debe mostrar `qwen3.5:4b` |
| `Faltan modelos: …` | Ejecuta `uv run python scripts/download_models.py` (se puede repetir; reanuda lo que falte) |
| El tutor no te oye | Permiso de micrófono del navegador (icono del candado) y del sistema; usa `http://127.0.0.1:8765`, no la IP |
| El tutor se interrumpe a sí mismo | Usa auriculares |
| Va lento o se queda sin memoria | Cierra otras apps o usa `TUTOR_OLLAMA_MODEL=qwen3.5:4b` (ver Configuración) |

## Pruebas

```bash
uv run pytest -m "not models"                 # rápidas (lógica): detector, aprendizaje, planner…
uv run pytest                                 # + lección completa con los modelos reales (~75 s)
uv run python scripts/eval_turns.py           # banco de fin de turno (cortes prematuros, latencia)
uv run python scripts/eval_turns.py --wav mi_grabacion.wav --end 7.4   # con una grabación tuya
uv run python scripts/measure_memory.py       # RAM con modelos y pico bajo carga
uv run python scripts/compare_llm.py gemma3:4b qwen3.5:4b   # compara modelos locales como tutor
uv run python scripts/simulate_class.py --classes 3   # clases completas sólo texto: el modelo local de profesor y de alumno
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
scripts/      descarga de modelos (download_models.py), banco de turnos, medición de RAM, prueba WebSocket
docs/         ARCHITECTURE.md
```

## Configuración (`.env`)

Opcional. Copia `.env.example` a `.env` (Windows: `copy .env.example .env`; Linux/macOS: `cp .env.example .env`)
y descomenta lo que quieras cambiar.

| Variable | Por defecto | |
|---|---|---|
| `TUTOR_LLM` | `auto` | `auto` = Ollama si está; si no, Claude si hay clave; si no, demo. O fuerza `ollama` / `anthropic` / `mock` |
| `TUTOR_OLLAMA_MODEL` | `qwen3.5:4b` | Modelo local (alternativa: `gemma3:4b`, algo más rápido, explica peor) |
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
