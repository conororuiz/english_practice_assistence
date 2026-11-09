# Tutor de polaco — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que la misma app del tutor de inglés dé también clases de polaco (A1–A2 → B1, explicaciones en español), con un modelo propio «tutor-ola», funcionando en Linux y macOS.

**Architecture:** Lo que depende del idioma pasa a «paquetes de idioma» (`tutor/lang/en`, `tutor/lang/pl`) que la sesión recibe como objeto `LanguagePack`; el resto del código es común. Progreso en un fichero de BD por idioma. Voz: Whisper multilingüe con `language`, Piper para polaco + Kokoro para el español. Correcciones: reglas propias + LanguageTool + modelo afinado.

**Tech Stack:** Python 3.12 / FastAPI / uv, Ollama, faster-whisper (CUDA/CPU) y mlx-whisper, piper-tts, kokoro-onnx, language_tool_python (Java), PyTorch + PEFT para entrenar en el PC (`training/scripts/train_torch.py`), llama.cpp para GGUF.

**Spec:** `docs/superpowers/specs/2026-11-09-polish-tutor-design.md`

## Global Constraints

- Coste cero: sólo modelos y herramientas locales; nada de API de pago.
- Linux y macOS (Apple Silicon **e** Intel), sin suponer nada instalado más allá del README; la app se adapta sola (GPU, Java, modelos presentes o no).
- El tutor de inglés no cambia de comportamiento: los tests existentes (`uv run pytest -q`, 107 al empezar) pasan tras cada tarea.
- Explicaciones al alumno en español; ≤70 palabras en el análisis (coach).
- Voz polaca por defecto `gosia` (`TUTOR_VOICE_PL`); voz española `ef_dora` (Kokoro).
- Modelo del profesor de polaco: `tutor-ola` si está en Ollama; si no, `qwen3.5:9b` con aviso en el log (`TUTOR_OLLAMA_MODEL_PL` lo fuerza).
- Criterio de aceptación de tutor-ola en el test de polaco: falsos errores ≤ 5 % y precisión de correcciones ≥ 90 %.
- El GGUF de tutor-ola nunca se publica; se pasa a otros equipos a mano (AirDrop). Pesos y GGUF fuera de git.
- Rama de trabajo `tutor-polaco`; commits pequeños por tarea.

## Review Focus

1. **Frase polaca correcta → ningún error.** Lo peor que puede pasar es «corregir» lo que está bien (Task 6: test con 30 frases correctas).
2. **El alumno habla en español en mitad de la clase** («no entiendo», «¿cómo se dice…?») → se trata como petición de ayuda, no como error de polaco (Task 6).
3. **Clase de inglés después de una de polaco en el mismo proceso** → usa paquete, voz, BD y modelo de inglés (Task 8: test e2e en secuencia).
4. **Equipo sin GPU, sin Java o sin tutor-ola** → la app arranca en polaco con lo que haya y lo dice en el log, no se cae (Tasks 3, 6, 12).
5. **Whisper devuelve mayúsculas, puntuación o una palabra sin diacríticos** («zolty») → las reglas no se rompen ni marcan errores falsos por ello (Task 6).

---

### Task 0: Preparar la rama

- [ ] **Step 1:** Los cambios sin commitear de `main` (tanda 3, pausa, STT automático) se commitean primero **con el visto bueno del usuario**; luego `git checkout -b tutor-polaco`.
- [ ] **Step 2:** `uv run pytest -q` → 107 passed. Si no, parar.

### Task 1: LanguagePack y paquete de inglés

Mover el contenido del inglés a `tutor/lang/en/` **sin cambiar ni una línea de contenido**, y que el código común lo reciba por un objeto.

**Files:**
- Create: `tutor/lang/__init__.py`, `tutor/lang/en/__init__.py`
- Move (git mv) a `tutor/lang/en/`: `learning/route.py`, `learning/curriculum.py`, `learning/roleplays.py`, `learning/pronunciation.py`, `ai/prompts.py`, `ai/compact.py`; las reglas del inglés de `ai/evaluator.py` (`rule_based_evaluation`, `third_person`, `_PAST`, `_PP`, `_CONTRACTIONS`, `_PRONOUNS`, `_HELP_REQUEST`, `_WORD_REQUEST`, `word_request`, `asks_for_help`) → `tutor/lang/en/rules.py`; los prompts de `ai/helper.py` y `ai/facts.py` → `tutor/lang/en/prompts.py`.
- Las rutas antiguas quedan como reexportaciones de una línea (`from tutor.lang.en.route import *  # noqa`) para que tests y scripts sigan funcionando.
- Modify: `ai/evaluator.py`, `ai/coach.py`, `ai/conversation.py`, `ai/helper.py`, `ai/facts.py`, `ai/ollama_llm.py`, `learning/engine.py`, `learning/planner.py`, `learning/memory.py`, `learning/fluency.py`, `session/lesson.py`, `main.py`: en vez de importar constantes del inglés, usan `pack.<campo>`.
- Test: `tests/test_lang_pack.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class LanguagePack:
      code: str                    # "en" | "pl"
      name_es: str                 # "Inglés" | "Polaco"
      whisper_lang: str            # "en" | "pl"
      route: tuple[Unit, ...]; route_by_key: dict[str, Unit]; talk: ...   # lo que hoy exporta route.py
      grammar: dict; scenarios: tuple; scenarios_by_key: dict             # curriculum.py
      roleplays: tuple; roleplays_by_key: dict                            # roleplays.py
      filler_words: tuple[str, ...]
      prompts: ModuleType          # mismas funciones/constantes que hoy tiene ai/prompts.py + las de helper/facts
      compact: ModuleType          # mismas funciones que ai/compact.py
      rules: ModuleType            # rule_based_evaluation(text, plan) -> TurnEvaluation, word_request, asks_for_help
      pron_tip: Callable[[str], str]; candidate_word: Callable[[str], str]
      tuned_model_prefix: str      # "tutor-sam" | "tutor-ola" → prompts compactos
  def get_pack(code: str) -> LanguagePack   # ValueError si no existe
  ```
- Constructores que pasan a recibir `pack: LanguagePack` (último parámetro, por defecto `get_pack("en")` para no romper llamadas existentes): `Evaluator`, `Coach`, `ConversationAgent`, `Helper`, `FactExtractor`, `OllamaLLM` (decide `compact` con `pack.tuned_model_prefix`), `LearningEngine`, `Planner`, `UserMemory`, `LessonSession`.

- [ ] **Step 1: Test que falla** — `tests/test_lang_pack.py`:
  ```python
  def test_en_pack_is_current_content():
      from tutor.learning.route import ROUTE
      p = get_pack("en")
      assert p.route == ROUTE and p.whisper_lang == "en" and p.tuned_model_prefix == "tutor-sam"
  def test_unknown_pack():
      with pytest.raises(ValueError): get_pack("xx")
  def test_pack_prompt_api_matches():   # se reutiliza en Task 7 para pl
      for name in PROMPT_API: assert callable(getattr(get_pack("en").prompts, name)) or isinstance(getattr(get_pack("en").prompts, name), str)
  ```
  `PROMPT_API` = lista en `tutor/lang/__init__.py` con los nombres que hoy importa el código común (`tutor_system`, `phase_instruction`, `teach_steps`, `drill_note`, `produce_reminder`, `evaluator_prompt`, `EVALUATOR_SYSTEM`, `coach_prompt`, `COACH_SYSTEM`, `summary_prompt`, `SUMMARY_SYSTEM`, `help_prompt`, y los de facts).
- [ ] **Step 2:** `uv run pytest tests/test_lang_pack.py -v` → FAIL (no existe `tutor.lang`).
- [ ] **Step 3:** Crear el paquete, mover con `git mv`, reexportar y pasar `pack` por los constructores.
- [ ] **Step 4:** `uv run pytest -q` → todo verde (107 + nuevos). `scripts/simulate_class.py --replay 1` produce la misma clase que antes (comparar a ojo las primeras 5 respuestas con `TUTOR_LLM=mock`).
- [ ] **Step 5:** Commit `refactor: paquetes de idioma (inglés sin cambios)`.

### Task 2: Ajustes, BD y modelo por idioma

**Files:** Modify `tutor/config.py`, `tutor/main.py`, `tutor/session/models.py` · Test `tests/test_lang_settings.py`

**Interfaces:**
- `Settings.lang: str = _env("TUTOR_LANG", "en")` (idioma por defecto del selector).
- `Settings.db_path_for(lang: str) -> Path`: `en` → `data_dir/"tutor.db"` (igual que hoy), otro → `data_dir/f"tutor_{lang}.db"`. `db_path` se mantiene como `db_path_for("en")`.
- `Settings.ollama_model_pl: str = _env("TUTOR_OLLAMA_MODEL_PL", "auto")`, `Settings.voice_pl: str = _env("TUTOR_VOICE_PL", "gosia")`.
- `choose_ollama_model(lang: str, installed: list[str], settings) -> str` en `tutor/ai/ollama_llm.py`: `en` → `settings.ollama_model`; `pl` → el forzado si no es `auto`, si no `tutor-ola` si está en `installed` (con o sin `:latest`), si no `qwen3.5:9b` + `log.warning`.
- `main.py`: `state.dbs: dict[str, Database]` creado al vuelo; `state.llms: dict[str, LLM]`; el mensaje WS `{"type":"start", "lang": "pl", ...}` elige pack, BD y LLM; los endpoints `/api/today`, `/api/progress`, `/api/roleplays`, `/api/report` aceptan `?lang=` (por defecto `settings.lang`).

- [ ] **Step 1: Tests que fallan:** `test_db_paths` (`db_path_for("en").name == "tutor.db"`, `db_path_for("pl").name == "tutor_pl.db"`); `test_choose_model` (pl + `["tutor-ola:latest"]` → `"tutor-ola"`; pl + `[]` → `"qwen3.5:9b"`; pl forzado `"gemma3:4b"` → `"gemma3:4b"`; en → `settings.ollama_model`).
- [ ] **Step 2:** Ejecutar → FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** `uv run pytest -q` → verde.
- [ ] **Step 5:** Commit `feat: idioma por sesión, BD y modelo por idioma`.

### Task 3: Reconocimiento de voz con idioma

**Files:** Modify `tutor/audio/stt.py`, `tutor/session/models.py`, `scripts/download_models.py` · Test `tests/test_stt_lang.py`

**Interfaces:**
- `SpeechToText.transcribe(audio, context="", language="en")` y lo mismo en `MLXSpeechToText`; el calentamiento usa el idioma del modelo.
- `CT2_SMALL_MULTI = "faster-whisper-small"`.
- `resolve_stt(models_dir, name="auto", device="auto", lang="en") -> tuple[str, str]`: igual que hoy para `en`; para otro idioma nunca devuelve un modelo `.en`: Apple Silicon → MLX turbo; CUDA usable + turbo CT2 → turbo/cuda; si no → `faster-whisper-small`/cpu.
- `Models.stt_for(lang) -> SpeechToText | MLXSpeechToText`: carga perezosa y caché por nombre de modelo (turbo se comparte entre idiomas).
- `download_models.py` baja `Systran/faster-whisper-small` (multilingüe) en equipos sin Apple Silicon ni NVIDIA.

- [ ] **Step 1: Tests que fallan** (con `tmp_path` creando carpetas vacías y `monkeypatch` de `platform.system/machine` y `cuda_available`): `pl` en Linux sin CUDA → `("faster-whisper-small","cpu")`; `pl` con CUDA y turbo → `("faster-whisper-large-v3-turbo","cuda")`; `pl` en Mac Intel → `("faster-whisper-small","cpu")`; `pl` en Mac arm con MLX → `("whisper-large-v3-turbo-q4-mlx", ...)`; `en` mantiene los resultados actuales.
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde; además en el PC: transcribir `~/pl-recon/out/gosia_1.wav` con `stt_for("pl")` → «Wczoraj poszedłem do sklepu…».
- [ ] **Step 5:** Commit `feat: STT multilingüe por idioma`.

### Task 4: Voz polaca (Piper + Kokoro)

**Files:** Create `tutor/audio/tts_pl.py` · Modify `pyproject.toml` (`piper-tts`), `scripts/download_models.py`, `tutor/session/models.py` · Test `tests/test_tts_pl.py`

**Interfaces:**
- `class PolishTTS: __init__(piper_model: str, kokoro: Kokoro, voice_es: str, speed: float = 1.0)`; `synthesize(text: str, speed: float | None = None) -> np.ndarray` float32 mono a `OUTPUT_SR` (24 kHz). Spans `<es>` → Kokoro `lang="es"`; resto → Piper (`SynthesisConfig(length_scale=1/speed)`), remuestreo 22 050 → 24 000 con `numpy` (interpolación lineal) o `soxr` si ya está.
- `Models.tts_for(lang)`: `en` → `TextToSpeech` actual; `pl` → `PolishTTS` (comparte la instancia de Kokoro).
- `download_models.py`: `pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx(.json)` de `rhasspy/piper-voices` → `models/piper/`.

- [ ] **Step 1: Tests que fallan:** `test_split_pl_es` (texto «Dobrze! <es>Muy bien.</es> Dalej.» → 3 spans con idiomas pl/es/pl); `test_resample_length` (1 s a 22 050 Hz → 24 000 muestras ±1); `test_synthesize_real` (skip si falta el modelo): devuelve >0,5 s de audio y `speed=0.8` es más largo que `speed=1.0`.
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde.
- [ ] **Step 5:** Commit `feat: voz del profesor en polaco`.

### Task 5: Contenido polaco (ruta, situaciones, role-plays)

**Files:** Create `tutor/lang/pl/{__init__,route,curriculum,roleplays,pronunciation}.py` · Test `tests/test_pl_content.py`

Contenido escrito por Claude siguiendo §5.3 y §5.4 del diseño, con la misma estructura `Unit`/`Scenario`/role-play que el inglés. Muletillas: `("yyy","eee","eem","no","wiesz","jakby","znaczy","w sensie","po prostu")`. Pronunciación: consejos en español para sz/ś/si, cz/ć/ci, rz/ż, dz/dź/dż, ą/ę, y, ł, w final ensordecida.

**Interfaces:** `get_pack("pl")` devuelve un `LanguagePack` completo (prompts y reglas pueden ser provisionales hasta Tasks 6–7: usar los mínimos para que la app arranque).

- [ ] **Step 1: Tests que fallan** (parametrizados por unidad): 20 unidades, claves únicas; cada una con 8 términos de vocabulario, ≥3 ejemplos, ≥8 ejercicios, ≥3 pares `watch` con mal≠bien; `re.compile(detect)` coincide con ≥1 ejemplo de la unidad y con 0 de las frases `watch` incorrectas cuando la unidad es de forma (ver nota del test); todos los `scenarios` de las unidades existen en `scenarios_by_key`; ≥14 situaciones de la vida diaria (lista del diseño) y ≥8 role-plays con objetivos.
- [ ] **Step 2:** FAIL. **Step 3:** Escribir el contenido. **Step 4:** verde.
- [ ] **Step 5:** Commit `feat: ruta y situaciones de polaco`.

### Task 6: Correcciones en polaco (reglas + LanguageTool)

**Files:** Create `tutor/lang/pl/rules.py`, `tutor/lang/pl/languagetool.py`, `scripts/get_java.py` · Modify `pyproject.toml` (`language_tool_python`), `scripts/download_models.py` (llama a `get_java.py` si no hay `java`) · Test `tests/test_pl_rules.py`

**Interfaces:**
- `rules.rule_based_evaluation(text: str, plan: LessonPlan | None) -> TurnEvaluation` (mismo tipo que el inglés): reglas propias + `LanguageToolChecker.check(text)`.
- Reglas propias mínimas: pronombre sujeto redundante (`ja idę` → `idę`, severidad minor), `wczoraj` + verbo en presente, `od` + `dwa|trzy|cztery` + nominativo (`od dwa lata` → `od dwóch lat`), `z` + nominativo femenino en -a (`z moja koleżanka` → `z moją koleżanką`), `do` + nominativo de lugares frecuentes (`do kino` → `do kina`).
- `asks_for_help(text)` y `word_request(text)` reconocen español y polaco («no entiendo», «¿cómo se dice X?», «nie rozumiem», «jak się mówi…», «powtórz»).
- `class LanguageToolChecker`: arranca perezoso un servidor local (`language_tool_python.LanguageTool("pl-PL")`, Java de `models/jre/` si existe o del PATH); sólo acepta categorías `GRAMMAR` y `CASING` no; descarta reglas de ortografía/estilo/puntuación; si falla el arranque → `available=False`, `log.warning` una vez, `check()` devuelve `[]`.
- `scripts/get_java.py`: si `java -version` falla, descarga Temurin JRE 21 de la API de Adoptium para SO/arquitectura (`linux|mac` × `x64|aarch64`) a `models/jre/`. Sin sudo.

- [ ] **Step 1: Tests que fallan:**
  - Errores esperados (con LT desactivado, sólo reglas propias): las 5 frases de arriba → el `original` y `correction` exactos indicados.
  - **Falsos positivos (Review Focus 1, 5):** 30 frases correctas de las 20 unidades, más versiones en mayúsculas, sin puntuación y con «zolty» → `errors == []`.
  - **Español en clase (Review Focus 2):** «no entiendo», «¿cómo se dice ventana?» → `asks_for_help` / `word_request == "ventana"` y sin errores.
  - **Sin Java (Review Focus 4):** con `monkeypatch` haciendo fallar el arranque → `check()` devuelve `[]` y no lanza.
  - Con LT disponible (skip si no): «Chciałbym zamówić jedna kawa» → algún error que cubra «jedna kawa».
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde en Mac (con Java) y en el PC (tras `get_java.py`).
- [ ] **Step 5:** Commit `feat: correcciones de polaco con reglas y LanguageTool`.

### Task 7: Prompts de polaco (completos y compactos)

**Files:** Create `tutor/lang/pl/prompts.py`, `tutor/lang/pl/compact.py` · Test `tests/test_pl_prompts.py`

Mismas funciones que `PROMPT_API`, reescritas para polaco: profesora «Ola», respuestas en polaco de 1–2 frases sencillas con una sola pregunta, recast natural de los errores, `<es>…</es>` para lo que se dice en español; evaluador y coach con explicación en español; ayuda («ayúdame a responder») con inicios de frase en polaco; traducción al español. `compact.py` con la misma API que el del inglés (prompts cortos con los que se entrenará tutor-ola).

- [ ] **Step 1: Tests que fallan:** `test_pack_prompt_api_matches` parametrizado con `"pl"`; `tutor_system(plan_pl, snapshot)` contiene el título de la unidad y «po polsku»; `evaluator_prompt(...)` pide JSON y explicación «en español»; `compact_tutor_system(tutor_system(...))` tiene < 1/3 de tokens (palabras) del completo.
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde.
- [ ] **Step 5:** Commit `feat: prompts del profesor de polaco`.

### Task 8: Interfaz con selector de idioma

**Files:** Modify `web/index.html`, `web/app.js`, `web/style.css`, `tutor/main.py` · Test `tests/test_lang_web.py`

- Selector «🇬🇧 Inglés / 🇵🇱 Polaco» en la cabecera; se guarda en `localStorage` (con try/catch); recarga «Hoy», role-plays y Progreso con `?lang=`; el `start` del WS lleva `lang`. Textos fijos en español sin cambios; los que nombraban «inglés» pasan a usar `name_es` del paquete (`/api/today` devuelve `lang` y `lang_name`).

- [ ] **Step 1: Tests que fallan** (TestClient de FastAPI, `TUTOR_LLM=mock`, modelos de voz falsos como en `test_session_e2e.py`): `/api/today?lang=pl` → unidad 1 de polaco; **secuencia (Review Focus 3):** clase `pl` y después `en` en el mismo cliente → la segunda usa `ROUTE` del inglés, `tutor.db` y TTS de Kokoro inglés (comprobar el pack y la BD en la sesión creada).
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde.
- [ ] **Step 5:** Commit `feat: selector de idioma en la web`.

### Task 9: Clase de polaco jugable con qwen3.5:9b

- [ ] **Step 1:** `scripts/simulate_class.py --lang pl` (alumno simulado A2 hispanohablante con errores típicos, generado desde la ruta) en el PC con `qwen3.5:9b`: la clase de 15 min termina sin excepciones y recorre los 6 pasos.
- [ ] **Step 2:** Arrancar la app en el PC (`run_pc.sh`, túnel a `localhost`) y avisar al usuario para una prueba corta con micrófono en polaco; anotar en `docs/ARCHITECTURE.md` §19 lo que falle.
- [ ] **Step 3:** Commit `feat: simulación de clase de polaco`.

### Task 10: Datos de entrenamiento de tutor-ola

**Files:** Modify `training/scripts/collect.py`, `build_dataset.py`, `eval_student.py` (opción `--lang pl`) · Create `training/gold_pl/{chat,eval,coach,help,facts}_gold.py`, `training/scripts/check_gold_pl.py` · Test `tests/test_gold_pl.py`

- Entradas: `collect.py --lang pl` (clases simuladas con qwen3.5:9b; sólo se guardan las entradas) + frases de alumno escritas por error tipo.
- Respuestas: escritas por Claude en tandas de ~100 en `training/gold_pl/`, mismo formato que `training/gold/`. Volumen: ~700 chat, ~400 evaluación, ~250 coach, ~150 ayuda/datos personales. Conjunto de test aparte (`training/data/pl_test.jsonl`, 15 %, nunca en el entrenamiento).
- `check_gold_pl.py`: pasa cada `correction` por LanguageTool (y por `morfeusz2` si se instala en la plataforma) y lista las que LT marca o cuyo lema no existe; las marcadas se revisan a mano antes de entrenar.

- [ ] **Step 1: Test que falla:** `check_gold_pl` marca un ítem trampa («dwa piwo» como corrección) y no marca 10 correcciones válidas; todos los ítems tienen explicación en español (`is_spanish`) y ≤70 palabras en coach.
- [ ] **Step 2:** FAIL. **Step 3:** Implementar el comprobador y escribir las tandas. **Step 4:** verde y `check_gold_pl.py` sin marcas pendientes.
- [ ] **Step 5:** Commit `data: gold de polaco para tutor-ola` (sólo los .py del gold y scripts; nada de pesos).

### Task 11: Entrenar y evaluar tutor-ola (PC)

- [ ] **Step 1:** `build_dataset.py --lang pl` → `training/data/pl_train.jsonl` / `pl_valid.jsonl`.
- [ ] **Step 2:** `train_torch.py` con base `Qwen3.5-4B`, adaptador en `training/adapters/tutor-ola-4b` (mismos hiperparámetros que tutor-sam como punto de partida).
- [ ] **Step 3:** `make_ollama_linux.sh training/adapters/tutor-ola-4b tutor-ola` (ya incrusta la plantilla Jinja) → `training/gguf/tutor-ola-q4_k_m.gguf`.
- [ ] **Step 4:** `eval_student.py --lang pl --models qwen3.5:9b tutor-ola --out training/data/eval_pl.json`. Aceptar si falsos errores ≤ 5 % y precisión ≥ 90 %. Si no: repetir con Qwen3.5-9B QLoRA (`train_torch.py --qlora`, nueva opción) y volver a evaluar; si tampoco, parar y avisar al usuario con los números.
- [ ] **Step 5:** Commit `feat: tutor-ola` (scripts y resultados de evaluación; el GGUF no).

### Task 12: Instalar tutor-ola en otro equipo

**Files:** Create `scripts/install_model.py`, `training/ollama/Modelfile.ola` · Test `tests/test_install_model.py`

**Interfaces:** `find_gguf(arg: str | None) -> Path | None` (ruta dada o el `tutor-ola*.gguf` más reciente de `~/Downloads`); `main()` escribe un Modelfile temporal con `FROM <gguf>` + parámetros de `Modelfile.ola` y ejecuta `ollama create tutor-ola`.

- [ ] **Step 1: Tests que fallan:** `find_gguf` con ruta explícita, con `~/Downloads` simulado (`monkeypatch` de `HOME`) y sin nada → `None` con mensaje claro; **(Review Focus 4)** con Ollama sin `tutor-ola`, `choose_ollama_model("pl", [...])` → `qwen3.5:9b`.
- [ ] **Step 2:** FAIL. **Step 3:** Implementar. **Step 4:** verde; en el Mac de desarrollo, instalar desde una copia del GGUF y dar una clase simulada con `tutor-ola`.
- [ ] **Step 5:** Commit `feat: instalador de tutor-ola`.

### Task 13: Documentación y prueba en el Mac de destino

- [ ] **Step 1:** README: sección «Polaco» (selector, `download_models.py`, Java automático, copiar el GGUF por AirDrop y `uv run python scripts/install_model.py`), variables `TUTOR_LANG`, `TUTOR_OLLAMA_MODEL_PL`, `TUTOR_VOICE_PL`. `docs/ARCHITECTURE.md` §19 con el diseño final y los números de la evaluación.
- [ ] **Step 2:** Lista de comprobación para el usuario en el otro Mac (instalación desde cero siguiendo el README, una clase de polaco y una de inglés, mirar el log: qué Whisper, qué voz, si LanguageTool arrancó, qué modelo). Ajustar lo que falle.
- [ ] **Step 3:** Commit `docs: tutor de polaco`.
