# Diseño técnico — English Voice Tutor (MVP)

Documento de diseño: análisis de requisitos, decisiones, arquitectura y cómo se validó cada pieza.
Las cifras de rendimiento son **medidas** en un MacBook con Apple M4 (16 GB), salvo donde se indique.

---

> **Actualización (modo gratuito por defecto).** El razonamiento ya no requiere la API de pago: por defecto
> se usa un modelo local, **Qwen 3.5 4B** vía Ollama, y todo el sistema funciona sin coste y sin conexión. La
> API de Claude sigue disponible como opción de más calidad (`TUTOR_LLM=anthropic`). Ver §12.

## 1. Requisitos y ambigüedades resueltas

| Ambigüedad | Decisión | Motivo |
|---|---|---|
| ¿Plataforma? | App web local: servidor Python + interfaz en el navegador | Sin instalación de apps; el navegador da cancelación de eco y micrófono gratis; el mismo backend sirve para móvil más adelante. |
| ¿Todo local o con API? | **Híbrido**: voz (VAD, STT, detección de turno, TTS) local; razonamiento (conversación, evaluación, resumen) con Claude por API | La calidad pedagógica y conversacional de un LLM grande no cabe en 8 GB; la voz local da latencia baja y privacidad del audio (el audio nunca sale del equipo). |
| ¿Un usuario o varios? | Un usuario local en el MVP; esquema multiusuario (`user_id` en todas las tablas) | No construir cuentas antes de validar el núcleo. |
| ¿"10 minutos" exactos? | Reloj de pared con fases proporcionales; cierre suave a partir de 9:00, cierre proactivo a 9:30 si el usuario está callado, corte duro a 10:25 | Nunca se corta al usuario a media frase, pero la lección no se alarga. |
| ¿Corregir todo? | Política explícita: máx. 1 corrección por turno, no en turnos consecutivos, sólo errores *major*; los *minor* se guardan para la práctica y el resumen | Prioridad pedida: naturalidad > fluidez > comprensión > corrección > gramática perfecta. |
| ¿Pronunciación? | Sólo señal experimental: palabras con baja confianza del reconocedor (`unclear_words`) | Evaluar pronunciación de verdad requiere modelos de *goodness of pronunciation* con alineamiento fonético; no es fiable con lo disponible y no se presenta como diagnóstico. |
| ¿Nivel B1? | El usuario empieza en B1 (`level_score = 0`), nunca desde cero; el nivel se reestima con una media móvil lenta | Lo indica el enunciado; evita oscilaciones. |

---

## 2. Arquitectura

```
 Navegador (UI mínima)                         Servidor local (Python, FastAPI)                       Remoto
 ───────────────────────                       ──────────────────────────────────────────────         ──────────
 Micrófono + AEC/NS/AGC                        LessonSession (una por lección)
 AudioWorklet → PCM 16 kHz, 32 ms ──WS bin──▶  ├─ Silero VAD (por trama, en el event loop)
                                               ├─ TurnDetector (lógica pura)  ◀── PatienceProfile (aprendido, SQLite)
                                               ├─ [pool STT 1 hilo] Whisper large-v3-turbo (MLX) + Smart Turn v3.2
                                               ├─ ConversationAgent ───────── streaming ───────────▶  Claude (conversación)
                                               │     frases + etiquetas silenciosas (<es>, <vocab>, <elicit>, <fix>)
                                               ├─ [pool TTS 1 hilo] Kokoro-82M (voz EN + voz ES)
 Reproductor en cola (24 kHz) ◀──WS bin──────  ├─ Evaluator (en paralelo, no bloquea) ─────────────▶  Claude (salida estructurada)
 playback_done ─────────────WS json────────▶   │     respaldo: reglas locales
                                               └─ al terminar: LearningEngine → Summarizer ────────▶  Claude (resumen)
                                               UserMemory / LessonPlanner / SQLite
```

### Separación lógica de responsabilidades (punto 19)

| Responsabilidad | Módulo | Implementación |
|---|---|---|
| Conversational AI | `tutor/ai/conversation.py`, `prompts.py` | Claude (streaming). System prompt estable por lección (cacheado) + notas de sistema por turno. |
| Speech system | `tutor/audio/` | Silero VAD, Whisper large-v3-turbo (MLX; small.en fuera de Apple Silicon), Smart Turn v3.2, Kokoro-82M — todo local. |
| Evaluation system | `tutor/ai/evaluator.py` | Claude con salida estructurada (`TurnEvaluation`); reglas locales de respaldo. Métricas acústicas deterministas en `learning/fluency.py`. |
| Learning engine | `tutor/learning/engine.py`, `srs.py` | Determinista: dominio gramatical, repetición espaciada, errores recurrentes, nivel. |
| User memory | `tutor/learning/memory.py`, `db.py` | SQLite. |
| Lesson planner | `tutor/learning/planner.py`, `curriculum.py` | Determinista: arco semanal + adaptación al perfil. |

El LLM sólo hace lo que necesita lenguaje (conversar, juzgar errores, redactar el resumen). Todo lo que decide el
progreso (qué repasar, cuándo, qué nivel) es código determinista y testeable.

### Protocolo cliente-servidor (un único WebSocket)
- Cliente → servidor: binario = PCM Int16 16 kHz mono; JSON = `start`, `playback_done{reply_id}`, `end`.
- Servidor → cliente: binario = `uint32 reply_id` + PCM Int16 24 kHz; JSON = `state`, `clock`, `tutor_text`,
  `user_partial`, `user_final`, `reply_end`, `stop_audio`, `turn_eval`, `turn_debug`, `summary`.
- `reply_id` permite descartar audio obsoleto tras una interrupción.

---

## 3. Flujo de una clase de 15 minutos (desde 2026-10-08)

> Antes (hasta la revisión del §13) la lección era una conversación libre de 10 minutos con la gramática "provocada".
> Ahora es una **clase guiada** con un tema concreto de la ruta de aprendizaje (§14). Lo anterior se mantiene: detección
> de turno, correcciones dosificadas, repetición espaciada, errores recurrentes, métricas, resumen.

| Tiempo (máx.) | Paso | Qué hace el profesor |
|---|---|---|
| 0:00–1:03 | warmup | Saludo; si hay una unidad dominada pendiente de repaso, una pregunta rápida de ella. Termina tras 1–2 intercambios. |
| …–4:03 | teach | Presentación en 2–3 turnos que da el orquestador: tema y objetivo, regla en español (`explain_es`) y ejemplos; después las palabras nuevas de dos en dos (significado en español + ejemplo con la estructura + pregunta para usarlas). En clases 2+ de la unidad: repaso de la regla en una frase y el foco de hoy. |
| …–8:15 | drill | Ejercicios guiados de la unidad, **uno por turno**, en orden (primero los que aún no resolvió bien). Si la respuesta falla la estructura, el profesor corrige y pide repetir una vez; luego pasa al siguiente. |
| …–12:18 | produce | Conversación real (o role-play) con las preguntas abiertas de la unidad (`TALK`), una a una con repreguntas; corrige brevemente sólo la estructura del día y errores importantes. |
| …–13:57 | review | Repaso: un ítem por turno (errores de hoy con la estructura, palabras sin usar); si vuelve a fallar, lo repite. |
| …–15:00 | wrapup | Cierre: qué hemos aprendido, ejercicios bien, qué practicar; el resumen y el progreso de la unidad salen en pantalla. |

Cada paso termina al cumplirse su objetivo (presentación dada, ejercicios agotados) o, como tarde, al tiempo de la
tabla; nunca se vuelve atrás. Garantías de duración: a 14:15 si el usuario no está hablando el profesor cierra él
mismo; a 15:25 corte duro con una despedida local. Las notas al modelo ya **no llevan el reloj**: con él a la vista,
Gemma 4B tendía a despedirse antes de tiempo.

---

## 4. Detección de final de turno (punto 5)

Implementación: `tutor/audio/turn_detector.py` (lógica pura) + `session/lesson.py` (orquestación).

**Señales combinadas**

1. **VAD** (Silero, 32 ms): inicio de voz = 96 ms de voz; pausa = 256 ms de silencio.
2. **Smart Turn v3.2** (8 MB, ~20 ms CPU): clasificador acústico entrenado para fin de turno; capta la
   **entonación** (tono descendente vs. de continuación, alargamientos).
3. **Texto parcial** (Whisper sobre el segmento recién terminado): muletilla final (`umm`, `let me think`) → 0.05;
   palabra que deja la frase abierta (`because`, `to`, `the`, `and`…) → 0.15; respuesta corta con auxiliar
   (`Yes, I have.`) → completa; frase con cierre y longitud razonable → ~0.8.
4. **Contexto**: si la pregunta del tutor pedía una respuesta larga (`Tell me…`, `Why…`) una respuesta de 1–4 palabras
   se considera menos terminada.
5. **Historial del usuario** (`PatienceProfile`, persistido): cuantiles p50/p90 de **sus** pausas internas (pausas
   tras las que siguió hablando), latencia típica de respuesta y un multiplicador que se corrige con feedback.
6. **Historial del turno**: si ya hizo una pausa de 2 s y siguió, no se cierra antes de ese margen salvo frase clara.
7. **Velocidad**: < 80 palabras/min → +15 % de espera.

**Ventana de paciencia**

```
fast = clamp(0.9·p50 + 0.35, 0.6, 1.6) · m      (frase claramente terminada)
slow = clamp(1.6·p90, 2.5, 7.0) · m             (frase dudosa)
c    = fusión(SmartTurn, texto)                  (el texto "abierto" veta la entonación)
silencio requerido = fast + (1 − c)^1.3 · (slow − fast)
```

Frase **claramente abierta** (muletilla o `because…` al final): no se cierra nunca por silencio normal; al llegar
al techo (1.5·slow ≤ 12 s) se emite `STUCK` y el tutor **ayuda** (arranque de frase, palabra, pregunta más simple)
en vez de contestar a media frase. El turno no se cierra nunca sin la transcripción del último segmento (con
número de secuencia para descartar análisis obsoletos — bug real encontrado en la prueba E2E).

**Aprendizaje progresivo**
- Cada pausa interna observada actualiza p50/p90 (cuantiles estocásticos).
- **Falso final** = el usuario sigue hablando antes de que suene el tutor (`RESUMED`: la respuesta se descarta y el
  turno sigue abierto) o interrumpe al tutor en sus 2 primeros segundos → multiplicador ×1.08.
- Esperas largas con frases claras → multiplicador ×0.985 (hasta 0.75). Empieza en ×1.15 (B1: paciente).

**Latencia sin prisas**
- `SPECULATE`: con c ≥ 0.7 se empieza a generar la respuesta (LLM + TTS) durante el propio silencio; si el usuario
  retoma, se cancela. El LLM queda "escondido" dentro de la ventana de paciencia.
- Ventana de gracia: si c < 0.85, el audio del tutor no empieza hasta 0.8 s después del fin detectado.
- Barge-in: voz clara (p ≥ 0.8) sostenida 256 ms mientras habla el tutor → se corta el audio y se cede la palabra;
  el historial guarda sólo lo que el usuario llegó a oír.

**Resultados medidos** (`scripts/eval_turns.py`, 10 escenarios: pausas de pensamiento, muletillas, autocorrección,
enumeraciones, habla lenta, condicional con pausas, respuestas cortas…, simulando en tiempo real el coste de
cómputo del análisis):

| Prosodia del alumno sintético | Cortes prematuros | Recuperados en la gracia | Latencia fin de turno (mediana / máx) |
|---|---|---|---|
| Natural (frase entera, pausas insertadas) | **0/10** | 2/10 | 1.54 s / 2.50 s |
| Pesimista (cada trozo con entonación final) | **0/10** | 4/10 | 1.39 s / 2.70 s |

"Recuperado" = el sistema creyó que había terminado pero el alumno siguió antes de que sonara el tutor: el turno se
reabre sin que el usuario lo note. La latencia percibida "fin de turno → voz del tutor" en la prueba WebSocket en
tiempo real fue de 0.64 s (mediana) con el tutor simulado; con Claude depende del tiempo hasta la primera frase, que
la especulación solapa con la espera.

---

## 5. Evaluación de errores (puntos 4 y 9)

- Por cada turno, **en paralelo** a la respuesta (nunca la retrasa): `TurnEvaluation` con errores
  (`original`, `corrected`, `category`, `skill_key`, `severity major|minor`, explicación corta en español),
  vocabulario objetivo usado, estructuras intentadas (correctas/incorrectas), vocabulario B1+/B2 bien usado,
  complejidad 1–5, si entendió la pregunta, y si el turno parece un fallo del reconocedor (se descarta).
- **major**: tiempo verbal / gramática nuclear, palabra que cambia el significado, error que impide entender, o que
  coincide con un error recurrente. **minor**: el resto.
- Uso en vivo: errores *major* que son recurrentes o de la gramática objetivo pasan a una cola de foco que el tutor
  puede "recastear" en un turno posterior (respetando el ritmo de correcciones). Alimentan la fase de práctica.
- Respaldo local (`rule_based_evaluation`): reglas para errores típicos de hispanohablantes (`I have went`,
  `yesterday I go`, `I am agree`, `depend of`, `make a meeting`, `I have 30 years`, `since 3 years`…). Se usa si la API
  falla y en modo demo.
- Patrones recurrentes: clave normalizada `original -> corrected` agregada entre sesiones (`errors.pattern_key`).

---

## 6. Ruta adaptativa, memoria y repetición espaciada (puntos 10–13, 16)

**Modelo de datos** (`tutor/learning/db.py`):

| Tabla | Contenido |
|---|---|
| `users` | nivel, `level_score`, `patience_json` (PatienceProfile), `profile_json` (fortalezas, debilidades, mejoras recientes) |
| `sessions` | plan con el que se ejecutó, métricas completas, resumen mostrado, estado |
| `turns` | cada intervención (usuario y tutor) con fase, tiempos, métricas acústicas y evaluación |
| `errors` | cada error con categoría, skill, severidad, corrección, explicación y `pattern_key` |
| `skills` | dominio 0–1 por estructura gramatical, totales, próxima revisión |
| `vocab` | palabra/expresión, significado, caja Leitner, estado, exposiciones, provocaciones, usos provocados/espontáneos, fallos |
| `vocab_events` | traza de eventos de vocabulario (introducida, provocada, usada…, fallo) |

**Planner** (antes de cada lección):
1. Arco semanal (5 lecciones/semana): conversación básica → vocabulario → respuestas largas → espontaneidad →
   situaciones profesionales → ideas complejas. Si las respuestas siguen siendo muy cortas, no avanza el foco.
2. Alterna cotidiano / trabajo (más trabajo en la semana profesional).
3. Escenario (24 en `curriculum.py`): puntúa dominio, dificultad según nivel, encaje con las estructuras débiles y
   evita los últimos 4.
4. Gramática objetivo: las 1–2 estructuras con menor dominio y revisión pendiente → se provocan en conversación
   (p. ej. present perfect → "Have you ever…?"). Sin datos: diagnóstico con la estructura del escenario.
5. Vocabulario: repaso de lo que toca según la caja (`reuse` / `elicit` / `check_spontaneous`) + 1–4 palabras nuevas.
6. Notas de adaptación: si una estructura cruzó de <0.7 a ≥0.7 de dominio con ≥2 usos correctos, el tutor lo dice y
   corrige menos ese tema; si hubo muchos atascos o pausas largas → corrección "light".

**Vocabulario**: una palabra no avanza por verla, sino por **usarla**. El tutor marca en silencio cuándo la presenta
(`<vocab>`) y cuándo intenta provocarla (`<elicit>`). El uso del alumno se clasifica como *provocado* (se pidió en
los 2 turnos previos o el tutor la acababa de decir) o *espontáneo*. Calendario: cajas con intervalos 1, 1, 3, 5, 10,
21, 45 días; uso provocado +1 caja, espontáneo +2, fallo (provocada y no usada, o mal usada) → caja 1. Dominada con
caja ≥5 y ≥2 usos espontáneos.

**Nivel**: rendimiento de la sesión = 50 % índice de fluidez + 30 % (1 − errores graves/100 palabras) + 20 %
complejidad; media móvil lenta (15 %) → B1 / B1+ / B2 / B2+ / C1. Heurístico y conservador; ver §10.

---

## 7. Métricas de fluidez (punto 14)

Calculadas sin IA a partir de los tiempos del VAD y la transcripción (`learning/fluency.py`), por turno y por sesión:
tiempo antes de responder (mediana), duración media de pausas internas, pausas ≥2 s por minuto, palabras/min de
articulación (sólo voz) y de habla (con pausas), palabras por respuesta (media y máx.), muletillas por 100 palabras,
diversidad léxica (MATTR), palabras sobreusadas, tiempo de habla y proporción sobre la lección, atascos (`STUCK`),
errores graves/leves por 100 palabras, complejidad media (LLM), uso de vocabulario objetivo y estructuras.
Un **índice de fluidez** 0–100 combina las principales para ver tendencias (se compara con la media de las 3
lecciones anteriores: "better / similar / lower").

---

## 8. Componentes con IA, local vs. remoto, y RAM (puntos 17–18)

| Componente | Modelo | Dónde | RAM aprox. |
|---|---|---|---|
| VAD | Silero VAD v5 (2 MB) | local | ~20 MB |
| Fin de turno acústico | Smart Turn v3.2 int8 (8 MB, Pipecat) | local | ~40 MB |
| Speech-to-Text | Whisper small.en int8 (CTranslate2 / faster-whisper) | local | ~600 MB |
| Text-to-Speech | Kokoro-82M fp32 ONNX + voces (EN `af_heart`, ES `ef_dora`) | local | ~500 MB |
| Conversación | Claude Opus 5.5, esfuerzo `low` | remoto (API) | — |
| Evaluación por turno | Claude Opus 5.5, salida estructurada | remoto (API) | — |
| Resumen | Claude Opus 5.5, esfuerzo `medium` | remoto (API) | — |
| Planner, SRS, métricas, nivel | código determinista | local | — |

**RAM medida** (`scripts/measure_memory.py`): backend con todos los modelos cargados **1.37 GB**; pico con STT de
30 s + Smart Turn + TTS concurrentes **1.84 GB**; servidor real tras una lección ~1.5 GB. Un navegador con la
pestaña de la app suele añadir 0.3–0.6 GB. Total < 2.5 GB, muy por debajo de 8 GB.

**Cómo se garantiza el límite**: modelos elegidos por tamaño (small.en int8, Kokoro 82M); un único hilo por modelo
pesado (nunca dos inferencias simultáneas del mismo modelo); una sola lección activa a la vez; carga única
compartida; `TUTOR_RAM_BUDGET_MB` (3.5 GB por defecto) con aviso en el arranque; sin modelos de lenguaje locales.
El audio de la lección no se guarda (sólo transcripciones).

**CPU**: Whisper ~0.3–0.6 s por segmento típico de 2–8 s (4 hilos); Kokoro ~0.25× tiempo real; Smart Turn ~20 ms;
VAD <1 ms por trama. No requiere GPU.

**¿Entrenar o ajustar modelos propios?** No en el MVP, a propósito:
- Fin de turno: Smart Turn v3.2 ya está entrenado específicamente para esto (incluye hablantes no nativos y
  muletillas) y, combinado con las señales de texto y la paciencia aprendida por usuario, da 0 cortes prematuros
  en el banco de pruebas. El ajuste por usuario se hace con estadística online, no con reentrenamiento.
- Clasificación de errores: un LLM grande con salida estructurada es claramente mejor que cualquier clasificador
  pequeño entrenable sin datos etiquetados.
- **Cuándo sí** (fase 2, con datos reales): fine-tuning de Smart Turn (8M parámetros, cabeza lineal sobre
  Whisper-tiny) con grabaciones de hispanohablantes B1 etiquetadas automáticamente con los eventos `RESUMED` y
  barge-in tempranos (falsos finales) y los finales correctos. Coste: unas pocas horas en una GPU de consumo con
  el script de entrenamiento público de Pipecat; inferencia idéntica (8 MB, ~40 MB RAM); evaluación con el banco
  `eval_turns.py` sobre grabaciones reales retenidas (tasa de cortes prematuros y latencia de fin de turno).

**Seguridad**: el servidor escucha sólo en `127.0.0.1`; el WebSocket rechaza orígenes ajenos (evita
*cross-site WebSocket hijacking*); la clave de API sólo vive en el servidor (`.env`, ignorado por git); el audio no
sale del equipo; a Claude sólo se envían transcripciones.

**Escalabilidad (más adelante)**: para muchos usuarios, mover STT/TTS a un servicio con GPU (o a STT/TTS en
streaming gestionados), un proceso por sesión detrás de un balanceador con afinidad de WebSocket, SQLite → Postgres,
y caché de prompts por usuario. El diseño por componentes con interfaces pequeñas (`LLM`, `Transport`, `Models`)
permite sustituir piezas sin tocar el orquestador.

---

## 9. Cómo se ha probado

| Prueba | Qué verifica |
|---|---|
| `tests/test_turn_detector.py` (14) | Pausas de pensamiento no cierran el turno; muletilla → espera/ayuda; respuesta corta rápida; especulación; sin cierre sin transcripción; análisis obsoleto no cierra; reapertura en la gracia; barge-in sostenido; aprendizaje de p50/p90 y del multiplicador; aviso si no responde. |
| `tests/test_learning.py` (13) | Calendario SRS (día 1/2/5/10), reinicio por fallo, dominio gramatical, métricas, parser de etiquetas en streaming, separación EN/ES para TTS, evaluador local, flexiones de vocabulario, **ruta adaptativa** (4 lecciones con errores de present perfect → la 5ª lo trabaja), mejora → deja de ser foco, flujo de vocabulario entre lecciones. |
| `tests/test_session_e2e.py` (2, modelos reales) | Lección completa acelerada: las 4 fases, ≥3 turnos con transcripción real, errores detectados y guardados, vocabulario seguido, paciencia persistida, duración acotada; barge-in corta al tutor. |
| `scripts/eval_turns.py` | Banco de fin de turno con el pipeline real (ver §4). Acepta grabaciones reales (`--wav --end`). |
| `scripts/ws_smoke.py` | Servidor real por WebSocket en tiempo real, de principio a fin. |
| `scripts/browser_e2e.py` | La interfaz real en Chrome headless: AudioWorklet 48→16 kHz, WebSocket, reproducción Web Audio, `playback_done`, resumen en pantalla. En macOS headless `getUserMedia` se bloquea, así que el micrófono se sustituye por un stream Web Audio con un WAV. |
| `scripts/measure_memory.py` | RAM base, con modelos y pico bajo carga. |
| Petición a la API de Claude contra un servidor falso | Forma de las peticiones (streaming, mensajes de sistema intermedios, historial con bloques de pensamiento, salida estructurada, `fallbacks`). **No** se ha probado contra la API real (no había clave en este equipo). |
| No probado | Una lección con micrófono real y una persona, y con la API real de Claude (latencia real del LLM). |

---

## 10. Qué es MVP y qué viene después

**MVP (hecho)**: lección de 10 min por voz manos libres; detección de turno adaptativa; tutor con mezcla EN/ES
pedagógica y correcciones dosificadas; evaluación por turno; memoria, SRS, planner adaptativo; métricas de fluidez;
resumen final; vista de progreso; modo demo sin clave.

**Siguiente**:
1. Validar con usuarios reales: recoger (con consentimiento) eventos de turno y ajustar umbrales; banco con
   grabaciones reales de hispanohablantes.
2. Fine-tuning de Smart Turn con esos datos (ver §8).
3. Pronunciación real (alineamiento fonético + GOP) para sonidos problemáticos de hispanohablantes (/ɪ/–/iː/, /v/–/b/, -ed).
4. Calibrar la estimación de nivel contra una prueba externa (p. ej. evaluación CEFR con rúbrica en un subconjunto).
5. Streaming STT incremental durante el habla (reduciría ~0.3 s más de latencia) y TTS por cláusulas.
6. App móvil (mismo backend), recordatorios diarios, cuentas y sincronización.
7. Panel de progreso semanal/mensual con tendencias por métrica.

## 11. Cómo comprobar que mejora la fluidez con las semanas

- **Métricas longitudinales por lección** (ya almacenadas): tiempo antes de responder, pausas largas/min, palabras/min
  de habla, palabras por respuesta, muletillas/100 palabras, diversidad léxica, errores graves/100 palabras, usos
  espontáneos de vocabulario, atascos. Tendencia con regresión robusta por semana; controlar por dificultad del
  escenario (comparar sólo lecciones de la misma dificultad o incluirla como covariable).
- **Tarea de referencia fija** cada 2 semanas (p. ej. "describe tu trabajo / cuenta tu fin de semana" de 2 minutos,
  mismo prompt): comparación directa de las métricas sin efecto del tema.
- **Vocabulario activo**: nº de palabras que pasan a uso espontáneo y se mantienen ≥30 días.
- **Validación externa**: evaluaciones CEFR a ciegas (humanas o con rúbrica) de grabaciones de la tarea de referencia
  al inicio y cada mes; y autopercepción de confianza (1 pregunta semanal).
- Si las métricas se estancan, el planner ya ajusta foco (respuestas largas, corrección light), y el análisis por
  métrica indica qué componente revisar.

---

## 12. Modo local gratuito (Ollama)

**Por qué**: la API de Claude se factura aparte de la suscripción de Claude; el usuario pidió una solución
gratuita. Un modelo local de ~4B parámetros cabe en el límite de 8 GB junto con la voz.

**Elección del modelo** (`scripts/compare_llm.py`, mismos prompts y conversación de 8 turnos):

| | Gemma 3 4B | Qwen 3.5 4B |
|---|---|---|
| Latencia hasta la 1ª frase (media) | **1.0 s** | 1.5 s |
| Estilo | reformulación implícita, 1 corrección breve, natural | correcciones explícitas más largas, alguna explicación confusa |
| Errores detectados (con reglas) | 6/6 | 5/6 |
| Memoria del sistema ocupada (8k de contexto) | ~4.8–5.2 GB | ~3.8 GB |

Se eligió primero **Gemma 3 4B** por calidad conversacional y latencia. Desde el 2026-10-23 el modelo por defecto es
**Qwen 3.5 4B** (decisión del usuario, ver §15): explica la gramática mucho mejor en el análisis de respuestas y
ocupa ~1,1 GB menos; Gemma queda como alternativa (`TUTOR_OLLAMA_MODEL=gemma3:4b`).

**Memoria total medida** (lección completa en Chrome: backend + Ollama/Gemma + navegador, incremento de memoria
ocupada del sistema): **5.8–7.3 GB** de pico en dos ejecuciones → dentro de 8 GB, con poco margen. Con Qwen ~1.1 GB menos.
Ollama se configura con atención flash, caché KV en 8 bits, un modelo y una petición a la vez.

**Compensaciones para un modelo pequeño** (todas en código, independientes del proveedor):
- *Prioridad de la conversación*: un único modelo hace todo; si el tutor tiene que hablar mientras se evalúa un turno,
  la evaluación se cancela y se reintenta después (`OllamaLLM`), así nunca retrasa la voz.
- *Español sólo para enseñar*: `SpokenFilter` elimina los `<es>` que traducen preguntas o frases sin relación con
  una corrección o palabra nueva (máx. uno por respuesta) y el markdown.
- *Vocabulario sin depender de etiquetas*: se detecta en el texto del tutor si dijo una palabra objetivo
  (presentada) o la usó en una pregunta (provocada).
- *Evaluación*: el modelo local sólo busca errores (esquema reducido); cada error se verifica (el fragmento debe
  estar en lo que dijo el alumno, la corrección debe cambiar algo, no vale sólo cambiar el sujeto) y se une con las
  reglas locales. Estructuras usadas y complejidad, por reglas.
- *Resumen determinista*: con el modelo local el resumen se construye sólo con hechos medidos (un 4B se inventaba
  mejoras en la primera lección).
- *Whisper*: se recortan palabras finales de muy baja confianza y frases típicas de alucinación sobre ruido.

**Qué se pierde frente a Claude**: menos matiz pedagógico, explicaciones en español más pobres, evaluación menos
completa (p. ej. no siempre detecta "we talk" → "we talked") y sin resumen redactado. El diseño permite cambiar de
proveedor con una variable.

---

## 13. Revisión tras la primera lección real (2026-10-06)

Primera lección con una persona y micrófono real (grabada). Problemas encontrados y cambios:

| Problema observado | Causa | Cambio |
|---|---|---|
| Transcripciones erróneas con acento español ("my breath was sandwich", "Work Miration", "I understand the question" en vez de "I don't…") | small.en transcribiendo **trozo a trozo** (sin contexto) + prompt de muletillas | Whisper **large-v3-turbo 4 bits en MLX** (GPU, ~1.05 s, ~0.5 GB, igual que small.en en CPU). Se transcribe el **turno entero** hasta cada pausa (Whisper siempre procesa 30 s: cuesta lo mismo); los análisis obsoletos se saltan. Prompt = sólo la pregunta del tutor, sin comillas. |
| "¡Vamos!" ×80 en un turno (y métricas de fluidez absurdas: 632 ppm, "vamos" como palabra repetida) | Bucle de repetición de Whisper | `collapse_repetitions` + reintento con temperatura si el *compression ratio* > 2.2 |
| "Thank you." inventado al final de 3 turnos | Alucinación sobre el ruido de cola | Se quita un "Thank you/Thanks/Bye" final cuando va como frase aparte |
| Turnos marcados "atascado" y 7.5 s de espera del tutor | El "…thank you" inventado terminaba en "you" (palabra "abierta") → techo de espera + STUCK | `text_completeness`: frases de cierre ("thank you", "I don't know"…) y preguntas al tutor ("Can you repeat?") = turno completo |
| Al decir "I don't understand the question" el tutor respondió "That's great!" y siguió | Nada detectaba la petición de ayuda; el evaluador local fijaba `understood_question=True` | `asks_for_help()`: nota al tutor para reformular más simple (se permite traducir esa pregunta al español una vez), `understood_question=False`, y no se corrigen errores en esa petición |
| El tutor proponía palabras inventadas ("Perhaps you could say 'work management'") | Gemma "corregía" basura del reconocedor | Prompt: sólo corregir si está seguro; si algo suena raro, preguntar. La nota de palabras dudosas prohíbe sugerir reemplazos |
| El tutor traducía cada pregunta al español (aunque no sonaba) | Gemma ignora la regla y se ve a sí mismo haciéndolo en el historial | Al historial y a la BD va sólo lo que realmente sonó |
| Errores falsos: "I'm" → "I am" (major, duplicado), "breath was sandwich" → "breath was fresh", "Yes, I enjoyed it" → "I'm starting a new job", "I make migrations" → "I am making" | Evaluador local (Gemma 4B) poco fiable | Filtros: expandir contracciones no es error; no se juzgan palabras dudosas del ASR; la corrección debe parecerse a lo dicho (ratio ≥ 0.5); sin duplicados; word_choice/naturalness = minor; prompt con contraejemplos de tiempos verbales |
| La última frase ("I work. Can you repeat?") se perdió al pulsar Terminar; el resumen tardó ~25 s | El turno sin respuesta se descartaba; se esperaba hasta 25 s a evaluaciones en cola | El turno pendiente se guarda y evalúa; espera máx. 10 s y las que falten se evalúan con reglas |

Validación: la lección grabada se repite de extremo a extremo con los turnos reales, Gemma y tiempo real
(transcripciones correctas, el tutor reformula ante "I don't understand", fin de turno 1.4–2.7 s, tutor sonando
1.2–1.8 s después). Banco sintético (`scripts/eval_turns.py`): 0 cortes, mediana 1.66 s. Coste conocido: large-v3-turbo
"limpia" casi todas las muletillas, así que `fillers_per_100w` sale más bajo que con small.en.

## 14. Clases guiadas con ruta de aprendizaje (2026-10-08)

Petición: que la app se comporte como **un profesor que imparte clases** de 15 minutos, cada una de un tema concreto
(p. ej. presente simple con ciertas palabras), siguiendo una ruta, sin subir de nivel hasta dominar el tema, y que se
note desde el primer minuto. Se mantiene todo lo anterior (voz, turnos, correcciones dosificadas, SRS, errores
recurrentes, métricas, resumen).

**Ruta** (`learning/route.py`): 20 unidades en orden (presente simple → frecuencia → presente continuo → pasado →
did/didn't → pasado continuo → going to → will → comparativos → present perfect ×3 → should/have to → peticiones →
used to → condicionales ×2 → opiniones → verb + -ing/to → preguntas). Cada unidad: regla en español para decirla
en voz alta, esquema, ejemplos, errores típicos de hispanohablantes, 8 palabras, 10 ejercicios guiados, 6 preguntas
de conversación (`TALK`), contextos de role-play y un detector (regex) de intentos de la estructura.

**Dominio** (tabla `unit_progress`): por clase se cuentan usos correctos de la estructura (respuestas a ejercicios
sin error de la estructura + frases libres donde el detector la ve sin error) y errores de la estructura (`skill_key`
de la unidad o coincidencia con sus errores típicos). La evidencia anterior se multiplica por 0,5 en cada clase nueva
(un mal día no bloquea para siempre). Dominada si: ≥2 clases, ≥12 usos correctos (ponderados), ≥75 % de acierto
ponderado, ≥80 % en la última clase y 6/8 palabras usadas. Si no, la siguiente clase es de la **misma unidad**, con
las palabras que faltan, los ejercicios no resueltos primero, otro contexto y un foco explícito ("la última vez 60 %
de acierto…"); desde la 4ª clase se pide explicarla de otra forma. Al dominarla: felicitación y unidad nueva; la
dominada vuelve como pregunta de repaso a los ≥2 días.

**Orquestación** (`session/lesson.py`, ver §3): el modelo pequeño no decide la estructura de la clase; el
orquestador le da en cada turno el paso exacto (parte de la presentación, ejercicio N, pregunta de conversación,
ítem de repaso) y comprueba con lo que realmente dijo si se avanzó (`drill_asked`, similitud de palabras clave).
**Comprobación instantánea**: antes de cada respuesta se pasan las reglas locales (instantáneas) a lo que dijo el
alumno; si hay un error seguro (p. ej. "my sister work", "he deal", "my boss no work", "I go always") la nota se lo dice
al profesor ("no digas que está bien; corrige y pide repetir"). Sin esto Gemma 4B daba por buenas respuestas
erróneas. Se añadieron reglas: 3ª persona sin -s (con excepciones tras auxiliares), negación "no + verbo", posición
de adverbios de frecuencia.

**Contexto**: 15 minutos no caben en 8k tokens; el historial se recorta de golpe (a ~7k caracteres, quitando notas
viejas) sólo al pasar de ~16k, para no invalidar la caché de prefijo de Ollama en cada turno.

**Validación** (`scripts/simulate_class.py`: Gemma de profesor + Gemma de alumno B1 con errores típicos, reloj simulado):

| Problema visto en la simulación | Cambio |
|---|---|
| Daba por buena "My boss no work" ("Excellent, exactly right!") | Comprobación instantánea con reglas + nota explícita |
| Corregía pero pasaba al siguiente ejercicio sin pedir repetir | Nota "no des el siguiente ejercicio todavía"; el orquestador detecta si repitió o avanzó |
| Al acabar los ejercicios: "Would you like another session?" | El último ejercicio enlaza directamente con la conversación |
| Hacia el minuto 10 se inventó un tema nuevo / "let's recap" / deberes / despedida | Preguntas de conversación por unidad, una a una; recordatorio en cada turno; sin reloj en las notas; regla "nunca cambies de tema ni termines" |
| Repaso: "My friend work" → "Excellent!" | El repaso funciona como los ejercicios (corrige y repite) |
| "(Pause)" leído en voz alta | El filtro de voz quita acotaciones |
| La precisión acumulada no dejaba dominar la unidad tras un mal día | Evidencia ponderada (×0,5 por clase) |

Resultado con un alumno simulado con errores en ~50 % de las frases: 3 clases de la unidad 1, cada una de 14:29–14:53
min, con el orden saludo → explicación → 10 ejercicios → conversación → repaso → cierre; dominio 64 % → 79 % → 85 %;
con la ponderación final la unidad queda dominada en la 3ª clase. Latencia media del profesor (respuesta completa,
no primera frase) 3,7–4,4 s. Límite conocido: errores que no cubren las reglas dependen de Gemma, que a veces los
pasa por alto; con la API de Claude (`TUTOR_LLM=anthropic`) la evaluación es más fiable.

## 15. Análisis de cada respuesta en pantalla (2026-10-23)

Tras evaluar cada turno, `tutor/ai/coach.py` escribe un análisis para la pantalla (no se dice en voz alta, para no
cortar la clase): una valoración corta y hasta 3 sugerencias, cada una con **qué decir**, **por qué**, **cómo se usa**
y **otro ejemplo**, más una **respuesta más completa** a la pregunta del profesor. Se envía como `turn_feedback`, se
guarda en `turns.eval_json["feedback"]` y vuelve en el resumen (`summary.feedback`).

- Las correcciones salen sólo de los errores ya verificados por el evaluador: el modelo las explica pero no puede
  añadir otras. Lo que no se puede comprobar o no está en español se descarta.
- Si el error es de una estructura de la ruta (la corrección introduce la estructura de una unidad), la regla y el
  ejemplo salen de la unidad, revisados a mano, porque un modelo de 4B explica mal la gramática.
- Va en segundo plano detrás de la evaluación (con Ollama, ~8-18 s por turno; la conversación tiene prioridad), así
  que el análisis suele llegar durante la respuesta siguiente. Al terminar se esperan como mucho 15 s.
- Comparativa con 5 respuestas reales: qwen3.5:4b explica bastante mejor que gemma3:4b (gemma inventa reglas,
  "have + participio" para "had a party"), pero es ~1,5× más lento.
- Desde el 2026-10-23 Qwen 3.5 4B es el modelo por defecto. Clase simulada completa (`scripts/simulate_class.py`):
  latencia media del profesor 4,2 s, sigue los 6 pasos y corrige siempre la -s. Defecto visto: a veces empieza con
  "Perfect!" y a continuación corrige, y repite mucho la fórmula "Just one small thing… Can you say that part again?".
