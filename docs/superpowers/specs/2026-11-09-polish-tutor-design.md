# Tutor de polaco — diseño

Fecha: 2026-11-09 · Estado: listo para aprobar

## 1. Intención

Un profesor de polaco por voz con **la misma dinámica que el tutor de inglés** de este repositorio: clases guiadas
de 15 min, ruta de unidades que no avanza hasta dominar la actual, correcciones precisas con explicación en
español, botones de ayuda (Repite / Más despacio / Traducir / Ayúdame a responder / Dila tú / Pausa), mazo de
frases con repetición espaciada, role-plays e informe de la clase. Todo local y gratis.

Para quién: una persona hispanohablante que **ya se defiende con lo básico** (A1–A2) y quiere practicar **todas
las situaciones de la vida diaria**, no un tema concreto.

Por qué un modelo propio: en las pruebas del 2026-11-09 (§9) ningún modelo local corrige ni explica bien el
polaco. Una explicación gramatical falsa es peor que ninguna, así que el profesor necesita un modelo entrenado
para esto («tutor-ola»).

## 2. Objetivos, no-objetivos y anti-objetivos

Objetivos
- Misma experiencia que el de inglés, en polaco, eligiendo el idioma al empezar.
- Correcciones fiables: casos, conjugación, género en pasado, aspecto, concordancia de numerales, preposiciones.
- Explicaciones en español correctas y cortas (≤70 palabras), con otro ejemplo en polaco.
- Funcionar con o sin GPU sin tocar código (igual que el inglés: se adapta solo).

No-objetivos (por ahora)
- Nivel B2+ o escritura/ortografía como tema propio.
- Otros idiomas de explicación distintos del español.
- Reconocimiento de pronunciación por fonemas (se mantiene el enfoque actual: palabras que Whisper entiende mal).

Anti-objetivos (sería un fracaso aunque funcione)
- Que el profesor «corrija» frases correctas o invente reglas (p. ej. «piwo es masculino»).
- Romper o empeorar el tutor de inglés: sus 107 tests deben seguir pasando y su comportamiento no cambia.
- Duplicar el proyecto entero en una copia que luego diverge.

## 3. Restricciones

- Coste cero: modelos locales; nada de API de pago.
- **Tiene que funcionar en Linux y en macOS.** El Mac de destino **no es el de desarrollo** sino otro, cuyas
  características no se conocen: no se puede suponer Apple Silicon, ni RAM, ni Java, ni nada instalado. Todo se
  instala con los pasos del README y la app se adapta sola al equipo (como ya hace el inglés con la GPU).
- Equipos disponibles para desarrollar y probar: Mac Apple Silicon de 16 GB (disco justo, ~30 GB libres) y PC
  Linux con RTX 5070 (12 GB de VRAM), Ryzen 5 8500G, 32 GB de RAM. El entrenamiento se hace en el PC.
- Mismo stack: Python/FastAPI, web mínima, Ollama, Whisper, Silero VAD, Smart Turn v3.2.

## 4. Decisiones ya tomadas

- **Explicaciones en español**, como en el de inglés (lo dijo el usuario).
- **Modelo propio entrenado** para el profesor de polaco, siguiendo el pipeline de `training/` (lo aprobó el usuario).
- **Los datos de entrenamiento los escribe Claude** en sesión (correcciones, explicaciones, turnos del profesor),
  porque no hay un modelo maestro local que sepa polaco (aprobado).
- **LanguageTool** (reglas para polaco, local) como detector de errores seguros, en el papel de las reglas
  instantáneas del inglés (aprobado).
- Reconocimiento: **Whisper large-v3-turbo** multilingüe (MLX en el Mac, CUDA en el PC) con `language="pl"`.
- Fin de turno: **Smart Turn v3.2** tal cual (incluye polaco: 94,5 % de acierto según Daily).
- Voz del profesor en polaco: **Piper** (voces `pl_PL`, MIT/CC0, ~63 MB cada una, 0,08 s por frase en el PC).

## 5. Diseño

### 5.1 Un código, dos idiomas (paquetes de idioma)

Todo lo que depende del idioma que se aprende sale de un **paquete de idioma**, y el resto del código (lección,
detector de turnos, SRS, informe, web) es común:

```
tutor/lang/
  __init__.py        # interfaz LanguagePack + get_pack("en"|"pl")
  en/                # lo que hoy está repartido: route, scenarios, roleplays, reglas, prompts, muletillas,
                     #   pronunciación, voces → se mueve aquí SIN cambiar contenido
  pl/
    route.py         # ~20 unidades A1→B1 (lista en §5.3)
    scenarios.py     # situaciones de la vida diaria (§5.4)
    roleplays.py
    rules.py         # LanguageTool + reglas propias de errores típicos de hispanohablantes
    prompts.py       # prompts completos (para modelos genéricos) y compactos (para tutor-ola)
    fluency.py       # muletillas polacas (yyy, eee, no, wiesz, jakby, znaczy)
    pronunciation.py # sonidos difíciles para hispanohablantes (sz/ś/cz/ć/rz/ż, ą/ę, y, ł)
```

El paquete del inglés se crea moviendo lo que existe, con los tests actuales como red: si pasan igual, el
movimiento es correcto.

Elección del idioma: un selector **Inglés / Polaco** en la pantalla de inicio (recordado en el navegador) y
`TUTOR_LANG` como valor por defecto. **Progreso separado por idioma**: misma BD con una columna `lang` en las
tablas de progreso (o un fichero por idioma; lo decide quien lo construya, ver §8).

### 5.2 Voz

- STT: `resolve_stt` ya elige motor y dispositivo; se añade el idioma. En CPU sin GPU, `small.en` no sirve para
  polaco → se usa `faster-whisper-small` multilingüe (peor, pero funciona) y se avisa en el log.
- TTS: el texto del profesor sigue usando `<es>…</es>` para el español. Partes en polaco → Piper (22,05 kHz,
  remuestreado a 24 kHz), partes en español → Kokoro `ef_dora` (ya lo usa el inglés). «Más despacio» con
  `length_scale` de Piper.
- Voz por defecto: **gosia** (femenina, encaja con «Ola»); configurable con `TUTOR_VOICE_PL`.
- `download_models.py` baja la voz de Piper y, si hace falta, Whisper multilingüe.

### 5.3 Ruta de unidades (A1–A2 → B1)

Cada unidad con la misma estructura `Unit` del inglés (explicación en español, forma, ejemplos, errores típicos de
hispanohablantes, vocabulario de 8 términos, ejercicios, tarea de producción, escenarios, regex de detección).
Orden propuesto:

1. Presentarse: być, mieć, nazywać się · 2. Presente -ę/-esz (robić, mówić) · 3. Presente -am/-asz, -em/-esz ·
4. Acusativo (biernik): lubię kawę, mam brata · 5. Locativo con w/na: mieszkam w Krakowie, na poczcie ·
6. Instrumental: jestem nauczycielem, z koleżanką · 7. Genitivo tras negación y cantidad: nie mam czasu, dużo
ludzi · 8. Numerales y edad: dwa/dwie, pięć + genitivo, od dwóch lat · 9. Pasado con género: byłem/byłam, poszedł/
poszła · 10. Aspecto I: robiłem vs zrobiłem · 11. Futuro: będę robić / zrobię · 12. Verbos de movimiento: iść/
chodzić, jechać/jeździć · 13. Preposiciones de lugar y dirección: do, w/na, z, od · 14. Verbos modales: chcieć,
móc, musieć, trzeba · 15. Cortesía y condicional: chciałbym, poproszę, czy mógłby pan · 16. Imperativo: idź,
weź, proszę + infinitivo · 17. Dativo: dać komuś, podoba mi się · 18. Reflexivos y się · 19. Comparativos:
lepszy, najlepszy, bardziej · 20. Conectores y relato: bo, więc, kiedy, który.

### 5.4 Situaciones («todas las situaciones posibles»)

Los escenarios y role-plays cubren la vida diaria en Polonia, alternándose entre clases: tienda y mercado,
restaurante y café, transporte (billetes, preguntar direcciones), médico y farmacia, banco y correos, oficina de
extranjería y trámites, alquilar piso y casero, trabajo y compañeros, vecinos, teléfono y citas, compras de ropa,
reparaciones, ocio y planes con amigos, emergencias. Los role-plays de 10 min del inglés tienen su equivalente.

### 5.5 Correcciones: reglas + modelo

Igual que en inglés: errores seguros por reglas, y el modelo para el resto; las reglas ganan en solapes
(`merge_evaluations`, `narrow_error`).
- **LanguageTool** en local (servidor Java, `language_tool_python` o el .jar directo) con `pl-PL`; solo se aceptan
  las categorías de gramática (no estilo ni ortografía, porque el texto viene de Whisper).
- Reglas propias para errores típicos de hispanohablantes que LanguageTool no ve (pronombre sujeto redundante «ja
  idę», presente por pasado con «wczoraj», «od dwa lata»).
- Si Java no está, se sigue sin LanguageTool (aviso en el log), no se rompe la app.

### 5.6 Modelo «tutor-ola»

Mismo pipeline que `tutor-sam` (§training), en el PC:
- Base: **Qwen3.5-4B** (rápido, ~150 tok/s en la 5070). Si tras la evaluación no da la talla, **Qwen3.5-9B con
  QLoRA** (cabe en 12 GB) como segundo intento. El de 9B sin entrenar ya conversa bien en polaco.
- Tareas, igual que en inglés: chat del profesor, evaluación (JSON), coach (análisis de la respuesta), ayuda
  («ayúdame a responder»), datos personales (facts).
- Datos:
  1. Entradas: clases simuladas (`collect.py` con alumno simulado con errores típicos, generado a partir de la ruta)
     + frases de alumno escritas a mano por error tipo.
  2. Respuestas modelo: **escritas por Claude** por tandas en ficheros de `training/gold_pl/`; cada corrección se
     contrasta con LanguageTool y con un analizador morfológico (Morfeusz 2) cuando se pueda.
  3. Prompts compactos (destilación de contexto, como `compact.py`).
- Volumen objetivo: ~1.500 ejemplos (≈ 700 chat, 400 evaluación, 250 coach, 150 ayuda/facts); conjunto de test
  aparte, que no se usa para entrenar.
- Salida: GGUF Q4_K_M con la plantilla Jinja del entrenamiento ya incrustada (lección aprendida con `tutor-sam`),
  `ollama create tutor-ola`. El mismo GGUF sirve en Linux (CUDA o CPU) y en macOS (Metal en Apple Silicon, CPU en
  Intel): un solo fichero de 2,8 GB para todos los equipos, sin repetir la fusión.
- Instalación en otro equipo: **el usuario copia el GGUF a mano (AirDrop)**; no se publica en ningún sitio. Un paso
  del README (`scripts/install_model.py <ruta al .gguf>`, que también lo busca en ~/Downloads) crea `tutor-ola`
  en el Ollama local con su Modelfile. Sin el modelo, la app sigue funcionando con qwen3.5:9b y lo avisa.

### 5.7 Medir antes de dar por bueno

`eval_student.py` con un conjunto de test de polaco: F1 de errores, **falsos errores en frases correctas** (lo
más importante), JSON válido, % de explicaciones que pasan el control, reglas del profesor, latencia.
Comparación con qwen3.5:9b de serie. Criterio para darlo por bueno: falsos errores ≤ 5 % y precisión de
correcciones ≥ 90 % en el test; si no, segundo intento con 9B.

## 6. Orden de trabajo

1. Paquetes de idioma: mover el inglés a `tutor/lang/en` sin cambios de comportamiento (tests verdes).
2. Voz polaca: STT con idioma, TTS Piper + Kokoro, descarga de modelos. Prueba con frases grabadas.
3. Contenido polaco: ruta, escenarios, role-plays, reglas + LanguageTool, prompts.
4. App jugable en polaco con qwen3.5:9b (sin entrenar) para probar la dinámica.
5. Datos de entrenamiento (gold escrito por Claude) → entrenar tutor-ola → evaluar → integrar.
6. Clase real con micrófono y revisión, como se hizo con el inglés.

## 7. Riesgos

- El polaco que escribe Claude puede tener algún error; se mitiga contrastando cada corrección con LanguageTool y
  Morfeusz 2. *Decisión de Claude:* no se cuenta con revisión de un nativo (no se ha confirmado que haya uno); si
  aparece, revisaría una muestra de ~50 correcciones del conjunto de test.
- Whisper con acento español en polaco no está medido con voz real (las pruebas fueron con voces sintéticas).
- Un 4B quizá no tenga suficiente conocimiento del polaco ni con entrenamiento → plan B: 9B QLoRA.
- El Mac de destino es desconocido: si es Intel, Whisper va en CPU (small multilingüe, menos preciso) y el modelo
  de 4B irá más lento; si tiene 8 GB de RAM, irá justo. Hay que probarlo allí antes de darlo por terminado.
- LanguageTool necesita Java (~200 MB) en cada equipo. El Mac ya tiene OpenJDK 25 (Homebrew); el PC no tiene Java y
  sin sudo se instala un JRE portable (Temurin) en la carpeta del usuario.

## 8. Lo que queda a criterio de quien lo construya

- Esquema exacto de la BD para separar el progreso por idioma (columna `lang` o fichero aparte).
- Detalles de la interfaz `LanguagePack` y nombres de los ficheros.
- Redacción final de cada unidad, ejercicio y role-play.
- Hiperparámetros del entrenamiento.

## 9. Pruebas previas (2026-11-09, PC)

- Whisper turbo (CUDA) con 18 frases de 3 voces Piper: transcribe bien y conserva los errores de estudiante (solo
  una vez cambió «piwo» → «piwa»).
- LLM, frases de alumno con errores (resumen): qwen3.5:4b corrige mal; gemma3:4b no corrige o inventa; Bielik
  4.5B mezcla idiomas e inventa en español; qwen3.5:9b conversa bien pero explica la gramática mal. Ninguno es
  aceptable para las explicaciones → modelo propio.
- Scripts y salidas de la prueba: `~/pl-recon/` en el PC.
