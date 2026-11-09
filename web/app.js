// Cliente: micrófono → WebSocket (PCM 16 kHz) · audio del tutor (PCM 24 kHz) → reproducción en cola.
const $ = (id) => document.getElementById(id);
const RING = 578;
const STATUS = {
  listening: "Te escucho… tómate tu tiempo",
  user_speaking: "Te escucho…",
  user_pause: "Te escucho… (sigue cuando quieras)",
  thinking: "…",
  tutor_speaking: "",
  summarizing: "Preparando tu resumen…",
};
const PHASES = { warmup: "Saludo", teach: "Explicación", drill: "Ejercicios", produce: "Conversación", review: "Repaso", wrapup: "Cierre" };
const ORDER = Object.keys(PHASES);

let ws, micCtx, micNode, micStream, player;
let roleplays = [], chosenRP = null, lessonMode = "unit";
let captionsOn = localStorage.getItem("cc") !== "0";

// ---------------------------------------------------------------- reproducción
class Player {
  constructor() {
    this.ctx = new AudioContext({ sampleRate: 24000 });
    this.next = 0;
    this.sources = new Map(); // reply_id -> Set(sources)
    this.ended = new Set(); // reply_id con reply_end recibido
    this.onDone = () => {};
  }
  play(replyId, int16) {
    const f = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, 24000);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const t = Math.max(this.ctx.currentTime + 0.03, this.next);
    src.start(t);
    this.next = t + buf.duration;
    if (!this.sources.has(replyId)) this.sources.set(replyId, new Set());
    this.sources.get(replyId).add(src);
    src.onended = () => {
      const set = this.sources.get(replyId);
      if (!set) return;
      set.delete(src);
      this.check(replyId);
    };
  }
  end(replyId) {
    this.ended.add(replyId);
    this.check(replyId);
  }
  check(replyId) {
    const set = this.sources.get(replyId);
    if (this.ended.has(replyId) && (!set || set.size === 0)) {
      this.ended.delete(replyId);
      this.sources.delete(replyId);
      this.onDone(replyId);
    }
  }
  stop(replyId) {
    const set = this.sources.get(replyId);
    if (set) for (const s of set) { try { s.stop(); } catch {} }
    this.sources.delete(replyId);
    this.ended.delete(replyId);
    this.next = 0;
  }
}

// ---------------------------------------------------------------- lección
async function start() {
  $("start").disabled = true;
  $("summary").classList.add("hidden");
  $("feedback").classList.add("hidden");
  $("feedback-list").innerHTML = "";
  setStatus("Conectando…");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (e) {
    setStatus("Necesito permiso para usar el micrófono.");
    $("start").disabled = false;
    return;
  }
  player = new Player();
  await player.ctx.resume();
  micCtx = new AudioContext();
  await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  const src = micCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(micCtx, "mic-processor");
  src.connect(micNode);

  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/lesson`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", roleplay: chosenRP?.key || null }));
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onAudio(ev.data));
  ws.onclose = () => cleanup();
  micNode.port.onmessage = ({ data }) => {
    if (paused) return;
    if (ws && ws.readyState === 1) ws.send(data.pcm);
    const core = document.querySelector(".orb .core");
    const s = $("orb").className.includes("user") ? 1 + Math.min(data.rms * 6, 0.25) : 1;
    core.style.transform = `scale(${s})`;
  };
  player.onDone = (id) => ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "playback_done", reply_id: id }));
  $("start").classList.add("hidden");
  $("end").classList.remove("hidden");
  $("pause").classList.remove("hidden");
  $("tools").classList.remove("hidden");
  $("rp-picker").classList.add("hidden");
}

function onAudio(buf) {
  const id = new DataView(buf).getUint32(0, true);
  player.play(id, new Int16Array(buf.slice(4)));
}

function onJson(m) {
  switch (m.type) {
    case "hello":
      if (m.llm === "mock") banner("Modo demo: no hay modelo de lenguaje disponible (instala Ollama y el modelo, ver README), el tutor sigue un guion fijo. El reconocimiento de voz, la detección de turnos y las métricas sí son reales.");
      break;
    case "lesson": {
      const p = m.plan;
      $("lesson-meta").textContent = p.unit
        ? `Clase ${p.lesson_number} · unidad ${p.unit.number} de ${p.unit.count} · ${m.level}`
        : `Clase ${p.lesson_number} · ${p.title} · ${m.level}`;
      lessonMode = p.mode || "unit";
      if (p.roleplay) {
        $("lesson-meta").textContent = `Role-play · ${p.roleplay.title_es} · ${m.level}`;
        showRoleplay(p.roleplay);
      } else if (p.unit) showToday(p.unit, p.new_vocab);
      setStepLabels(lessonMode);
      $("steps").classList.remove("hidden");
      break;
    }
    case "paused":
      paused = m.paused;
      $("pause").textContent = paused ? "▶ Reanudar" : "⏸ Pausa";
      $("pause").classList.toggle("on", paused);
      document.body.classList.toggle("is-paused", paused);
      setStatus(paused ? "En pausa. Pulsa «Reanudar» cuando quieras seguir." : "");
      break;
    case "state":
      setOrb(m.state);
      if (m.state in STATUS) setStatus(STATUS[m.state]);
      if (m.state === "listening") $("cap-user").textContent = "";
      break;
    case "clock": {
      const left = Math.max(0, m.total - m.elapsed);
      $("clock").textContent = `${String(Math.floor(left / 60)).padStart(2, "0")}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
      $("ring-fg").style.strokeDashoffset = RING * (1 - Math.min(m.elapsed / m.total, 1));
      $("phase").textContent = "";
      setStep(m.phase);
      break;
    }
    case "tutor_text":
      $("cap-tutor").textContent = m.text;
      setStatus("");
      break;
    case "user_partial":
    case "user_final":
      $("cap-user").textContent = m.text;
      break;
    case "reply_end":
      player.end(m.reply_id);
      break;
    case "stop_audio":
      player.stop(m.reply_id);
      break;
    case "turn_debug":
      $("debug").textContent = `smart-turn=${m.smart_turn}  texto=${m.text_p?.toFixed?.(2) ?? "-"}  completitud=${m.completeness}  silencio requerido=${m.required_silence}s  multiplicador=${m.multiplier}`;
      break;
    case "turn_feedback":
      addFeedback(m);
      break;
    case "summary":
      if (m.summary.feedback?.length) {
        $("feedback-list").innerHTML = "";
        m.summary.feedback.forEach(addFeedback);
      }
      showSummary(m.summary);
      ws.close();
      break;
    case "error":
      banner(m.message);
      break;
    case "busy":
      assist(`<p class="muted">${m.what === "repeat_phrase" ? "Espera a que termine el turno y vuelve a pulsar «Dila tú»." : "Espera a que termine el turno actual y vuelve a pulsar."}</p>`);
      break;
    case "translation":
      assist(m.text ? `<div class="lbl">Traducción</div><p>${esc(m.text)}</p>` : `<p class="muted">No he podido traducirlo esta vez.</p>`);
      break;
    case "help":
      assist(`<div class="lbl">Puedes empezar así</div>
        <ul class="starters">${m.starters.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        ${m.words?.length ? `<p class="muted">${m.words.map((w) => `<b>${esc(w.en)}</b> = ${esc(w.es)}`).join(" · ")}</p>` : ""}`);
      break;
    case "repeat_start":
      assist(`<div class="lbl">Escucha y repite</div><p class="target">${esc(m.target)}</p>`);
      break;
    case "repeat_result":
      if (m.cancelled) { assist(""); break; }
      showRepeat(m);
      break;
  }
}

function assist(html) {
  $("assist").innerHTML = html;
  $("assist").classList.toggle("hidden", !html);
}

let repeatCard = null;
function showRepeat(m) {
  const missing = m.missing?.length ? `<p class="muted">Revisa: ${m.missing.map((w) => `<b>${esc(w)}</b>`).join(", ")}${m.unclear?.length ? ` · no se entendió bien: ${m.unclear.map(esc).join(", ")}` : ""}</p>` : "";
  const html = `<div class="lbl">Tu repetición · <span class="score ${m.score >= 85 ? "ok" : m.score >= 70 ? "mid" : "low"}">${m.score}/100</span></div>
    <p class="better">${wordDiff(m.said, m.target)}</p>${missing}`;
  assist(html);
  if (repeatCard) {
    let box = repeatCard.querySelector(".rep");
    if (!box) { box = document.createElement("div"); box.className = "rep"; repeatCard.appendChild(box); }
    box.innerHTML = html;
  }
}

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function cleanup() {
  try { micStream && micStream.getTracks().forEach((t) => t.stop()); } catch {}
  try { micCtx && micCtx.close(); } catch {}
  ws = null;
  setOrb("idle");
  $("start").classList.remove("hidden");
  $("start").disabled = false;
  $("start").textContent = "Nueva clase";
  $("steps").classList.add("hidden");
  $("tools").classList.add("hidden");
  $("pause").classList.add("hidden");
  $("pause").textContent = "⏸ Pausa";
  paused = false;
  document.body.classList.remove("is-paused");
  $("rp-picker").classList.remove("hidden");
  chosenRP = null;
  document.querySelectorAll(".rp").forEach((b) => b.classList.remove("on"));
  $("start").textContent = "Nueva clase";
  assist("");
  document.querySelectorAll(".say").forEach((b) => b.remove());
  loadToday();
  $("end").classList.add("hidden");
}

function setStepLabels(mode) {
  const rp = mode === "roleplay";
  const labels = rp ? { warmup: "Preparación", produce: "Role-play", review: "Feedback", wrapup: "Cierre" } : PHASES;
  document.querySelectorAll("#steps li").forEach((li) => {
    const p = li.dataset.p;
    li.classList.toggle("hidden", rp && (p === "teach" || p === "drill"));
    li.textContent = labels[p] || PHASES[p];
  });
}

// Role-play: la escena, tu papel, los objetivos y frases útiles (antes y durante la práctica).
function showRoleplay(r) {
  const el = $("today");
  el.innerHTML = `
    <div class="unit">Role-play profesional · 10 minutos</div>
    <h2>${esc(r.title_es)}</h2>
    <p class="goal">${esc(r.scene_es)}</p>
    <p class="goal"><b>Tu papel:</b> ${esc(r.learner_role_es)}</p>
    <div class="unit" style="margin-top:10px">Objetivos</div>
    <ul class="goals">${r.goals.map((g) => `<li>${esc(g.es)} <span class="muted">— ${esc(g.en)}</span></li>`).join("")}</ul>
    <div class="unit" style="margin-top:10px">Frases útiles</div>
    <ul class="phrases">${r.phrases.map(([en, es]) => `<li><b>${esc(en)}</b> <span class="muted">${esc(es)}</span></li>`).join("")}</ul>`;
  el.classList.remove("hidden");
}

async function loadRoleplays() {
  try { roleplays = await fetch("/api/roleplays").then((x) => x.json()); } catch { return; }
  $("rp-list").innerHTML = roleplays.map((r) => `<button class="chip rp" data-key="${esc(r.key)}">${esc(r.title_es)}</button>`).join("");
  $("rp-list").querySelectorAll(".rp").forEach((b) => (b.onclick = () => chooseRP(b.dataset.key)));
}

function chooseRP(key) {
  const r = roleplays.find((x) => x.key === key);
  const same = chosenRP && chosenRP.key === key;
  chosenRP = same ? null : r;
  document.querySelectorAll(".rp").forEach((b) => b.classList.toggle("on", !same && b.dataset.key === key));
  if (chosenRP) {
    showRoleplay(chosenRP);
    $("start").textContent = "Empezar role-play";
  } else {
    $("start").textContent = "Empezar clase";
    loadToday();
  }
}

function setStep(phase) {
  const i = ORDER.indexOf(phase);
  document.querySelectorAll("#steps li").forEach((li, j) => {
    li.classList.toggle("done", j < i);
    li.classList.toggle("now", j === i);
  });
}

// Tarjeta del tema del día (antes y durante la clase): qué aprendemos hoy y cómo va la unidad.
function showToday(u, newVocab) {
  const el = $("today");
  const words = (newVocab || []).map((v) => esc(v.term)).join(" · ");
  el.innerHTML = `
    <div class="unit">Unidad ${u.number} de ${u.count} · clase ${u.lesson} de la unidad</div>
    <h2>${esc(u.title_es)}</h2>
    <p class="goal">${esc(u.goal)}</p>
    <div class="form">${esc(u.form)}</div>
    ${words ? `<p class="goal" style="margin-top:8px">Palabras nuevas: ${words}</p>` : ""}
    <div class="bar"><i style="width:${u.score || 0}%"></i></div>
    <div class="unit">Dominio de la unidad: ${u.score || 0}%</div>`;
  el.classList.remove("hidden");
}

async function loadToday() {
  try {
    const r = await fetch("/api/profile").then((x) => x.json());
    showStreak(r.streak);
    if (chosenRP) return;
    const p = r.next_plan;
    if (p.unit) showToday({ number: p.unit_number, count: p.unit_count, lesson: p.unit_lesson, title_es: p.unit.title_es,
                            goal: p.unit.goal, form: p.unit.form, score: p.unit_status?.score || 0 }, p.new_vocab);
  } catch {}
}

function showStreak(st) {
  if (!st) return;
  const el = $("streak");
  el.textContent = `🔥 ${st.current}`;
  el.title = `Racha: ${st.current} día${st.current === 1 ? "" : "s"} seguidos · récord ${st.best} · esta semana ${st.this_week}/${st.week_goal}` +
    (st.current && !st.today ? " · ¡haz una clase hoy para no perderla!" : "");
  el.classList.toggle("warn", !!st.current && !st.today);
  el.classList.remove("hidden");
}

function setOrb(s) { $("orb").className = `orb ${s}`; }
function setStatus(t) { $("status").textContent = t; }
function banner(t) { $("banner").textContent = t; $("banner").classList.remove("hidden"); }
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// Marca palabra a palabra qué cambia: lo que sobra o está mal (tachado) y lo nuevo (resaltado).
function wordDiff(said, better) {
  const a = said.split(/\s+/).filter(Boolean), b = better.split(/\s+/).filter(Boolean);
  const norm = (w) => w.toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9']/g, "");
  const n = a.length, m = b.length;
  const L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i][j] = norm(a[i]) === norm(b[j]) ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const left = [], right = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && norm(a[i]) === norm(b[j])) { left.push(esc(a[i])); right.push(esc(b[j])); i++; j++; }
    else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) { right.push(`<mark>${esc(b[j])}</mark>`); j++; }
    else { left.push(`<del>${esc(a[i])}</del>`); i++; }
  }
  return `<span class="old">${left.join(" ")}</span> → <b>${right.join(" ")}</b>`;
}

const KIND = { correction: "Corrección", more_natural: "Más natural", richer_answer: "Respuesta más completa" };

function addFeedback(m) {
  const fb = m.feedback;
  const card = document.createElement("article");
  card.className = "fb";
  card.innerHTML = `
    <p class="said">“${esc(m.text)}”</p>
    <p class="analysis">${esc(fb.analysis_es)}</p>
    ${fb.suggestions.map((s) => `
      <div class="sug ${s.kind}">
        <div class="kind">${KIND[s.kind] || ""}</div>
        <p class="better">${s.you_said ? wordDiff(s.you_said, s.better) : `<b>${esc(s.better)}</b>`}</p>
        ${s.why_es ? `<p><span class="lbl">Por qué:</span> ${esc(s.why_es)}</p>` : ""}
        ${s.how_es ? `<p><span class="lbl">Cómo se usa:</span> ${esc(s.how_es)}</p>` : ""}
        ${s.example ? `<p class="ex"><span class="lbl">Otro ejemplo:</span> ${esc(s.example)}</p>` : ""}
        ${ws && (s.sentence || s.better) ? `<button class="ghost small say" data-text="${esc(s.sentence || s.better)}" data-key="${s.in_deck ? esc(s.better) : ""}">🎙️ Dila tú</button>` : ""}
      </div>`).join("")}`;
  card.querySelectorAll(".say").forEach((b) => (b.onclick = () => {
    repeatCard = b.closest(".sug");
    send({ type: "repeat_phrase", text: b.dataset.text, key: b.dataset.key });
  }));
  $("feedback-list").prepend(card);
  $("feedback").classList.remove("hidden");
}

function showSummary(s) {
  setStatus("¡Lección terminada!");
  $("phase").textContent = "";
  const el = $("summary");
  if (s.aborted) {
    el.innerHTML = `<h2>Lección interrumpida</h2><p class="muted">La lección fue demasiado corta para guardar progreso.</p>`;
  } else {
    el.innerHTML = `
      <h2>Today's progress</h2>
      ${unitCard(s.unit, s.next_unit)}
      <h3>🔥 Fluency</h3><p>${esc(s.fluency_line)}</p>
      ${s.well_done?.length ? `<h3>👏 Well done</h3><ul>${s.well_done.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${s.new_vocabulary?.length ? `<h3>📚 New vocabulary</h3><ul>${s.new_vocabulary.map((v) => `<li><b>${esc(v.term)}</b> — ${esc(v.meaning_es)}</li>`).join("")}</ul>` : ""}
      <h3>⚠️ Let's improve</h3><ul>${(s.improve || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <h3>💬 Useful expression</h3><p class="expr">"${esc(s.useful_expression)}" <span class="muted">— ${esc(s.useful_expression_es)}</span></p>
      ${s.roleplay ? `<h3>🎯 Objetivos del role-play</h3><ul class="goals">${s.roleplay.goals.map((g) => `<li>${g.ok ? "✅" : "⬜"} ${esc(g.es)}</li>`).join("")}</ul>` : ""}
      ${deckLine(s.deck)}
      ${listenSection(s)}
      ${pronSection(s.pronunciation)}
      ${s.streak ? `<h3>🔥 Racha</h3><p>${s.streak.current} día${s.streak.current === 1 ? "" : "s"} seguido${s.streak.current === 1 ? "" : "s"} · récord ${s.streak.best} · esta semana ${s.streak.this_week} de ${s.streak.week_goal} clases.</p>` : ""}
      <h3>Next class</h3><p>${esc(s.tomorrow)}</p>
      ${s.session_id ? `<p><a class="primary dl" href="/api/report/${s.session_id}" download>⬇️ Descargar informe con audio</a></p>` : ""}
      <div class="kpis">
        ${kpi(s.fluency?.speaking_wpm, "palabras/min")}
        ${kpi(s.fluency?.mean_turn_words, "palabras por respuesta")}
        ${kpi(s.fluency?.median_onset_s != null ? s.fluency.median_onset_s + " s" : "–", "antes de responder")}
        ${kpi(s.level, "nivel estimado")}
      </div>`;
  }
  el.classList.remove("hidden");
}
function unitCard(u, next) {
  if (!u) return "";
  const head = u.mastered
    ? `<b>🎉 ¡Unidad dominada!</b> ${esc(u.title_es)}<br><span class="muted">Siguiente clase: ${esc(next?.title)} (unidad ${next?.number})</span>`
    : `<b>📈 ${esc(u.title_es)}</b> <span class="muted">— unidad ${u.number}, clase ${u.lesson}</span>`;
  const missing = !u.mastered && u.missing?.length
    ? `<p class="muted" style="margin:6px 0 0">Para dominarla te falta: ${u.missing.map(esc).join(" · ")}. Seguimos con ella en la próxima clase.</p>` : "";
  return `<div class="unitcard ${u.mastered ? "mastered" : ""}">${head}
    <div class="bar"><i style="width:${u.score}%"></i></div>
    <span class="muted">Dominio ${u.score_before}% → <b>${u.score}%</b> · ejercicios bien ${u.drills_ok}/${u.drills_tried} ·
      palabras usadas ${u.words_used}/${u.words_total}</span>${missing}</div>`;
}
function listenSection(s) {
  const seen = new Set(), items = [];
  for (const f of s.feedback || []) for (const x of f.feedback.suggestions || []) {
    const t = x.sentence || x.better;
    if (t && !seen.has(t.toLowerCase()) && x.kind !== "richer_answer") { seen.add(t.toLowerCase()); items.push(t); }
  }
  if (!items.length) return "";
  return `<h3>🎧 Escucha tus frases corregidas</h3><ul class="listen">${items.slice(0, 8).map((t) =>
    `<li><button class="ghost small play" data-text="${esc(t)}">🔊</button> ${esc(t)}</li>`).join("")}</ul>`;
}
function pronSection(list) {
  if (!list?.length) return "";
  return `<h3>🗣️ Pronunciación</h3><p class="muted">Palabras que costó entender hoy:</p><ul class="listen">${list.map((p) =>
    `<li><button class="ghost small play" data-text="${esc(p.word)}">🔊</button> <b>${esc(p.word)}</b>${p.heard ? ` <span class="muted">(se entendió «${esc(p.heard)}»)</span>` : ""}<br><span class="muted">${esc(p.tip)}</span></li>`).join("")}</ul>`;
}
function deckLine(d) {
  if (!d) return "";
  const parts = [];
  if (d.added?.length) parts.push(`${d.added.length} frase${d.added.length === 1 ? "" : "s"} nueva${d.added.length === 1 ? "" : "s"} en tu mazo`);
  const ok = (d.reviewed || []).filter((r) => r.ok), bad = (d.reviewed || []).filter((r) => !r.ok);
  if (ok.length) parts.push(`usaste bien ${ok.map((r) => `“${esc(r.better)}”`).join(", ")}`);
  if (bad.length) parts.push(`aún dijiste ${bad.map((r) => `“${esc(r.said)}”`).join(", ")} (vuelve mañana)`);
  if (d.repeats?.length) parts.push(`repetiste ${d.repeats.length} frase${d.repeats.length === 1 ? "" : "s"} en voz alta`);
  return parts.length ? `<h3>🃏 Tu mazo de frases</h3><p>${parts.join(" · ")}.</p>` : "";
}
function playText(text) { new Audio(`/api/tts?text=${encodeURIComponent(text)}`).play().catch(() => {}); }
const kpi = (v, label) => `<div class="kpi"><b>${esc(v ?? "–")}</b><span>${esc(label)}</span></div>`;

// ---------------------------------------------------------------- progreso
async function showProgress() {
  const r = await fetch("/api/profile").then((x) => x.json());
  const s = r.snapshot;
  const pts = r.history.filter((h) => h.fluency_index != null);
  const spark = pts.length > 1 ? sparkline(pts.map((p) => p.fluency_index)) : `<p class="muted">El gráfico aparece a partir de la 2ª clase.</p>`;
  const plan = r.next_plan;
  $("progress").innerHTML = `
    <h2>Tu progreso</h2>
    <div class="kpis">
      ${kpi(`🔥 ${r.streak?.current ?? 0}`, `días seguidos (récord ${r.streak?.best ?? 0})`)}
      ${kpi(`${r.streak?.this_week ?? 0}/${r.streak?.week_goal ?? 5}`, "clases esta semana")}
      ${kpi(s.level, "nivel estimado")}
      ${kpi(s.lessons_completed, "clases")}
      ${kpi(s.vocab.active ?? 0, "palabras activas")}
      ${kpi(s.vocab.mastered ?? 0, "dominadas")}
    </div>
    <h3>Fluidez (índice por clase)</h3>${spark}
    ${s.strengths.length ? `<h3>Fortalezas</h3><ul>${s.strengths.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${s.weaknesses.length ? `<h3>A mejorar</h3><ul>${s.weaknesses.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${Object.keys(s.skills).length ? `<h3>Gramática (dominio)</h3><div class="chips">${Object.entries(s.skills).map(([k, v]) => `<span class="chip">${esc(k.replace(/_/g, " "))} ${Math.round(v * 100)}%</span>`).join("")}</div>` : ""}
    ${s.recurring_mistakes.length ? `<h3>Errores recurrentes</h3><ul>${s.recurring_mistakes.map((m) => `<li>"${esc(m.original)}" → "${esc(m.corrected)}" <span class="muted">(${m.count}×)</span></li>`).join("")}</ul>` : ""}
    ${r.vocab.length ? `<h3>Vocabulario</h3><div class="chips">${r.vocab.map((v) => `<span class="chip ${v.status}" title="${esc(v.meaning_es)}">${esc(v.term)}</span>`).join("")}</div>` : ""}
    ${calendar(r.streak?.days || [])}
    ${deckSection(r.phrases || [])}
    ${progressPron(r.pron_words || [])}
    <h3>Próxima clase</h3>
    <p>${esc(plan.unit?.title_es || plan.title)} <span class="muted">(unidad ${plan.unit_number}, clase ${plan.unit_lesson} · ${esc(plan.weekly_goal)})</span></p>
    <h3>Tu ruta</h3>
    <ul class="route">${r.route.map((u) => `
      <li class="${u.status}"><span class="n">${u.status === "mastered" ? "✅" : u.status === "current" ? "▶" : u.number}</span>
        <span>${esc(u.title_es)}${u.status === "current" ? `<div class="bar small"><i style="width:${u.score}%"></i></div>` : ""}</span>
        <span class="muted">${u.status === "mastered" ? "dominada" : u.status === "current" ? `${u.score}% · ${u.lessons} clase${u.lessons === 1 ? "" : "s"}` : ""}</span></li>`).join("")}
    </ul>
    <p class="muted" style="margin-top:14px">Paciencia aprendida: pausa típica ${s.patience.pause_p50.toFixed(1)} s · pausas largas ${s.patience.pause_p90.toFixed(1)} s · multiplicador ×${s.patience.multiplier.toFixed(2)}</p>`;
}

function calendar(days) {
  const set = new Set(days), out = [];
  const today = new Date();
  for (let i = 27; i >= 0; i--) {
    const d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - i);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    out.push(`<i class="${set.has(key) ? "on" : ""}" title="${key}"></i>`);
  }
  return `<h3>Últimas 4 semanas</h3><div class="cal">${out.join("")}</div>`;
}
function progressPron(list) {
  if (!list.length) return "";
  return `<h3>Pronunciación <span class="muted">(palabras que más cuesta entender)</span></h3><ul class="deck">${list.map((p) => `
    <li><button class="ghost small play" data-text="${esc(p.word)}">🔊</button>
      <span><b>${esc(p.word)}</b>${p.heard ? ` <span class="muted">(se entendió «${esc(p.heard)}»)</span>` : ""}<br><span class="muted">${esc(p.tip)}</span></span>
      <span class="muted">${p.sessions} clase${p.sessions === 1 ? "" : "s"}</span></li>`).join("")}</ul>`;
}
function deckSection(items) {
  if (!items.length) return "";
  const now = Date.now() / 1000;
  const due = items.filter((p) => p.next_due <= now).length;
  return `<h3>Tu mazo de frases <span class="muted">(${items.length} · ${due} para repasar hoy)</span></h3>
    <ul class="deck">${items.slice(0, 40).map((p) => `
      <li class="${p.next_due <= now ? "due" : ""}">
        <button class="ghost small play" data-text="${esc(p.sentence || p.better)}" title="Escuchar">🔊</button>
        <span>${p.said ? wordDiff(p.said, p.better) : `<b>${esc(p.better)}</b>`}
          ${p.why_es ? `<br><span class="muted">${esc(p.why_es)}</span>` : ""}</span>
        <span class="muted">${"●".repeat(p.box)}${"○".repeat(Math.max(0, 6 - p.box))}</span>
      </li>`).join("")}</ul>`;
}

function sparkline(values) {
  const w = 600, h = 70, max = Math.max(...values, 1), min = Math.min(...values, 0);
  const xy = values.map((v, i) => [(i / (values.length - 1)) * (w - 10) + 5, h - 5 - ((v - min) / (max - min || 1)) * (h - 10)]);
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="M${xy.map((p) => p.join(",")).join(" L")}"/>${xy.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="3"/>`).join("")}</svg>`;
}

// ---------------------------------------------------------------- navegación
function view(name) {
  $("view-lesson").classList.toggle("hidden", name !== "lesson");
  $("view-progress").classList.toggle("hidden", name !== "progress");
  $("nav-lesson").classList.toggle("active", name === "lesson");
  $("nav-progress").classList.toggle("active", name === "progress");
  if (name === "progress") showProgress();
}

$("start").onclick = start;
$("t-repeat").onclick = () => send({ type: "repeat" });
$("t-slower").onclick = () => send({ type: "slower" });
$("t-translate").onclick = () => send({ type: "translate" });
$("t-help").onclick = () => { assist(`<p class="muted">Pensando ideas…</p>`); send({ type: "help" }); };
document.addEventListener("click", (e) => {
  const b = e.target.closest?.(".play");
  if (b) playText(b.dataset.text);
});
$("end").onclick = () => ws && ws.send(JSON.stringify({ type: "end" }));
let paused = false;
async function togglePause() {
  if (!ws || ws.readyState !== 1) return;
  const on = !paused;
  // La voz del profesor se congela/sigue en el mismo punto: se suspende el contexto de audio del reproductor.
  try { on ? await player.ctx.suspend() : await player.ctx.resume(); } catch {}
  send({ type: on ? "pause" : "resume" });
}
$("pause").onclick = togglePause;
$("nav-lesson").onclick = () => view("lesson");
$("nav-progress").onclick = () => view("progress");
$("cc").onclick = () => {
  captionsOn = !captionsOn;
  localStorage.setItem("cc", captionsOn ? "1" : "0");
  applyCC();
};
function applyCC() {
  $("captions").classList.toggle("off", !captionsOn);
  $("cc").classList.toggle("on", captionsOn);
}
applyCC();
loadToday();
loadRoleplays();
document.addEventListener("keydown", (e) => {
  if ((e.key === "p" || e.key === "P") && !e.altKey && !e.ctrlKey && !e.metaKey &&
      !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) togglePause();
  if (e.key === "d" && e.altKey) $("debug").classList.toggle("hidden"); // Alt+D: panel de detección de turnos
});
