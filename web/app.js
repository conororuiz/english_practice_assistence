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
const PHASES = { warmup: "Calentamiento", main: "Conversación", practice: "Práctica", wrapup: "Cierre" };

let ws, micCtx, micNode, micStream, player;
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
  ws.onopen = () => ws.send(JSON.stringify({ type: "start" }));
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onAudio(ev.data));
  ws.onclose = () => cleanup();
  micNode.port.onmessage = ({ data }) => {
    if (ws && ws.readyState === 1) ws.send(data.pcm);
    const core = document.querySelector(".orb .core");
    const s = $("orb").className.includes("user") ? 1 + Math.min(data.rms * 6, 0.25) : 1;
    core.style.transform = `scale(${s})`;
  };
  player.onDone = (id) => ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "playback_done", reply_id: id }));
  $("start").classList.add("hidden");
  $("end").classList.remove("hidden");
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
      $("lesson-meta").textContent = `Lección ${p.lesson_number} · semana ${p.week} · ${p.title} · ${m.level}`;
      break;
    }
    case "state":
      setOrb(m.state);
      if (m.state in STATUS) setStatus(STATUS[m.state]);
      if (m.state === "listening") $("cap-user").textContent = "";
      break;
    case "clock": {
      const left = Math.max(0, m.total - m.elapsed);
      $("clock").textContent = `${String(Math.floor(left / 60)).padStart(2, "0")}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
      $("ring-fg").style.strokeDashoffset = RING * (1 - Math.min(m.elapsed / m.total, 1));
      $("phase").textContent = PHASES[m.phase] || "";
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
    case "summary":
      showSummary(m.summary);
      ws.close();
      break;
    case "error":
      banner(m.message);
      break;
  }
}

function cleanup() {
  try { micStream && micStream.getTracks().forEach((t) => t.stop()); } catch {}
  try { micCtx && micCtx.close(); } catch {}
  ws = null;
  setOrb("idle");
  $("start").classList.remove("hidden");
  $("start").disabled = false;
  $("start").textContent = "Nueva lección";
  $("end").classList.add("hidden");
}

function setOrb(s) { $("orb").className = `orb ${s}`; }
function setStatus(t) { $("status").textContent = t; }
function banner(t) { $("banner").textContent = t; $("banner").classList.remove("hidden"); }
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function showSummary(s) {
  setStatus("¡Lección terminada!");
  $("phase").textContent = "";
  const el = $("summary");
  if (s.aborted) {
    el.innerHTML = `<h2>Lección interrumpida</h2><p class="muted">La lección fue demasiado corta para guardar progreso.</p>`;
  } else {
    el.innerHTML = `
      <h2>Today's progress</h2>
      <h3>🔥 Fluency</h3><p>${esc(s.fluency_line)}</p>
      ${s.well_done?.length ? `<h3>👏 Well done</h3><ul>${s.well_done.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${s.new_vocabulary?.length ? `<h3>📚 New vocabulary</h3><ul>${s.new_vocabulary.map((v) => `<li><b>${esc(v.term)}</b> — ${esc(v.meaning_es)}</li>`).join("")}</ul>` : ""}
      <h3>⚠️ Let's improve</h3><ul>${(s.improve || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <h3>💬 Useful expression</h3><p class="expr">"${esc(s.useful_expression)}" <span class="muted">— ${esc(s.useful_expression_es)}</span></p>
      <h3>Tomorrow</h3><p>${esc(s.tomorrow)}</p>
      <div class="kpis">
        ${kpi(s.fluency?.speaking_wpm, "palabras/min")}
        ${kpi(s.fluency?.mean_turn_words, "palabras por respuesta")}
        ${kpi(s.fluency?.median_onset_s != null ? s.fluency.median_onset_s + " s" : "–", "antes de responder")}
        ${kpi(s.level, "nivel estimado")}
      </div>`;
  }
  el.classList.remove("hidden");
}
const kpi = (v, label) => `<div class="kpi"><b>${esc(v ?? "–")}</b><span>${esc(label)}</span></div>`;

// ---------------------------------------------------------------- progreso
async function showProgress() {
  const r = await fetch("/api/profile").then((x) => x.json());
  const s = r.snapshot;
  const pts = r.history.filter((h) => h.fluency_index != null);
  const spark = pts.length > 1 ? sparkline(pts.map((p) => p.fluency_index)) : `<p class="muted">El gráfico aparece a partir de la 2ª lección.</p>`;
  const plan = r.next_plan;
  $("progress").innerHTML = `
    <h2>Tu progreso</h2>
    <div class="kpis">
      ${kpi(s.level, "nivel estimado")}
      ${kpi(s.lessons_completed, "lecciones")}
      ${kpi(s.vocab.active ?? 0, "palabras activas")}
      ${kpi(s.vocab.mastered ?? 0, "dominadas")}
    </div>
    <h3>Fluidez (índice por lección)</h3>${spark}
    ${s.strengths.length ? `<h3>Fortalezas</h3><ul>${s.strengths.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${s.weaknesses.length ? `<h3>A mejorar</h3><ul>${s.weaknesses.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${Object.keys(s.skills).length ? `<h3>Gramática (dominio)</h3><div class="chips">${Object.entries(s.skills).map(([k, v]) => `<span class="chip">${esc(k.replace(/_/g, " "))} ${Math.round(v * 100)}%</span>`).join("")}</div>` : ""}
    ${s.recurring_mistakes.length ? `<h3>Errores recurrentes</h3><ul>${s.recurring_mistakes.map((m) => `<li>"${esc(m.original)}" → "${esc(m.corrected)}" <span class="muted">(${m.count}×)</span></li>`).join("")}</ul>` : ""}
    ${r.vocab.length ? `<h3>Vocabulario</h3><div class="chips">${r.vocab.map((v) => `<span class="chip ${v.status}" title="${esc(v.meaning_es)}">${esc(v.term)}</span>`).join("")}</div>` : ""}
    <h3>Próxima lección</h3>
    <p>${esc(plan.title)} <span class="muted">(${esc(plan.domain)} · semana ${plan.week}: ${esc(plan.weekly_goal)})</span></p>
    ${plan.target_grammar.length ? `<p class="muted">Gramática: ${plan.target_grammar.map((g) => esc(g.name)).join(", ")}</p>` : ""}
    <p class="muted">Paciencia aprendida: pausa típica ${s.patience.pause_p50.toFixed(1)} s · pausas largas ${s.patience.pause_p90.toFixed(1)} s · multiplicador ×${s.patience.multiplier.toFixed(2)}</p>`;
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
$("end").onclick = () => ws && ws.send(JSON.stringify({ type: "end" }));
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
document.addEventListener("keydown", (e) => {
  if (e.key === "d" && e.altKey) $("debug").classList.toggle("hidden"); // Alt+D: panel de detección de turnos
});
