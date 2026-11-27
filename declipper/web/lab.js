"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  payload: null,
  view: "T0",
  hypothesis: "conservative",
  spec: null,
  defaults: null,
  truthRevealed: false,
  lastRevealPayload: null,
};

function showError(err) {
  const banner = $("errorBanner");
  banner.hidden = false;
  banner.textContent = `教学台异常：${err && err.message ? err.message : String(err)}`;
  console.error(err);
}
function clearError() {
  $("errorBanner").hidden = true;
}
window.addEventListener("unhandledrejection", (e) => showError(e.reason));

async function getJSON(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.message || `读取失败 ${url}`);
  return data;
}
async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.message || `${resp.status}`);
  return data;
}

function fmt(v, d = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "∞";
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(2);
  return Number(v).toFixed(d);
}

function normSpec(payload) {
  const s = payload.spec;
  return {
    sample_rate: s.sample_rate,
    duration_seconds: s.duration_seconds,
    event_start: s.event_start,
    clip_threshold: s.clip_threshold,
    reveal_threshold: s.reveal_threshold,
    decoy_value: s.decoy_value,
    pre_roll: s.pre_roll,
    post_roll: s.post_roll,
    body: s.body,
    core: s.core,
  };
}

function configureControls(spec) {
  $("fundamentalInput").value = spec.body.fundamental_hz;
  $("bodyAmpInput").value = spec.body.amplitude;
  $("bodyDecayInput").value = spec.body.decay;
  $("harmonicsInput").value = spec.body.harmonics.length;
  $("clipThresholdInput").value = spec.clip_threshold;
  $("revealThresholdInput").value = spec.reveal_threshold;
  $("revealValueInput").value = spec.core.reveal_value;
  const lowAnch = spec.core.low_anchors.reduce((m, a) => Math.max(m, a[1]), 0);
  const highAnch = spec.core.high_anchors.reduce((m, a) => Math.max(m, a[1]), 0);
  $("lowPeakInput").value = lowAnch.toFixed(2);
  $("highPeakInput").value = highAnch.toFixed(2);
  $("decoyInput").value = spec.decoy_value;
}

function specFromControls() {
  const spec = JSON.parse(JSON.stringify(state.spec));
  const fundamental = parseFloat($("fundamentalInput").value);
  const bodyAmp = parseFloat($("bodyAmpInput").value);
  const bodyDecay = parseFloat($("bodyDecayInput").value);
  const harmonics = Math.max(1, Math.min(8, parseInt($("harmonicsInput").value, 10) || 1));
  const lowPeak = parseFloat($("lowPeakInput").value);
  const highPeak = parseFloat($("highPeakInput").value);
  const revealValue = parseFloat($("revealValueInput").value);
  const decoy = parseFloat($("decoyInput").value);
  const clipT = parseFloat($("clipThresholdInput").value);
  const revealT = parseFloat($("revealThresholdInput").value);

  spec.body.fundamental_hz = fundamental;
  spec.body.amplitude = bodyAmp;
  spec.body.decay = bodyDecay;
  // Truncate / extend harmonic set while preserving the first entries.
  const base = spec.body.harmonics.slice(0, harmonics);
  while (base.length < harmonics) {
    const k = base.length + 1;
    base.push({ multiple: k, amplitude: Math.max(0.05, 1.0 / k), phase: (k * 1.3) % 6.283, decay: 1.6 + 0.5 * k });
  }
  spec.body.harmonics = base;

  const len = spec.core.length;
  spec.core.low_anchors = [[0, revealValue], [1, Math.max(revealValue + 0.03, revealT + 0.02)],
                           [Math.floor((len - 1) / 2), lowPeak], [len - 1, Math.max(revealT + 0.01, 0.8 * lowPeak)]];
  spec.core.high_anchors = [[0, revealValue], [1, Math.max(revealValue + 0.03, revealT + 0.02)],
                            [Math.floor((len - 1) / 2), highPeak], [len - 1, Math.max(revealT + 0.01, 0.8 * lowPeak)]];
  spec.core.reveal_value = revealValue;
  spec.decoy_value = decoy;
  spec.clip_threshold = clipT;
  spec.reveal_threshold = revealT;
  return spec;
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(320, rect.width || canvas.clientWidth || 800);
  const cssH = canvas.clientHeight || Number(canvas.getAttribute("height")) || 240;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

function drawSeries(ctx, data, lo, hi, xFor, yFor, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = lo; i <= hi; i++) {
    const x = xFor(i);
    const y = yFor(data[i] || 0);
    if (i === lo) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawObservation() {
  if (!state.payload) return;
  const p = state.payload;
  const key = state.view;
  const obs = p.waveforms.observation[key];
  const canvas = $("obsCanvas");
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080e1f";
  ctx.fillRect(0, 0, w, h);

  const start = p.event.start, end = p.event.end;
  const pad = Math.max(24, Math.round((end - start + 1) * 2.4));
  const lo = Math.max(0, start - pad);
  const hi = Math.min(obs.length - 1, end + pad);
  let peak = 0.5;
  for (let i = lo; i <= hi; i++) peak = Math.max(peak, Math.abs(obs[i] || 0));
  peak *= 1.15;
  const xFor = (i) => ((i - lo) / (hi - lo)) * w;
  const yFor = (v) => h / 2 - (v / peak) * (h / 2 - 16);

  // mark unidentifiable plateau
  ctx.fillStyle = "rgba(85,214,255,.16)";
  ctx.fillRect(xFor(start), 0, Math.max(2, xFor(end) - xFor(start)), h);

  // threshold lines
  const T = p.blind[key].threshold;
  for (const sign of [1, -1]) {
    const y = yFor(sign * T);
    ctx.strokeStyle = "rgba(255,207,110,.7)";
    ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.setLineDash([]);
  }

  // grid zero
  ctx.strokeStyle = "rgba(159,176,208,.25)";
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  drawSeries(ctx, obs, lo, hi, xFor, yFor, "rgba(159,176,208,.9)", 1.2);

  // revealed sample marker (only meaningful at T↑)
  if (key === "T1") {
    const ri = p.event.reveal_index;
    ctx.strokeStyle = "rgba(255,140,70,.95)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xFor(ri), 0); ctx.lineTo(xFor(ri), h); ctx.stroke();
    ctx.fillStyle = "rgba(255,140,70,.95)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("新露出样本", xFor(ri) + 6, 16);
  }

  // labels
  ctx.fillStyle = "rgba(232,238,252,.8)";
  ctx.font = "12px sans-serif";
  ctx.fillText(`阈值 ${T.toFixed(3)} · 样本 ${lo}–${hi}`, 12, h - 12);

  // legend
  $("obsLegend").innerHTML = `
    <span><i class="dot cyan"></i>无法辨认平台 ${start}–${end}</span>
    <span><i class="dot amber"></i>阈值 T=${T.toFixed(3)}</span>
    ${key === "T1" ? `<span><i class="dot orange"></i>抬阈值新露出 index ${p.event.reveal_index}</span>` : ""}
    <span>小信号区逐样本保持</span>`;
}

function currentHyp() {
  const p = state.payload;
  const list = p.blind[state.view].hypotheses;
  return list.find(h => h.name === state.hypothesis) || list[0];
}

function drawHypothesis() {
  if (!state.payload) return;
  const p = state.payload;
  const key = state.view;
  const obs = p.waveforms.observation[key];
  const hypSamples = p.waveforms.blind[`${key}.${state.hypothesis}`];
  const canvas = $("hypCanvas");
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080e1f"; ctx.fillRect(0, 0, w, h);

  const start = p.event.start, end = p.event.end;
  const pad = Math.max(24, Math.round((end - start + 1) * 2.4));
  const lo = Math.max(0, start - pad);
  const hi = Math.min(obs.length - 1, end + pad);
  let peak = 0.5;
  for (let i = lo; i <= hi; i++) {
    peak = Math.max(peak, Math.abs(obs[i] || 0), Math.abs(hypSamples[i] || 0));
  }
  peak *= 1.15;
  const xFor = (i) => ((i - lo) / (hi - lo)) * w;
  const yFor = (v) => h / 2 - (v / peak) * (h / 2 - 16);

  ctx.fillStyle = "rgba(85,214,255,.1)";
  ctx.fillRect(xFor(start), 0, Math.max(2, xFor(end) - xFor(start)), h);
  const T = p.blind[key].threshold;
  for (const sign of [1, -1]) {
    const y = yFor(sign * T);
    ctx.strokeStyle = "rgba(255,207,110,.6)";
    ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = "rgba(159,176,208,.25)";
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  drawSeries(ctx, obs, lo, hi, xFor, yFor, "rgba(255,107,122,.6)", 1.1);
  drawSeries(ctx, hypSamples, lo, hi, xFor, yFor, state.hypothesis === "continuation" ? "#b794ff" : "#62e6a4", 2.0);

  ctx.fillStyle = "rgba(232,238,252,.8)";
  ctx.font = "12px sans-serif";
  ctx.fillText(`红=观测 · ${state.hypothesis === "continuation" ? "紫=延拓先验" : "绿=保守修补"} · 平台区被重建`, 12, h - 12);
}

function renderHypButtons() {
  const p = state.payload;
  const box = $("hypButtons");
  box.innerHTML = "";
  for (const h of p.blind[state.view].hypotheses) {
    const label = h.name === "continuation" ? "先验延拓" : "保守修补";
    const b = document.createElement("button");
    b.textContent = `${label}${h.feasible ? "" : "（不可行）"}`;
    b.className = h.name === state.hypothesis ? "active" : "";
    b.onclick = () => {
      state.hypothesis = h.name;
      renderHypButtons();
      drawHypothesis();
      updateHypAudio();
      updateReclipStats();
    };
    box.appendChild(b);
  }
}

function dataUrl(b64) { return `data:audio/wav;base64,${b64}`; }

function updateObsAudio() {
  const p = state.payload;
  const b64 = p.audio_base64.observation[state.view];
  $("obsAudio").src = dataUrl(b64);
  $("downloadObs").href = dataUrl(b64);
  $("downloadObs").download = `${state.view}.observation.f32.wav`;
}

function updateHypAudio() {
  const p = state.payload;
  const name = `${state.view}.${state.hypothesis}`;
  const url = dataUrl(p.audio_base64.blind[name]);
  $("hypAudio").src = url;
  $("downloadHyp").href = url;
  $("downloadHyp").download = `${name}.f32.wav`;
  const label = state.hypothesis === "continuation" ? "先验延拓" : "保守修补";
  $("hypAudioTitle").textContent = `重建假设：${label}`;

  // "Residual" here is the injected guess x - clip(x): the part of the
  // reconstruction that re-clipping removes, i.e. exactly what the
  // observation cannot reveal. Both hypotheses re-clip back to the same
  // observation, so this is the honest audible difference to compare.
  const inj = p.audio_base64.injected[name];
  if (inj) {
    $("resAudio").src = dataUrl(inj);
    $("downloadRes").href = dataUrl(inj);
    $("downloadRes").download = `${name}.injected.f32.wav`;
  }
  const diffB64 = p.audio_base64.difference[state.view];
  if (diffB64) {
    $("diffAudio").src = dataUrl(diffB64);
    $("downloadDiff").href = dataUrl(diffB64);
    $("downloadDiff").download = `${state.view}.hypothesis-gap.f32.wav`;
  }
}

function updateReclipStats() {
  const h = currentHyp();
  $("reclipStats").innerHTML = `
    <div>峰值 <b>${fmt(h.peak, 4)}</b></div>
    <div>再削顶最大误差 <b class="${h.reclip_from_residual < 2e-5 ? "good" : "bad"}">${fmt(h.reclip_from_residual, 3)}</b></div>
    <div>未削样本误差 <b class="${h.unclipped_max_error < 1e-8 ? "good" : "bad"}">${fmt(h.unclipped_max_error, 3)}</b></div>
    <div>迭代 ${h.iterations} · ${h.stop_reason}</div>
    <div>${h.feasible ? "<span class='good'>通过再削顶+等式的可行候选</span>" : "<span class='bad'>不满足观测约束</span>"}</div>`;
}

function renderMetrics() {
  const p = state.payload;
  const ex = p.exclusion_experiment;
  $("mSameCandidates").textContent = ex.candidate_set_low.join(", ");
  $("mSurvivors").textContent = ex.candidate_set_raised.join(", ") || "（空）";
  $("mExcluded").textContent = ex.excluded_by_raising_threshold.join(", ") || "无";
  $("mReveal").textContent = `index ${ex.reveal_index} 的一个样本`;
  $("mIntact").textContent = ex.intact_region_sample_for_sample_unchanged
    ? "逐样本不变（max Δ=0）" : `有变化 max Δ=${fmt(ex.intact_region_max_change, 4)}`;
  $("mIntact").className = ex.intact_region_sample_for_sample_unchanged ? "good" : "bad";
  $("mShrank").textContent = ex.set_shrank ? "是" : "否";
  $("mShrank").className = ex.set_shrank ? "good" : "bad";
  $("mPassed").textContent = p.passed ? "通过" : "未通过";
  $("mPassed").className = p.passed ? "good" : "bad";
  const mm = p.construction_method_measurement;
  if (mm) {
    $("mMethod").innerHTML = mm.chosen === "construct_identical_observation"
      ? `实测选构造法：削干声变体仍差 ${mm.method_a_clip_known_dry_source.differing_sample_count}/${mm.method_a_clip_known_dry_source.sample_count} 点；构造法差 0`
      : `实测选直接削干声（已一致）`;
    $("mMethod").className = mm.chosen === "construct_identical_observation" ? "good" : "";
  }
}

function renderCandidateTable() {
  const p = state.payload;
  const out = $("candidateTable");
  if (!state.truthRevealed) {
    out.innerHTML = `
      <p>真值尚未揭示。当前只能看到：</p>
      <ul>
        <li>低阈值下与观测一致的候选：<b>${p.exclusion_experiment.candidate_set_low.join(", ")}</b></li>
        <li>抬阈值后存活：<b class="good">${p.exclusion_experiment.candidate_set_raised.join(", ")}</b></li>
        <li>抬阈值后排除：<b class="warn">${p.exclusion_experiment.excluded_by_raising_threshold.join(", ")}</b></li>
      </ul>
      <p class="hint">点击“揭示真值”后才会显示各候选与原音的真实误差与试听；这些数据不参与重建。</p>`;
    $("truthBadge").textContent = "真值未揭示";
    return;
  }
  const truth = p.truth;
  const rows = Object.entries(truth.candidates).map(([label, c]) => {
    const role = c.role === "surviving" ? "存活" : "被排除";
    const audio = c.audio_base64
      ? `<audio controls preload="none" src="data:audio/wav;base64,${c.audio_base64}" style="height:30px;"></audio>`
      : "—";
    return `<tr><td>${label}</td><td>${role}</td><td>${audio}</td></tr>`;
  }).join("");
  let auditRows = "";
  if (p.truth_audit) {
    auditRows = Object.entries(p.truth_audit).map(([name, rows]) => {
      const cells = rows.map(r =>
        `<td>${fmt(r.peak_abs_error_all, 4)}${r.consistent_with_observation ? "" : " ✗"}</td>`
      ).join("");
      return `<tr><td>${name}</td>${cells}</tr>`;
    }).join("");
    const heads = Object.keys(p.truth.candidates).map(l => `<th>${l} 误差</th>`).join("");
    auditRows = `<h3>事后误差（重建 vs 各真值；✗=该候选与观测不一致）</h3>
      <table><thead><tr><th>重建</th>${heads}</tr></thead><tbody>${auditRows}</tbody></table>`;
  }
  out.innerHTML = `
    <h3>候选原音（仅事后核对）</h3>
    <table><thead><tr><th>候选</th><th>角色</th><th>试听</th></tr></thead><tbody>${rows}</tbody></table>
    ${auditRows}
    <p class="hint">${truth.policy}</p>`;
  $("truthBadge").textContent = "真值已揭示（仅事后）";
}

function renderAll() {
  renderMetrics();
  renderHypButtons();
  drawObservation();
  drawHypothesis();
  updateObsAudio();
  updateHypAudio();
  updateReclipStats();
  renderCandidateTable();
}

async function loadLab() {
  clearError();
  $("specStatus").textContent = "合成中…";
  const payload = await getJSON("/api/lab");
  state.payload = payload;
  state.spec = normSpec(payload);
  state.defaults = JSON.parse(JSON.stringify(state.spec));
  state.truthRevealed = false;
  configureControls(state.spec);
  $("specStatus").textContent = `已合成：${payload.sample_rate} Hz · T=${payload.thresholds.T0} → T↑=${payload.thresholds.T1} · ${payload.passed ? "全部检查通过" : "检查未通过"}`;
  $("viewT0").classList.add("active");
  $("viewT1").classList.remove("active");
  state.view = "T0";
  state.hypothesis = "conservative";
  renderAll();
}

async function rebuild() {
  clearError();
  $("rebuildBtn").disabled = true;
  $("rebuildBtn").textContent = "重算中…";
  try {
    const spec = specFromControls();
    const prior = parseFloat($("priorStrengthInput").value);
    const payload = await postJSON("/api/lab", { spec, prior_strength: prior });
    state.payload = payload;
    state.spec = normSpec(payload);
    state.truthRevealed = false;
    $("specStatus").textContent = `已重合成：T=${payload.thresholds.T0} → T↑=${payload.thresholds.T1} · ${payload.passed ? "检查通过" : "检查未通过（检查参数是否满足约束）"}`;
    const names = payload.blind[state.view].hypotheses.map(h => h.name);
    if (!names.includes(state.hypothesis)) state.hypothesis = names[0];
    renderAll();
  } finally {
    $("rebuildBtn").disabled = false;
    $("rebuildBtn").textContent = "重新合成并盲比";
  }
}

async function revealTruth() {
  clearError();
  $("revealTruthBtn").disabled = true;
  try {
    const spec = specFromControls();
    const prior = parseFloat($("priorStrengthInput").value);
    const payload = await postJSON("/api/lab", { spec, prior_strength: prior, reveal_truth: true });
    state.payload = payload;
    state.spec = normSpec(payload);
    state.truthRevealed = true;
    renderAll();
  } finally {
    $("revealTruthBtn").disabled = false;
  }
}

function switchView(key) {
  state.view = key;
  $("viewT0").classList.toggle("active", key === "T0");
  $("viewT1").classList.toggle("active", key === "T1");
  renderHypButtons();
  drawObservation();
  drawHypothesis();
  updateObsAudio();
  updateHypAudio();
  updateReclipStats();
}

function bindEvents() {
  const v0 = $("viewT0"), v1 = $("viewT1");
  if (v0) v0.addEventListener("click", () => switchView("T0"));
  if (v1) v1.addEventListener("click", () => switchView("T1"));
  $("rebuildBtn").addEventListener("click", () => rebuild().catch(showError));
  $("revealTruthBtn").addEventListener("click", () => revealTruth().catch(showError));
  $("resetBtn").addEventListener("click", () => {
    if (state.defaults) {
      configureControls(state.defaults);
      state.spec = JSON.parse(JSON.stringify(state.defaults));
      rebuild().catch(showError);
    }
  });
  let t = null;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { drawObservation(); drawHypothesis(); }, 120);
  });
}

bindEvents();
loadLab().catch(showError);
