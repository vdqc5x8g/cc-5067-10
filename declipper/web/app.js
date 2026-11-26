"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  manifest: null,
  trackId: null,
  samples: null,
  sampleRate: 8000,
  custom: false,
  secondarySamples: null,
  secondaryName: "",
  selection: null,
  restore: null,
  candidate: "primary",
  audioAObjectUrl: null,
};

function showError(err) {
  const banner = $("errorBanner");
  const message = err && err.message ? err.message : String(err);
  banner.hidden = false;
  banner.textContent = `削波重建界面捕获异常：${message}`;
  console.error(err);
}

function clearError() {
  $("errorBanner").hidden = true;
  $("errorBanner").textContent = "";
}

window.addEventListener("error", (e) => showError(e.error || e.message));
window.addEventListener("unhandledrejection", (e) => showError(e.reason));

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(data.message || `${resp.status} ${resp.statusText}`);
    err.payload = data;
    throw err;
  }
  return data;
}

async function getJSON(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.message || `读取失败 ${url}`);
  return data;
}

function assetUrl(rel) {
  const clean = String(rel || "").replace(/^generated\//, "");
  return `/assets/${clean.split("/").map(encodeURIComponent).join("/")}`;
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "∞";
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(2);
  return Number(v).toFixed(digits);
}

function pct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

async function loadManifest() {
  state.manifest = await getJSON("/api/demo");
  $("thresholdInput").value = state.manifest.threshold;
  const select = $("trackSelect");
  select.innerHTML = "";
  for (const t of state.manifest.tracks) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.title;
    select.appendChild(opt);
  }
  const custom = document.createElement("option");
  custom.value = "__custom__";
  custom.textContent = "已打开的自定义 WAV";
  custom.hidden = true;
  select.appendChild(custom);
  await loadTrack(state.manifest.tracks[0].id);
}

async function loadTrack(trackId) {
  clearError();
  state.trackId = trackId;
  state.custom = false;
  state.restore = null;
  state.candidate = "primary";
  const data = await getJSON(`/api/track-samples?track_id=${encodeURIComponent(trackId)}`);
  state.samples = data.samples;
  state.sampleRate = data.sample_rate;
  state.selection = data.selection ? { ...data.selection } : null;
  if (data.threshold) $("thresholdInput").value = data.threshold;
  const track = state.manifest.tracks.find((t) => t.id === trackId);
  $("audioA").src = assetUrl(track.clipped_url);
  $("secondaryCheck").checked = Boolean(track.secondary_url);
  $("secondaryCheck").disabled = !track.secondary_url;
  $("forceAlignmentCheck").checked = false;
  state.secondarySamples = null;
  $("fileStatus").textContent = track.secondary_url ? "该演示轨道附带未削副麦" : "该轨道无副麦，单轨将给出峰高范围";
  updateSelectionInfo();
  drawMainWaveform();
  clearResult();
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

function drawWaveform(canvas, samples, options = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080e1f";
  ctx.fillRect(0, 0, w, h);

  const range = options.range || [0, samples.length - 1];
  const lo = Math.max(0, Math.floor(range[0]));
  const hi = Math.min(samples.length - 1, Math.ceil(range[1]));
  const count = Math.max(2, hi - lo + 1);
  const rangePeak = options.rangePeak || 1;
  const xFor = (i) => ((i - lo) / (count - 1)) * w;
  const yFor = (v) => h / 2 - (v / rangePeak) * (h / 2 - 14);

  if (options.selection) {
    const a = Math.min(options.selection.start, options.selection.end);
    const b = Math.max(options.selection.start, options.selection.end);
    const x1 = xFor(Math.max(lo, a));
    const x2 = xFor(Math.min(hi, b));
    ctx.fillStyle = "rgba(85,214,255,.13)";
    ctx.fillRect(x1, 0, Math.max(2, x2 - x1), h);
    ctx.strokeStyle = "rgba(85,214,255,.75)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, 0); ctx.lineTo(x1, h);
    ctx.moveTo(x2, 0); ctx.lineTo(x2, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = "rgba(159,176,208,.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  const T = options.threshold;
  if (T && T < rangePeak * 1.5) {
    for (const sign of [1, -1]) {
      const y = yFor(sign * T);
      ctx.strokeStyle = "rgba(255,207,110,.65)";
      ctx.setLineDash([7, 5]);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,207,110,.9)";
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`${sign > 0 ? "+" : "−"}T`, 8, y - 4);
    }
  }

  function drawSeries(data, color, width = 1) {
    if (!data) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    const pixelCount = Math.max(1, Math.floor(w));
    for (let px = 0; px < pixelCount; px++) {
      const i0 = lo + Math.floor((px / pixelCount) * count);
      const i1 = Math.min(hi, lo + Math.floor(((px + 1) / pixelCount) * count));
      let mn = Infinity, mx = -Infinity;
      for (let i = i0; i <= i1; i++) {
        const v = data[i] || 0;
        mn = Math.min(mn, v); mx = Math.max(mx, v);
      }
      if (!Number.isFinite(mn)) continue;
      const x = px + 0.5;
      ctx.moveTo(x, yFor(mn));
      ctx.lineTo(x, yFor(mx));
    }
    ctx.stroke();
  }

  drawSeries(samples, options.originalColor || "rgba(159,176,208,.72)", 1);
  if (options.overlay) drawSeries(options.overlay, options.overlayColor || "#55d6ff", 1.8);

  if (options.label) {
    ctx.fillStyle = "rgba(232,238,252,.72)";
    ctx.font = "12px sans-serif";
    ctx.fillText(options.label, 12, h - 12);
  }
}

function mainRangePeak() {
  let mx = 0.4;
  for (const v of state.samples) mx = Math.max(mx, Math.abs(v));
  return mx * 1.12;
}

function drawMainWaveform() {
  if (!state.samples) return;
  drawWaveform($("waveform"), state.samples, {
    selection: state.selection,
    threshold: parseFloat($("thresholdInput").value),
    rangePeak: mainRangePeak(),
    label: `${state.sampleRate} Hz · ${state.samples.length} samples`,
  });
}

function updateSelectionInfo() {
  if (!state.selection) {
    $("selectionInfo").textContent = "尚未选择";
    return;
  }
  const a = Math.min(state.selection.start, state.selection.end);
  const b = Math.max(state.selection.start, state.selection.end);
  $("selectionInfo").textContent = `样本 ${a}–${b} · ${b - a + 1} 点 · ${((b - a + 1) / state.sampleRate * 1000).toFixed(1)} ms`;
}

function installWaveformInteraction() {
  const canvas = $("waveform");
  let dragging = false;

  function eventIndex(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    return Math.max(0, Math.min(state.samples.length - 1, Math.floor((x / rect.width) * state.samples.length)));
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!state.samples) return;
    clearError();
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    const i = eventIndex(e);
    state.selection = { start: i, end: i };
    updateSelectionInfo();
    drawMainWaveform();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const i = eventIndex(e);
    state.selection.end = i;
    updateSelectionInfo();
    drawMainWaveform();
  });
  function endDrag(e) {
    if (!dragging) return;
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    if (state.selection && Math.abs(state.selection.end - state.selection.start) < 1) {
      // A click still selects one sample; detection expands to the full plateau.
      state.selection.end = state.selection.start + 1;
    }
    updateSelectionInfo();
    drawMainWaveform();
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
}

function clearResult() {
  const c = $("resultCanvas");
  const ctx = c.getContext("2d");
  const { w, h } = setupCanvas(c);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080e1f"; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#9fb0d0"; ctx.font = "14px sans-serif";
  ctx.fillText("运行恢复后，这里显示框选上下文内的削波原声与重建叠加波形", 20, h / 2);
  $("audioB").removeAttribute("src");
  $("downloadRestored").removeAttribute("href");
  $("candidateButtons").innerHTML = "";
  $("normalizePanel").hidden = true;
}

function getCandidateData() {
  const r = state.restore;
  if (!r) return { samples: null, b64: null };
  if (state.candidate === "primary") return { samples: r.samples, b64: r.wav_base64 };
  return {
    samples: r.alternatives[state.candidate],
    b64: r.alternative_wav_base64[state.candidate],
  };
}

function drawResult() {
  if (!state.restore) return;
  const r = state.restore;
  const candidate = getCandidateData();
  const rep = r.result.runs[0];
  const run = rep ? rep.run : { start: state.selection.start, end: state.selection.end };
  const length = run.end - run.start + 1;
  const pad = Math.max(32, Math.round(length * 1.8));
  const lo = Math.max(0, run.start - pad);
  const hi = Math.min(state.samples.length - 1, run.end + pad);
  let peak = 0.4;
  for (let i = lo; i <= hi; i++) {
    peak = Math.max(peak, Math.abs(state.samples[i] || 0), Math.abs(candidate.samples[i] || 0));
  }
  drawWaveform($("resultCanvas"), state.samples, {
    range: [lo, hi],
    overlay: candidate.samples,
    overlayColor: state.candidate === "continuation" ? "#b794ff" : state.candidate === "conservative" ? "#62e6a4" : "#55d6ff",
    threshold: r.threshold,
    rangePeak: peak * 1.12,
    originalColor: "rgba(255,107,122,.55)",
    label: `上下文 ${lo}–${hi}；灰红=削波原声，亮色=${state.candidate}`,
  });
}

function renderCandidateButtons() {
  const box = $("candidateButtons");
  box.innerHTML = "";
  if (!state.restore) return;
  const names = [["primary", "主解"]];
  for (const key of Object.keys(state.restore.alternative_wav_base64 || {})) {
    const label = { conservative: "保守解", continuation: "延拓解", dual: "双轨解" }[key] || key;
    names.push([key, label]);
  }
  for (const [key, label] of names) {
    const b = document.createElement("button");
    b.textContent = label;
    b.className = key === state.candidate ? "active" : "";
    b.onclick = () => {
      state.candidate = key;
      renderCandidateButtons();
      drawResult();
      updateAudioB();
    };
    box.appendChild(b);
  }
}

function updateAudioB() {
  const cand = getCandidateData();
  if (!cand.b64) return;
  const url = `data:audio/wav;base64,${cand.b64}`;
  $("audioB").src = url;
  $("downloadRestored").href = url;
  $("downloadRestored").download = `restored_${state.candidate}.f32.wav`;
  $("audioBTitle").textContent = `重建 B（${state.candidate} · 实际 WAV 导出）`;
}

function setMetric(id, text, cls = "") {
  const el = $(id);
  el.textContent = text;
  el.className = cls;
}

function renderMetrics(data) {
  const result = data.result;
  const rep = result.runs[0];
  const opt = rep ? rep.primary_optimizer : {};
  setMetric("mStatus", result.primary_kind, "info");
  setMetric("mStop", `${result.stop_reason}${result.runs.length > 1 ? `（首段，共 ${result.runs.length} 段）` : ""}`, opt.converged ? "good" : "warn");
  setMetric("mIterations", opt.iterations ?? "—");
  const viol = Number(result.constraint_violation_max || 0);
  setMetric("mViolation", viol.toExponential(3), viol < 1e-6 ? "good" : "bad");
  setMetric("mReclip", fmt(result.errors.reclip_max_error, 8), result.errors.reclip_max_error < 1e-6 ? "good" : "bad");
  setMetric("mUnclipped", fmt(result.errors.unclipped_max_error, 8), result.errors.unclipped_max_error < 1e-8 ? "good" : "bad");
  if (result.feasibility) {
    const feasible = result.feasibility.candidates.filter(c => c.feasible).map(c => c.name);
    const range = result.feasibility.sampled_feasible_peak_range || [];
    setMetric("mFeasible", feasible.join(", ") || "无", feasible.length ? "good" : "bad");
    setMetric("mFeasiblePeakRange", `[${fmt(range[0], 4)}, ${fmt(range[1], 4)}]（候选包络，不是全空间证明）`);
  }
  setMetric("mPeak", fmt(result.region_peak_estimate, 5), "info");
  setMetric("mHardRange", `[${fmt(result.threshold, 3)}, +∞)：只能证明 |x|≥T`);
  const pr = result.prior_peak_range || [];
  setMetric("mPriorRange", `[${fmt(pr[0], 3)}, ${fmt(pr[1], 4)}]：经验斜率/曲率先验外包络`, pr[1] === null ? "warn" : "");
  const certRange = result.certified_peak_range ? `；峰高上界 ${fmt(result.certified_peak_range[1], 4)}` : "";
  setMetric("mCertified", result.certified_peak ? `是：唯一时延 + 已对齐副麦${certRange}` : "否：峰高含先验/歧义，不能当真实余量", result.certified_peak ? "good" : "warn");
  if (data.alignment) {
    const a = data.alignment;
    setMetric("mAlignment", `${a.status} · shift=${a.shift ?? "—"} · ncc=${a.best ? fmt(a.best.ncc, 5) : "—"}`, a.certified ? "good" : "warn");
  } else {
    setMetric("mAlignment", "未使用副麦");
  }
  if (rep) {
    const b = rep.primary_boundary;
    setMetric("mBoundary", `qmax=${fmt(b.max_boundary_curvature, 5)}，斜率跳变=${fmt(b.max_slope_jump, 5)}`);
  }
  const wl = $("warningList");
  wl.innerHTML = "";
  for (const w of result.warnings || []) {
    const div = document.createElement("div");
    div.className = "warning";
    div.textContent = w;
    wl.appendChild(div);
  }
}

function renderNormalizer(data) {
  const panel = $("normalizePanel");
  const p = data.normalized;
  if (!p) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.className = `normalize-panel ${p.certified_headroom_db !== null ? "safe" : ""}`;
  const audio = data.normalized_wav_base64
    ? `<audio controls src="data:audio/wav;base64,${data.normalized_wav_base64}"></audio>` : "";
  panel.innerHTML = `
    <strong>响度归一器：</strong>${p.status || p.policy}<br/>
    输入 RMS ${fmt(p.input_rms_db, 2)} dB → 目标 ${fmt(p.target_rms_db, 1)} dB；
    实际增益 ${fmt(p.gain_db, 3)} dB。<br/>
    认证余量：${p.certified_headroom_db === null ? "<b class='warn'>null（未知，不得把推测峰高当真实余量）</b>" : `${fmt(p.certified_headroom_db, 2)} dB`}；
    被阻止的正向补偿：${fmt(p.boost_blocked_db || 0, 2)} dB。
    <div style="margin-top:10px">${audio}</div>`;
}

async function restore() {
  if (!state.samples || !state.selection) throw new Error("请先在波形上框选至少一个削波区");
  clearError();
  $("restoreBtn").disabled = true;
  $("restoreBtn").textContent = "求解中…";
  try {
    const start = Math.min(state.selection.start, state.selection.end);
    const end = Math.max(state.selection.start, state.selection.end);
    const payload = {
      threshold: parseFloat($("thresholdInput").value),
      start,
      end,
      mode: $("modeSelect").value,
      normalize: $("normalizeCheck").checked,
      target_rms_db: parseFloat($("targetRmsInput").value),
    };
    if (state.custom || !state.trackId || state.trackId === "__custom__") {
      payload.samples = state.samples;
      payload.sample_rate = state.sampleRate;
    } else {
      payload.track_id = state.trackId;
    }
    if ($("secondaryCheck").checked) {
      if (state.secondarySamples) payload.secondary = state.secondarySamples;
      else payload.use_secondary = true;
      payload.alignment_mode = $("forceAlignmentCheck").checked ? "force" : "auto";
    }
    const data = await postJSON("/api/restore", payload);
    state.restore = data;
    state.candidate = "primary";
    renderMetrics(data);
    renderCandidateButtons();
    drawResult();
    updateAudioB();
    renderNormalizer(data);
  } finally {
    $("restoreBtn").disabled = false;
    $("restoreBtn").textContent = "恢复框选削波区";
  }
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function handleWavFile(file) {
  clearError();
  const b64 = await readFileAsBase64(file);
  const data = await postJSON("/api/parse", { wav_base64: b64 });
  state.samples = data.samples;
  state.sampleRate = data.sample_rate;
  state.custom = true;
  state.trackId = "__custom__";
  state.selection = null;
  state.restore = null;
  state.secondarySamples = null;
  $("secondaryCheck").checked = false;
  $("secondaryCheck").disabled = true;
  $("forceAlignmentCheck").checked = false;
  const select = $("trackSelect");
  if (![...select.options].some(o => o.value === "__custom__")) {
    const opt = document.createElement("option");
    opt.value = "__custom__"; opt.textContent = "自定义 WAV"; select.appendChild(opt);
  }
  select.value = "__custom__";
  if (state.audioAObjectUrl) URL.revokeObjectURL(state.audioAObjectUrl);
  state.audioAObjectUrl = URL.createObjectURL(file);
  $("audioA").src = state.audioAObjectUrl;
  $("fileStatus").textContent = `已载入 ${file.name}：${data.samples.length} 样本 / ${data.sample_rate} Hz`;
  updateSelectionInfo();
  drawMainWaveform();
  clearResult();
}

async function handleSecondaryFile(file) {
  const b64 = await readFileAsBase64(file);
  const data = await postJSON("/api/parse", { wav_base64: b64 });
  state.secondarySamples = data.samples;
  state.secondaryName = file.name;
  $("secondaryCheck").checked = true;
  $("secondaryCheck").disabled = false;
  $("fileStatus").textContent = `副麦 ${file.name} 已解析：${data.samples.length} 样本`;
}

function table(headers, rows) {
  const thead = `<tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr>`;
  const tbody = rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

async function showReport(kind) {
  const out = $("reportOutput");
  out.innerHTML = "<p>实际读取报告中…</p>";
  try {
    if (kind === "evaluation") {
      const report = await getJSON(assetUrl(state.manifest.evaluation_report_url));
      const rows = report.items.map(it => {
        const m = it.metrics;
        return [
          `T=${fmt(it.threshold, 2)}`,
          fmt(m.repair_rmse_before, 4),
          `<span class="good">${fmt(m.repair_rmse_after, 4)}</span>`,
          fmt(m.repair_rmse_improvement_ratio, 2) + "×",
          fmt(m.repair_snr_db, 2),
          fmt(m.context_log_spectral_distance_db, 2),
          m.repair_region_covered_by_prior_range ? "<span class='good'>是</span>" : "<span class='bad'>否</span>",
          fmt(m.reclip.reclip_max_error, 2),
        ];
      });
      out.innerHTML = `<p>${report.note}</p>` + table(
        ["削波程度", "区前 RMSE", "区后 RMSE", "改善", "区 SNR dB", "上下文 LSD", "范围覆盖真值", "再削波误差"],
        rows
      );
    } else if (kind === "period") {
      const r = await getJSON(assetUrl(state.manifest.period_offset_report_url));
      const n = r.naive_normalizer;
      const g = r.guarded_normalizer;
      const e = r.evaluation_of_naive_gain_on_truth;
      const c = r.checks;
      out.innerHTML = `
        <h3>${r.scenario}</h3>
        <div class="kv-list">
          <div>加入的周期偏移</div><div>${r.period_samples_added} samples；错误 shift=${r.wrong_alignment.shift}</div>
          <div>错误重建峰高</div><div>${fmt(r.wrong_reconstructed_peak, 5)}（真值仅评估：${fmt(r.true_peak_evaluation_only, 5)}）</div>
          <div>朴素归一器自以为余量</div><div class="bad">${fmt(n.believed_headroom_db, 2)} dB，并给予 ${fmt(n.gain_db, 2)} dB 增益</div>
          <div>施加到真值后的峰值</div><div class="bad">${fmt(e.gained_true_peak, 4)}；超过 ceiling ${e.samples_over_ceiling} 点，超过满刻度 ${e.samples_over_digital_full_scale} 点</div>
          <div>守卫归一器</div><div class="good">认证余量=${g.certified_headroom_db}，增益=${fmt(g.gain_db, 3)} dB，阻止补偿 ${fmt(g.boost_blocked_db, 2)} dB</div>
          <div>检查结果</div><div>${Object.entries(c).map(([k,v]) => `${k}=<b class="${v ? "good" : "bad"}">${v}</b>`).join("<br>")}</div>
        </div>
        <div class="failure-audio">
          <a href="${assetUrl(r.audio.wrong_restored)}" target="_blank">听错误对齐重建</a>
          <a href="${assetUrl(r.audio.naive_master)}" target="_blank">听朴素增益后（评估）</a>
          <a href="${assetUrl(r.audio.guarded_master)}" target="_blank">听守卫增益后</a>
        </div>`;
    } else {
      const url = kind === "ambiguous" ? state.manifest.failure_ambiguous_delay_url : state.manifest.failure_long_plateau_url;
      const r = await getJSON(assetUrl(url));
      if (kind === "ambiguous") {
        const a = r.alignment;
        out.innerHTML = `
          <h3>${r.failure_type}</h3>
          <p>${r.policy}</p>
          <pre>${JSON.stringify({ status: a.status, reason: a.reason, shift: a.shift, candidates: a.candidates }, null, 2)}</pre>
          <div class="failure-audio">
            <a href="${assetUrl("failure_ambiguous_delay/periodic_clipped.f32.wav")}" target="_blank">听周期削波</a>
            <a href="${assetUrl("failure_ambiguous_delay/unclipped_secondary.f32.wav")}" target="_blank">听未削副麦</a>
          </div>`;
      } else {
        const e = r.evaluation_only;
        out.innerHTML = `
          <h3>${r.failure_type}</h3>
          <p>${r.policy}</p>
          ${table(["解", "真实区峰", "重建峰", "区 RMSE 前", "区 RMSE 后", "SNR", "LSD", "范围覆盖"], [
            ["主/延拓", fmt(e.primary.repair_region_true_peak,3), fmt(e.primary.repair_region_restored_peak,3), fmt(e.primary.repair_rmse_before,4), fmt(e.primary.repair_rmse_after,4), fmt(e.primary.repair_snr_db,2), fmt(e.primary.context_log_spectral_distance_db,2), e.primary.repair_region_covered_by_prior_range ? "是" : "否"],
            ["保守", fmt(e.conservative.repair_region_true_peak,3), fmt(e.conservative.repair_region_restored_peak,3), fmt(e.conservative.repair_rmse_before,4), fmt(e.conservative.repair_rmse_after,4), fmt(e.conservative.repair_snr_db,2), fmt(e.conservative.context_log_spectral_distance_db,2), e.conservative.repair_region_covered_by_prior_range ? "是" : "否"],
            ["延拓", fmt(e.continuation.repair_region_true_peak,3), fmt(e.continuation.repair_region_restored_peak,3), fmt(e.continuation.repair_rmse_before,4), fmt(e.continuation.repair_rmse_after,4), fmt(e.continuation.repair_snr_db,2), fmt(e.continuation.context_log_spectral_distance_db,2), e.continuation.repair_region_covered_by_prior_range ? "是" : "否"],
          ])}
          <div class="failure-audio">
            <a href="${assetUrl(r.audio.clipped)}" target="_blank">听长平台削波</a>
            <a href="${assetUrl(r.audio.conservative)}" target="_blank">听保守解</a>
            <a href="${assetUrl(r.audio.continuation)}" target="_blank">听延拓推测解</a>
            <a href="${assetUrl(r.audio.clean_evaluation_only)}" target="_blank">听真值（仅评估）</a>
          </div>
          <pre>${(r.solver.warnings || []).join("\n")}</pre>`;
      }
    }
  } catch (err) {
    out.innerHTML = `<p class="bad">报告加载/渲染失败：${String(err.message || err)}</p>`;
    showError(err);
  }
}

function bindEvents() {
  $("trackSelect").addEventListener("change", (e) => {
    if (e.target.value !== "__custom__") loadTrack(e.target.value).catch(showError);
  });
  $("restoreBtn").addEventListener("click", () => restore().catch(showError));
  $("wavFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) handleWavFile(f).catch(showError);
  });
  $("secondaryFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) handleSecondaryFile(f).catch(showError);
  });
  $("periodCheckBtn").addEventListener("click", () => showReport("period").catch(showError));
  document.querySelectorAll("[data-report]").forEach((b) => {
    b.addEventListener("click", () => showReport(b.dataset.report).catch(showError));
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawMainWaveform();
      if (state.restore) drawResult();
      else clearResult();
    }, 120);
  });
}

bindEvents();
installWaveformInteraction();
loadManifest().catch(showError);
