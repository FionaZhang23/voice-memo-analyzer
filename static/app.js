// ─── DOM refs ────────────────────────────────────────────────────────────────
const audioFileInput      = document.getElementById("audioFile");
const startRecordBtn      = document.getElementById("startRecordBtn");
const stopRecordBtn       = document.getElementById("stopRecordBtn");
const submitBtn           = document.getElementById("submitBtn");
const recordingStatus     = document.getElementById("recordingStatus");
const transcriptText      = document.getElementById("transcriptText");
const keyPhrasesBox       = document.getElementById("keyPhrases");
const sentimentBox        = document.getElementById("sentimentBox");
const entitiesTableBody   = document.querySelector("#entitiesTable tbody");
const linkedEntitiesBox   = document.getElementById("linkedEntities");
const summaryTextBox      = document.getElementById("summaryText");
const summaryAudio        = document.getElementById("summaryAudio");
const requestStatus       = document.getElementById("requestStatus");

const processForm =
  submitBtn?.closest("form") ||
  document.getElementById("processForm") ||
  document.querySelector("form");

// ─── State ───────────────────────────────────────────────────────────────────
let selectedAudioFile = null;

let audioCtx      = null;
let sourceNode    = null;
let processorNode = null;
let gainNode      = null;
let mediaStream   = null;
let pcmChunks     = [];
let totalSamples  = 0;
let sampleRate    = 44100;
let isRecording   = false;

// ─── Status helpers ──────────────────────────────────────────────────────────
function setRequestStatus(msg)   { console.log("[req]", msg); if (requestStatus)  requestStatus.textContent  = msg; }
function setRecordingStatus(msg) { console.log("[rec]", msg); if (recordingStatus) recordingStatus.textContent = msg; }
function nowLabel() { return new Date().toLocaleTimeString(); }

// ─── Clear results UI ────────────────────────────────────────────────────────
function clearResultsForNewRequest() {
  if (transcriptText)    transcriptText.textContent = "Processing new audio...";
  if (keyPhrasesBox)   { keyPhrasesBox.innerHTML = ""; keyPhrasesBox.textContent = "Processing..."; }
  if (sentimentBox)      sentimentBox.textContent = "Processing...";
  if (entitiesTableBody) entitiesTableBody.innerHTML = `<tr><td colspan="4">Processing...</td></tr>`;
  if (linkedEntitiesBox) linkedEntitiesBox.textContent = "Processing...";
  if (summaryTextBox)    summaryTextBox.textContent = "Processing...";
  if (summaryAudio)    { summaryAudio.removeAttribute("src"); summaryAudio.load(); }
}

// ─── File upload ─────────────────────────────────────────────────────────────
if (audioFileInput) {
  audioFileInput.addEventListener("change", () => {
    const file = audioFileInput.files[0];
    if (!file) return;
    selectedAudioFile = file;
    setRecordingStatus(`Selected: ${file.name}`);
    setRequestStatus(`Upload ready: ${file.name} (${file.size} bytes) at ${nowLabel()}`);
  });
}

// ─── START RECORDING ─────────────────────────────────────────────────────────
if (startRecordBtn) {
  startRecordBtn.addEventListener("click", async (e) => {
    e.preventDefault();

    try {
      setRequestStatus("Requesting microphone...");
      setRecordingStatus("Starting...");

      pcmChunks = []; totalSamples = 0;
      selectedAudioFile = null;
      if (audioFileInput) audioFileInput.value = "";

      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // AudioContext must be created inside a user-gesture handler
      audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
      sampleRate = audioCtx.sampleRate;
      console.log("[rec] AudioContext state:", audioCtx.state, "sampleRate:", sampleRate);

      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
        console.log("[rec] resumed, state:", audioCtx.state);
      }

      sourceNode    = audioCtx.createMediaStreamSource(mediaStream);
      processorNode = audioCtx.createScriptProcessor(4096, 2, 1);

      // Muted gain node connected to destination — keeps the audio graph
      // active (required for onaudioprocess to fire) but produces no output,
      // which prevents Chrome from silencing the mic input due to echo cancellation.
      gainNode = audioCtx.createGain();
      gainNode.gain.value = 0;   // silent — we do NOT want mic audio in speakers

      processorNode.onaudioprocess = (ev) => {
        if (!isRecording) return;
        const ch0   = ev.inputBuffer.getChannelData(0);
        const nCh   = ev.inputBuffer.numberOfChannels;
        const copy  = new Float32Array(ch0.length);
        if (nCh >= 2) {
          // Mix stereo to mono — avoids Chrome bug where ch0 is silent
          const ch1 = ev.inputBuffer.getChannelData(1);
          for (let i = 0; i < ch0.length; i++) copy[i] = (ch0[i] + ch1[i]) * 0.5;
        } else {
          copy.set(ch0);
        }
        pcmChunks.push(copy);
        totalSamples += copy.length;
      };

      // source → processor (captures audio)
      // processor → silent gain → destination (keeps graph alive, no speaker output)
      sourceNode.connect(processorNode);
      processorNode.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      isRecording = true;
      setRecordingStatus("Recording… click Stop when done");
      setRequestStatus("Microphone active. Speak now.");
      startRecordBtn.disabled = true;
      if (stopRecordBtn) stopRecordBtn.disabled = false;

    } catch (err) {
      console.error("[rec] start error:", err);
      setRequestStatus(`Mic error: ${err.message}`);
      setRecordingStatus("Failed to start.");
      await stopAndCleanupAudio();
      startRecordBtn.disabled = false;
      if (stopRecordBtn) stopRecordBtn.disabled = true;
    }
  });
}

// ─── STOP RECORDING ──────────────────────────────────────────────────────────
if (stopRecordBtn) {
  stopRecordBtn.addEventListener("click", async (e) => {
    e.preventDefault();

    if (!isRecording) {
      setRecordingStatus("No active recording.");
      if (startRecordBtn) startRecordBtn.disabled = false;
      stopRecordBtn.disabled = true;
      return;
    }

    isRecording = false;
    setRecordingStatus("Processing recording...");

    try {
      console.log("[rec] chunks:", pcmChunks.length, "totalSamples:", totalSamples);

      if (pcmChunks.length === 0 || totalSamples === 0) {
        throw new Error("No audio captured. Please check microphone permissions and try again.");
      }

      const durationSec = totalSamples / sampleRate;
      console.log("[rec] duration:", durationSec.toFixed(2), "s");

      if (durationSec < 0.5) {
        throw new Error("Recording too short. Please speak for at least 1 second.");
      }

      const merged = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of pcmChunks) { merged.set(chunk, offset); offset += chunk.length; }

      const trimmed    = trimSilence(merged, 0.005);
      const toEncode   = trimmed.length > 0 ? trimmed : merged;
      const normalized = normalizeSamples(toEncode, 0.85);
      // Downsample to 16 kHz — Azure Speech SDK expects 16kHz PCM
      const target     = 16000;
      const resampled  = sampleRate === target ? normalized : downsample(normalized, sampleRate, target);
      const wavBlob    = encodeWav(resampled, target);

      console.log("[rec] WAV size:", wavBlob.size, "bytes");
      if (wavBlob.size <= 44) throw new Error("Encoded WAV is empty.");

      selectedAudioFile = new File([wavBlob], "recorded.wav", { type: "audio/wav" });
      setRecordingStatus(`Ready: recorded.wav (${selectedAudioFile.size} bytes, ${durationSec.toFixed(1)}s)`);
      setRequestStatus(`Recording ready at ${nowLabel()}`);

    } catch (err) {
      console.error("[rec] stop error:", err);
      selectedAudioFile = null;
      setRecordingStatus(`Failed: ${err.message}`);
      setRequestStatus(`Recording failed: ${err.message}`);
    } finally {
      await stopAndCleanupAudio();
      if (startRecordBtn) startRecordBtn.disabled = false;
      stopRecordBtn.disabled = true;
    }
  });
}

// ─── Audio cleanup ────────────────────────────────────────────────────────────
async function stopAndCleanupAudio() {
  isRecording = false;
  try { if (processorNode) { processorNode.disconnect(); processorNode.onaudioprocess = null; } } catch(_){}
  try { if (gainNode)    gainNode.disconnect();  } catch(_){}
  try { if (sourceNode)  sourceNode.disconnect(); } catch(_){}
  try { if (mediaStream) mediaStream.getTracks().forEach(t => t.stop()); } catch(_){}
  try { if (audioCtx && audioCtx.state !== "closed") await audioCtx.close(); } catch(_){}
  processorNode = gainNode = sourceNode = mediaStream = audioCtx = null;
}

// ─── PCM helpers ──────────────────────────────────────────────────────────────
function trimSilence(samples, threshold = 0.005) {
  let start = 0, end = samples.length - 1;
  while (start < samples.length && Math.abs(samples[start]) < threshold) start++;
  while (end > start            && Math.abs(samples[end])   < threshold) end--;
  return (start >= end) ? new Float32Array(0) : samples.slice(start, end + 1);
}

function normalizeSamples(samples, targetPeak = 0.85) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) { const v = Math.abs(samples[i]); if (v > peak) peak = v; }
  if (peak < 0.0001) return samples;
  const gain = targetPeak / peak;
  const out  = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = Math.max(-1, Math.min(1, samples[i] * gain));
  return out;
}

function downsample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio     = fromRate / toRate;
  const outLength = Math.floor(samples.length / ratio);
  const out       = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    // Simple linear interpolation between neighbouring source samples
    const srcIdx = i * ratio;
    const lo     = Math.floor(srcIdx);
    const hi     = Math.min(lo + 1, samples.length - 1);
    const frac   = srcIdx - lo;
    out[i]       = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}

function encodeWav(samples, sr) {
  const bps = 2, ch = 1;
  const dataLen = samples.length * bps;
  const buf  = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buf);
  const ws   = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); view.setUint32(4, 36 + dataLen, true);
  ws(8, "WAVE"); ws(12, "fmt ");
  view.setUint32(16, 16,        true);
  view.setUint16(20, 1,         true); // PCM
  view.setUint16(22, ch,        true);
  view.setUint32(24, sr,        true);
  view.setUint32(28, sr*ch*bps, true);
  view.setUint16(32, ch*bps,    true);
  view.setUint16(34, 16,        true);
  ws(36, "data"); view.setUint32(40, dataLen, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ─── Submit ───────────────────────────────────────────────────────────────────
async function submitToProcess(event) {
  if (event) { event.preventDefault(); event.stopPropagation(); }

  if (!selectedAudioFile) { setRequestStatus("Please upload a file or record audio first."); return; }
  if (selectedAudioFile.size === 0) { setRequestStatus("Audio file is empty. Please try again."); return; }

  clearResultsForNewRequest();
  setRequestStatus(`Uploading ${selectedAudioFile.name} (${selectedAudioFile.size} bytes)…`);
  if (submitBtn) submitBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append("audio", selectedAudioFile);

    const response = await fetch("/process", { method: "POST", body: fd });
    const ct = response.headers.get("content-type") || "";

    if (ct.includes("application/json")) {
      const data = await response.json();
      if (!response.ok) { setRequestStatus(`Error ${response.status}: ${data.error || "Unknown"}`); return; }
      renderResults(data);
      setRequestStatus(`Done ✓  HTTP ${response.status} · ${nowLabel()}`);
    } else {
      const txt = await response.text();
      console.error("Non-JSON:", txt);
      setRequestStatus(`Server error (HTTP ${response.status}). Check console.`);
    }
  } catch (err) {
    console.error("submit error:", err);
    setRequestStatus(`Request failed: ${err.message}`);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

if (submitBtn)   submitBtn.addEventListener("click",   submitToProcess);
if (processForm) processForm.addEventListener("submit", submitToProcess);

// ─── Render helpers ───────────────────────────────────────────────────────────
function renderResults(data) {
  if (transcriptText) transcriptText.textContent = data.transcript || "No transcript.";
  renderKeyPhrases(data.analysis?.key_phrases || []);
  renderSentiment(data.analysis?.sentiment    || null);
  renderEntities(data.analysis?.entities      || []);
  renderLinkedEntities(data.analysis?.linked_entities || []);
  if (summaryTextBox) summaryTextBox.textContent = data.summary?.summary_text || "No summary.";
  if (summaryAudio) {
    if (data.summary?.audio_base64) {
      summaryAudio.src = "data:audio/mp3;base64," + data.summary.audio_base64;
      summaryAudio.load();
    } else {
      summaryAudio.removeAttribute("src"); summaryAudio.load();
    }
  }
}

function renderKeyPhrases(phrases) {
  if (!keyPhrasesBox) return;
  keyPhrasesBox.innerHTML = "";
  if (!phrases.length) { keyPhrasesBox.textContent = "No key phrases."; return; }
  phrases.forEach(p => {
    const span = document.createElement("span");
    span.className = "tag"; span.textContent = p;
    keyPhrasesBox.appendChild(span);
  });
}

function renderSentiment(s) {
  if (!sentimentBox) return;
  if (!s) { sentimentBox.textContent = "No sentiment."; return; }
  sentimentBox.innerHTML = `
    <p><strong>Label:</strong> ${s.label}</p>
    <p><strong>Positive:</strong> ${Number(s.positive).toFixed(3)}</p>
    <p><strong>Neutral:</strong>  ${Number(s.neutral).toFixed(3)}</p>
    <p><strong>Negative:</strong> ${Number(s.negative).toFixed(3)}</p>`;
}

function renderEntities(entities) {
  if (!entitiesTableBody) return;
  entitiesTableBody.innerHTML = "";
  if (!entities.length) { entitiesTableBody.innerHTML = `<tr><td colspan="4">No entities.</td></tr>`; return; }
  entities.forEach(en => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(en.text||"")}</td>
      <td>${escapeHtml(en.category||"")}</td>
      <td>${escapeHtml(en.subcategory||"")}</td>
      <td>${en.confidence!=null ? Number(en.confidence).toFixed(3) : ""}</td>`;
    entitiesTableBody.appendChild(tr);
  });
}

function renderLinkedEntities(list) {
  if (!linkedEntitiesBox) return;
  if (!list.length) { linkedEntitiesBox.textContent = "No linked entities."; return; }
  linkedEntitiesBox.innerHTML = list.map(en => {
    const matches = (en.matches||[]).map(m=>`${escapeHtml(m.text)} (${Number(m.confidence).toFixed(3)})`).join(", ");
    const url = en.url ? escapeHtml(en.url) : "";
    return `<div class="linked-entity">
      <p><strong>Name:</strong> ${escapeHtml(en.name||"")}</p>
      <p><strong>Source:</strong> ${escapeHtml(en.data_source||"")}</p>
      <p><strong>URL:</strong> ${url?`<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`:"N/A"}</p>
      <p><strong>Matches:</strong> ${matches||"None"}</p>
    </div>`;
  }).join("");
}

function escapeHtml(v) {
  return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

if (stopRecordBtn) stopRecordBtn.disabled = true;