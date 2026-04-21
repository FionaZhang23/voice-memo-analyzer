const audioFileInput = document.getElementById("audioFile");
const startRecordBtn = document.getElementById("startRecordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const submitBtn = document.getElementById("submitBtn");
const recordingStatus = document.getElementById("recordingStatus");
const transcriptText = document.getElementById("transcriptText");
const keyPhrasesBox = document.getElementById("keyPhrases");
const sentimentBox = document.getElementById("sentimentBox");
const entitiesTableBody = document.querySelector("#entitiesTable tbody");
const linkedEntitiesBox = document.getElementById("linkedEntities");
const summaryTextBox = document.getElementById("summaryText");
const summaryAudio = document.getElementById("summaryAudio");
const requestStatus = document.getElementById("requestStatus");

let selectedAudioFile = null;
let mediaRecorder = null;
let recordedChunks = [];
let mediaStream = null;

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0];
  if (!file) return;

  selectedAudioFile = file;
  recordingStatus.textContent = `Selected upload: ${file.name}`;
});

startRecordBtn.addEventListener("click", async () => {
  try {
    requestStatus.textContent = "Requesting microphone access...";
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    let mimeType = "";
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    } else if (MediaRecorder.isTypeSupported("audio/webm")) {
      mimeType = "audio/webm";
    }

    mediaRecorder = mimeType
      ? new MediaRecorder(mediaStream, { mimeType })
      : new MediaRecorder(mediaStream);

    recordedChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      try {
        recordingStatus.textContent = "Converting recording to WAV...";
        const recordedBlob = new Blob(recordedChunks, {
          type: mediaRecorder.mimeType || "audio/webm"
        });

        const wavFile = await convertRecordedBlobToWavFile(recordedBlob);
        selectedAudioFile = wavFile;
        recordingStatus.textContent = `Recorded file ready: ${wavFile.name}`;
      } catch (error) {
        console.error(error);
        recordingStatus.textContent = `Recording conversion failed: ${error.message}`;
      } finally {
        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
        }
      }
    };

    mediaRecorder.start();
    recordingStatus.textContent = "Recording...";
    requestStatus.textContent = "Microphone active.";
    startRecordBtn.disabled = true;
    stopRecordBtn.disabled = false;
  } catch (error) {
    console.error(error);
    requestStatus.textContent = `Microphone error: ${error.message}`;
  }
});

stopRecordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    recordingStatus.textContent = "Stopping recording...";
  }
  startRecordBtn.disabled = false;
  stopRecordBtn.disabled = true;
});

submitBtn.addEventListener("click", async () => {
  try {
    if (!selectedAudioFile) {
      requestStatus.textContent = "Please upload a WAV/MP3 file or record audio first.";
      return;
    }

    requestStatus.textContent = "Uploading audio and running /process...";
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("audio", selectedAudioFile);

    const response = await fetch("/process", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      requestStatus.textContent = data.error || "Request failed.";
      return;
    }

    renderResults(data);
    requestStatus.textContent = "Pipeline completed successfully.";
  } catch (error) {
    console.error(error);
    requestStatus.textContent = `Request failed: ${error.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});

function renderResults(data) {
  transcriptText.textContent = data.transcript || "No transcript returned.";

  renderKeyPhrases(data.analysis?.key_phrases || []);
  renderSentiment(data.analysis?.sentiment || null);
  renderEntities(data.analysis?.entities || []);
  renderLinkedEntities(data.analysis?.linked_entities || []);

  summaryTextBox.textContent =
    data.summary?.summary_text || "No summary text returned.";

  if (data.summary?.audio_base64) {
    summaryAudio.src = "data:audio/mp3;base64," + data.summary.audio_base64;
  } else {
    summaryAudio.removeAttribute("src");
  }
}

function renderKeyPhrases(phrases) {
  keyPhrasesBox.innerHTML = "";

  if (!phrases.length) {
    keyPhrasesBox.textContent = "No key phrases detected.";
    return;
  }

  phrases.forEach((phrase) => {
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = phrase;
    keyPhrasesBox.appendChild(span);
  });
}

function renderSentiment(sentiment) {
  if (!sentiment) {
    sentimentBox.textContent = "No sentiment result.";
    return;
  }

  sentimentBox.innerHTML = `
    <p><strong>Label:</strong> ${sentiment.label}</p>
    <p><strong>Positive:</strong> ${Number(sentiment.positive).toFixed(3)}</p>
    <p><strong>Neutral:</strong> ${Number(sentiment.neutral).toFixed(3)}</p>
    <p><strong>Negative:</strong> ${Number(sentiment.negative).toFixed(3)}</p>
  `;
}

function renderEntities(entities) {
  entitiesTableBody.innerHTML = "";

  if (!entities.length) {
    entitiesTableBody.innerHTML = `
      <tr>
        <td colspan="4">No entities detected.</td>
      </tr>
    `;
    return;
  }

  entities.forEach((entity) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(entity.text || "")}</td>
      <td>${escapeHtml(entity.category || "")}</td>
      <td>${escapeHtml(entity.subcategory || "")}</td>
      <td>${entity.confidence !== null && entity.confidence !== undefined ? Number(entity.confidence).toFixed(3) : ""}</td>
    `;
    entitiesTableBody.appendChild(row);
  });
}

function renderLinkedEntities(linkedEntities) {
  if (!linkedEntities.length) {
    linkedEntitiesBox.textContent = "No linked entities detected.";
    return;
  }

  const html = linkedEntities
    .map((entity) => {
      const matches = (entity.matches || [])
        .map((m) => `${escapeHtml(m.text)} (${Number(m.confidence).toFixed(3)})`)
        .join(", ");

      const safeUrl = entity.url ? escapeHtml(entity.url) : "";
      const safeName = escapeHtml(entity.name || "");
      const safeSource = escapeHtml(entity.data_source || "");

      return `
        <div class="linked-entity">
          <p><strong>Name:</strong> ${safeName}</p>
          <p><strong>Source:</strong> ${safeSource}</p>
          <p><strong>URL:</strong> ${
            safeUrl ? `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeUrl}</a>` : "N/A"
          }</p>
          <p><strong>Matches:</strong> ${matches || "None"}</p>
        </div>
      `;
    })
    .join("");

  linkedEntitiesBox.innerHTML = html;
}

async function convertRecordedBlobToWavFile(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  const wavArrayBuffer = encodeAudioBufferToWav(audioBuffer);
  audioContext.close();

  return new File([wavArrayBuffer], "recorded.wav", {
    type: "audio/wav"
  });
}

function encodeAudioBufferToWav(audioBuffer) {
  const numChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const channelData = downmixToMono(audioBuffer);

  const pcmData = floatTo16BitPCM(channelData);
  const wavBuffer = buildWavFile(pcmData, numChannels, sampleRate, 16);

  return wavBuffer;
}

function downmixToMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }

  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);
  const mono = new Float32Array(audioBuffer.length);

  for (let i = 0; i < audioBuffer.length; i++) {
    mono[i] = (left[i] + right[i]) / 2;
  }

  return mono;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);

  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let sample = Math.max(-1, Math.min(1, float32Array[i]));
    sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, sample, true);
  }

  return buffer;
}

function buildWavFile(pcmBuffer, numChannels, sampleRate, bitsPerSample) {
  const headerSize = 44;
  const dataSize = pcmBuffer.byteLength;
  const wavBuffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(wavBuffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
  view.setUint16(32, numChannels * bitsPerSample / 8, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  new Uint8Array(wavBuffer, 44).set(new Uint8Array(pcmBuffer));
  return wavBuffer;
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}