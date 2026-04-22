# Voice Memo Analyzer

Voice Memo Analyzer is a Flask web app that turns a spoken memo into structured results using Azure AI services.

## What it does

This app runs a 3-stage pipeline:

1. **Speech-to-Text**  
   Upload or record audio and transcribe it with Azure Speech.

2. **Language Analysis**  
   Analyze the transcript with Azure AI Language to extract:
   - key phrases
   - named entities
   - sentiment
   - linked entities

3. **Text-to-Speech**  
   Generate a short spoken summary and return it as playable audio.

The app also sends telemetry to **Azure Application Insights** so pipeline latency and confidence can be monitored.

---

## Main features

- Upload `.wav` or `.mp3` audio files
- Record audio in the browser
- View transcript, entities, sentiment, and key phrases
- Play an AI-generated audio summary
- View telemetry summary from `/telemetry-summary`

---

## Tech stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Azure Services:**
  - Azure Speech
  - Azure AI Language
  - Azure Application Insights

---

## Project structure

```text
voice-memo-analyzer/
├── app.py
├── telemetry.py
├── requirements.txt
├── .env
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    └── index.html
