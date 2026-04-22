import os

# Load local .env for localhost development only.
# On Azure App Service, App Settings provide these values.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Telemetry must initialize BEFORE Flask and Azure SDK imports.
from telemetry import init_telemetry
init_telemetry()

import base64
import json
import math
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from opentelemetry import metrics, trace


app = Flask(__name__)

# -----------------------------
# Basic app setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 25
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3", "webm", "ogg"}

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")

if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
    raise ValueError("Missing AZURE_LANGUAGE_KEY or AZURE_LANGUAGE_ENDPOINT")

# -----------------------------
# OpenTelemetry setup
# -----------------------------
tracer = trace.get_tracer("memo-analyzer")
meter = metrics.get_meter("memo-analyzer")

# Using histograms for all per-call numeric metrics so the values show up
# reliably as custom metrics in Azure Monitor.
stt_confidence_metric = meter.create_histogram("stt_confidence")
stt_duration_metric = meter.create_histogram("stt_duration_seconds")
stt_word_count_metric = meter.create_histogram("stt_word_count")
language_entity_count_metric = meter.create_histogram("language_entity_count")
language_keyphrase_count_metric = meter.create_histogram("language_keyphrase_count")
language_sentiment_metric = meter.create_histogram("language_sentiment")
tts_char_count_metric = meter.create_histogram("tts_char_count")

stage_stt_ms_metric = meter.create_histogram("stage_stt_ms")
stage_language_ms_metric = meter.create_histogram("stage_language_ms")
stage_tts_ms_metric = meter.create_histogram("stage_tts_ms")

# -----------------------------
# In-memory telemetry summary log
# -----------------------------
session_log = []


# -----------------------------
# Helper functions
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def error_response(message: str, status_code: int):
    return jsonify({"error": message}), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return error_response(
        f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        413,
    )


def get_language_client() -> TextAnalyticsClient:
    return TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY),
    )


def save_uploaded_audio(audio_file):
    safe_name = secure_filename(audio_file.filename)
    ext = safe_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = TEMP_AUDIO_DIR / unique_name
    audio_file.save(file_path)
    return file_path


def safe_delete_file(file_path):
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass


def timed_stage(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) and return (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def percentile_95(values):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(len(sorted_vals) * 0.95) - 1)
    return sorted_vals[idx]


# -----------------------------
# Pipeline metrics / events
# -----------------------------
def emit_pipeline_metrics(stt_result, language_result, tts_result, stage_timings, audio_format):
    attrs = {
        "audio_format": audio_format,
        "language": stt_result.get("language", "unknown"),
    }

    # STT metrics
    stt_confidence_metric.record(float(stt_result.get("confidence", 0.0)), attrs)
    stt_duration_metric.record(float(stt_result.get("duration_seconds", 0.0)), attrs)
    stt_word_count_metric.record(int(len((stt_result.get("transcript", "") or "").split())), attrs)

    # Language metrics
    language_entity_count_metric.record(int(len(language_result.get("entities", []))), attrs)
    language_keyphrase_count_metric.record(int(len(language_result.get("key_phrases", []))), attrs)

    sentiment_map = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
    }
    language_sentiment_metric.record(
        float(sentiment_map.get(language_result.get("sentiment", {}).get("label", "neutral"), 0.0)),
        attrs,
    )

    # TTS metrics
    tts_char_count_metric.record(int(tts_result.get("char_count", 0)), attrs)

    # Stage latency metrics
    stage_stt_ms_metric.record(float(stage_timings["stt_ms"]), attrs)
    stage_language_ms_metric.record(float(stage_timings["language_ms"]), attrs)
    stage_tts_ms_metric.record(float(stage_timings["tts_ms"]), attrs)


def emit_pipeline_event(stt_result=None, lang_result=None, audio_format=None,
                        success=True, error_stage=None, error_msg=None):
    span = trace.get_current_span()

    if success:
        span.set_attribute("event.name", "pipeline_completed")
        if stt_result:
            span.set_attribute("stt.confidence", float(stt_result.get("confidence", 0.0)))
            span.set_attribute("stt.language", stt_result.get("language", "unknown"))
        if lang_result:
            span.set_attribute("entities.count", int(len(lang_result.get("entities", []))))
            span.set_attribute("sentiment", lang_result.get("sentiment", {}).get("label", "unknown"))
        if audio_format:
            span.set_attribute("audio.format", audio_format)

        span.add_event(
            "pipeline_completed",
            {
                "audio.format": audio_format or "unknown",
                "stt.language": stt_result.get("language", "unknown") if stt_result else "unknown",
                "stt.confidence": float(stt_result.get("confidence", 0.0)) if stt_result else 0.0,
                "entities.count": int(len(lang_result.get("entities", []))) if lang_result else 0,
                "sentiment": lang_result.get("sentiment", {}).get("label", "unknown") if lang_result else "unknown",
            },
        )
    else:
        span.set_attribute("event.name", "pipeline_error")
        span.set_attribute("error.stage", error_stage or "unknown")
        span.set_attribute("error.message", error_msg or "unknown")
        if audio_format:
            span.set_attribute("audio.format", audio_format)

        span.add_event(
            "pipeline_error",
            {
                "audio.format": audio_format or "unknown",
                "error.stage": error_stage or "unknown",
                "error.message": error_msg or "unknown",
            },
        )

        if error_msg:
            span.record_exception(Exception(error_msg))


def log_pipeline_call(stt_result, lang_result, timings, audio_format):
    session_log.append(
        {
            "confidence": float(stt_result.get("confidence", 0.0)),
            "language": stt_result.get("language", "unknown"),
            "entity_count": len(lang_result.get("entities", [])),
            "keyphrase_count": len(lang_result.get("key_phrases", [])),
            "sentiment": lang_result.get("sentiment", {}).get("label", "unknown"),
            "stt_ms": float(timings["stt_ms"]),
            "language_ms": float(timings["language_ms"]),
            "tts_ms": float(timings["tts_ms"]),
            "audio_format": audio_format,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


# -----------------------------
# Speech-to-Text
# -----------------------------
def transcribe_file_with_azure(file_path: str) -> dict:
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    started_at = time.perf_counter()

    try:
        result = recognizer.recognize_once()
    except Exception as e:
        msg = str(e).lower()
        if (
            "invalid header" in msg
            or "unsupported" in msg
            or "audio format" in msg
            or "format" in msg
        ):
            raise ValueError(
                "Unsupported audio format. Please upload a valid .wav or .mp3 file. "
                "If your original file is .m4a, convert it to PCM WAV first."
            )
        raise RuntimeError(f"Speech recognition failed: {str(e)}")

    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text or ""
        language = "en-US"
        confidence = 0.0
        words = []

        if result.json:
            try:
                detailed = json.loads(result.json)
                nbest = detailed.get("NBest", [])
                if nbest:
                    top = nbest[0]
                    confidence = top.get("Confidence", 0.0) or 0.0

                    raw_words = top.get("Words", [])
                    for w in raw_words:
                        offset_ticks = w.get("Offset", 0)
                        duration_ticks = w.get("Duration", 0)

                        offset_seconds = round(offset_ticks / 10_000_000, 3)
                        duration_seconds = round(duration_ticks / 10_000_000, 3)

                        words.append(
                            {
                                "word": w.get("Word", ""),
                                "offset": offset_seconds,
                                "duration": duration_seconds,
                                "confidence": w.get("Confidence"),
                            }
                        )

                primary_language = detailed.get("PrimaryLanguage")
                if isinstance(primary_language, dict):
                    language = primary_language.get("Language", language)
            except Exception:
                pass

        return {
            "transcript": transcript,
            "language": language,
            "duration_seconds": elapsed_seconds,
            "confidence": confidence,
            "words": words,
        }

    if result.reason == speechsdk.ResultReason.NoMatch:
        return {
            "transcript": "",
            "language": "en-US",
            "duration_seconds": elapsed_seconds,
            "confidence": 0.0,
            "words": [],
        }

    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        error_text = cancellation.error_details or "Speech recognition canceled."
        lowered = error_text.lower()

        if (
            "invalid header" in lowered
            or "unsupported" in lowered
            or "audio format" in lowered
            or "format" in lowered
        ):
            raise ValueError(
                "Unsupported audio format. Please upload a valid .wav or .mp3 file. "
                "If your original file is .m4a, convert it to PCM WAV first."
            )

        raise RuntimeError(f"Azure Speech canceled: {error_text}")

    raise RuntimeError("Unknown Speech SDK result state.")


# -----------------------------
# Language Analysis
# -----------------------------
def analyze_text_with_azure(text: str) -> dict:
    client = get_language_client()

    key_phrases_result = client.extract_key_phrases([text])[0]
    if key_phrases_result.is_error:
        raise RuntimeError(f"Key phrase extraction failed: {key_phrases_result.error}")

    entities_result = client.recognize_entities([text])[0]
    if entities_result.is_error:
        raise RuntimeError(f"Entity recognition failed: {entities_result.error}")

    sentiment_result = client.analyze_sentiment([text])[0]
    if sentiment_result.is_error:
        raise RuntimeError(f"Sentiment analysis failed: {sentiment_result.error}")

    linked_entities_result = client.recognize_linked_entities([text])[0]
    if linked_entities_result.is_error:
        raise RuntimeError(f"Linked entity recognition failed: {linked_entities_result.error}")

    entities = []
    for entity in entities_result.entities:
        entities.append(
            {
                "text": entity.text,
                "category": entity.category,
                "subcategory": entity.subcategory,
                "confidence": entity.confidence_score,
            }
        )

    linked_entities = []
    for entity in linked_entities_result.entities:
        linked_entities.append(
            {
                "name": entity.name,
                "url": entity.url,
                "data_source": entity.data_source,
                "matches": [
                    {
                        "text": match.text,
                        "confidence": match.confidence_score,
                    }
                    for match in entity.matches
                ],
            }
        )

    sentiment_scores = sentiment_result.confidence_scores

    return {
        "key_phrases": list(key_phrases_result.key_phrases),
        "entities": entities,
        "sentiment": {
            "label": sentiment_result.sentiment,
            "positive": sentiment_scores.positive,
            "neutral": sentiment_scores.neutral,
            "negative": sentiment_scores.negative,
        },
        "linked_entities": linked_entities,
    }


# -----------------------------
# Summary + TTS
# -----------------------------
def build_summary_text(analysis_result: dict) -> str:
    key_phrases = analysis_result.get("key_phrases", [])
    sentiment = analysis_result.get("sentiment", {}).get("label", "neutral")
    entities = analysis_result.get("entities", [])

    top_phrases = key_phrases[:3]
    if top_phrases:
        phrase_text = ", ".join(top_phrases)
        phrase_sentence = (
            f"Your memo mentions {len(key_phrases)} key topic"
            f"{'s' if len(key_phrases) != 1 else ''}: {phrase_text}."
        )
    else:
        phrase_sentence = "I did not detect any strong key phrases in the memo."

    category_counts = Counter(entity["category"] for entity in entities if entity["category"])
    if category_counts:
        category_parts = []
        for category, count in category_counts.items():
            category_parts.append(f"{count} {category.lower()}")
        entity_sentence = "I detected " + ", ".join(category_parts) + "."
    else:
        entity_sentence = "I did not detect any named entities."

    sentiment_sentence = f"The overall tone is {sentiment}."
    return f"{phrase_sentence} {sentiment_sentence} {entity_sentence}"


def synthesize_summary_to_base64(summary_text: str, voice_name: str = "en-US-JennyNeural") -> dict:
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )

    speech_config.speech_synthesis_voice_name = voice_name
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )

    result = synthesizer.speak_text_async(summary_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_base64 = base64.b64encode(result.audio_data).decode("utf-8")
        return {
            "summary_text": summary_text,
            "audio_base64": audio_base64,
            "char_count": len(summary_text),
            "voice": voice_name,
        }

    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        error_text = cancellation.error_details or "Speech synthesis canceled."
        raise RuntimeError(f"Azure TTS canceled: {error_text}")

    raise RuntimeError("Unknown TTS result state.")


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return error_response("Missing file field 'audio'.", 400)

    audio_file = request.files["audio"]

    if not audio_file or audio_file.filename.strip() == "":
        return error_response("No file selected.", 400)

    if not allowed_file(audio_file.filename):
        return error_response(
            "Unsupported file type. Please upload a .wav or .mp3 file.",
            415,
        )

    file_path = None
    try:
        file_path = save_uploaded_audio(audio_file)
        result = transcribe_file_with_azure(str(file_path))
        return jsonify(result), 200

    except ValueError as e:
        return error_response(str(e), 415)

    except Exception as e:
        return error_response(f"Transcription failed: {str(e)}", 500)

    finally:
        safe_delete_file(file_path)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return error_response("Missing JSON body with a 'text' field.", 400)

    text = (data.get("text") or "").strip()
    if not text:
        return error_response("The 'text' field cannot be empty.", 400)

    try:
        analysis_result = analyze_text_with_azure(text)
        return jsonify(analysis_result), 200

    except Exception as e:
        return error_response(f"Language analysis failed: {str(e)}", 500)


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return error_response("Missing file field 'audio'.", 400)

    audio_file = request.files["audio"]
    audio_format = ""
    if audio_file and audio_file.filename:
        audio_format = audio_file.filename.rsplit(".", 1)[-1].lower()

    if not audio_file or audio_file.filename.strip() == "":
        return error_response("No file selected.", 400)

    if not allowed_file(audio_file.filename):
        return error_response(
            "Unsupported file type. Please upload a .wav or .mp3 file.",
            415,
        )

    file_path = None
    current_stage = "upload"

    with tracer.start_as_current_span("pipeline.process") as root_span:
        root_span.set_attribute("audio.format", audio_format)

        try:
            file_path = save_uploaded_audio(audio_file)

            # Stage 1 — STT
            current_stage = "speech_to_text"
            with tracer.start_as_current_span("stage.speech_to_text") as stt_span:
                stt_result, stt_ms = timed_stage(transcribe_file_with_azure, str(file_path))
                stt_span.set_attribute("stt.confidence", float(stt_result.get("confidence", 0.0)))
                stt_span.set_attribute("stt.word_count", len((stt_result.get("transcript", "") or "").split()))
                stt_span.set_attribute("duration_ms", float(stt_ms))

            transcript = stt_result.get("transcript", "").strip()
            if not transcript:
                emit_pipeline_event(
                    success=False,
                    error_stage=current_stage,
                    error_msg="Transcription completed but returned empty text.",
                    audio_format=audio_format,
                )
                return error_response("Transcription completed but returned empty text.", 400)

            # Stage 2 — Language
            current_stage = "language_analysis"
            with tracer.start_as_current_span("stage.language_analysis") as lang_span:
                lang_result, lang_ms = timed_stage(analyze_text_with_azure, transcript)
                lang_span.set_attribute("entity_count", len(lang_result.get("entities", [])))
                lang_span.set_attribute("sentiment", lang_result.get("sentiment", {}).get("label", "unknown"))
                lang_span.set_attribute("duration_ms", float(lang_ms))

            # Stage 3 — TTS
            current_stage = "text_to_speech"
            summary_text = build_summary_text(lang_result)
            with tracer.start_as_current_span("stage.text_to_speech") as tts_span:
                tts_result, tts_ms = timed_stage(synthesize_summary_to_base64, summary_text)
                tts_span.set_attribute("char_count", len(summary_text))
                tts_span.set_attribute("duration_ms", float(tts_ms))

            stage_timings = {
                "stt_ms": stt_ms,
                "language_ms": lang_ms,
                "tts_ms": tts_ms,
            }

            emit_pipeline_metrics(stt_result, lang_result, tts_result, stage_timings, audio_format)
            emit_pipeline_event(stt_result, lang_result, audio_format, success=True)
            log_pipeline_call(stt_result, lang_result, stage_timings, audio_format)

            return jsonify(
                {
                    **stt_result,
                    "analysis": lang_result,
                    "summary": tts_result,
                }
            ), 200

        except ValueError as e:
            emit_pipeline_event(
                success=False,
                error_stage=current_stage,
                error_msg=str(e),
                audio_format=audio_format,
            )
            return error_response(str(e), 415)

        except Exception as e:
            emit_pipeline_event(
                success=False,
                error_stage=current_stage,
                error_msg=str(e),
                audio_format=audio_format,
            )
            return error_response(f"Pipeline failed: {str(e)}", 500)

        finally:
            safe_delete_file(file_path)


@app.route("/telemetry-summary", methods=["GET"])
def telemetry_summary():
    if not session_log:
        return jsonify({"message": "No calls yet"})

    confidences = [entry["confidence"] for entry in session_log]
    stt_values = [entry["stt_ms"] for entry in session_log]
    lang_values = [entry["language_ms"] for entry in session_log]
    tts_values = [entry["tts_ms"] for entry in session_log]

    return jsonify(
        {
            "total_calls": len(session_log),
            "avg_confidence": round(statistics.mean(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "p95_stt_ms": round(percentile_95(stt_values), 2),
            "avg_stt_ms": round(statistics.mean(stt_values), 2),
            "avg_language_ms": round(statistics.mean(lang_values), 2),
            "avg_tts_ms": round(statistics.mean(tts_values), 2),
            "sentiment_breakdown": {
                "positive": sum(1 for entry in session_log if entry["sentiment"] == "positive"),
                "neutral": sum(1 for entry in session_log if entry["sentiment"] == "neutral"),
                "negative": sum(1 for entry in session_log if entry["sentiment"] == "negative"),
            },
            "calls": session_log[-10:],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)