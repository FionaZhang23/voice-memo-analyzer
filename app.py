import base64
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


# -----------------------------
# Basic setup
# -----------------------------
load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 25
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3"}

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in .env")

if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
    raise ValueError("Missing AZURE_LANGUAGE_KEY or AZURE_LANGUAGE_ENDPOINT in .env")


# -----------------------------
# Small helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


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


def safe_delete_file(file_path: Path):
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass


# -----------------------------
# Speech-to-Text
# -----------------------------
def transcribe_file_with_azure(file_path: str) -> dict:
    """
    Transcribe one audio file with Azure Speech SDK.
    Returns a dict matching the Part B response structure as closely as possible.
    """

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )

    # Ask for detailed output so we can parse confidence and word timings
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
                # Keep fallback if Azure's detailed JSON shape varies
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
# Azure Language
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
        raise RuntimeError(
            f"Linked entity recognition failed: {linked_entities_result.error}"
        )

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
# Summary builder + TTS
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

    # Neural voice
    speech_config.speech_synthesis_voice_name = voice_name

    # MP3 output
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

        stt_result = transcribe_file_with_azure(str(file_path))
        transcript = stt_result.get("transcript", "").strip()

        if not transcript:
            return error_response(
                "Transcription completed but returned empty text.",
                400,
            )

        analysis_result = analyze_text_with_azure(transcript)
        summary_text = build_summary_text(analysis_result)
        tts_result = synthesize_summary_to_base64(summary_text)

        return jsonify(
            {
                **stt_result,
                "analysis": analysis_result,
                "summary": tts_result,
            }
        ), 200

    except ValueError as e:
        return error_response(str(e), 415)

    except Exception as e:
        return error_response(f"Pipeline failed: {str(e)}", 500)

    finally:
        safe_delete_file(file_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)