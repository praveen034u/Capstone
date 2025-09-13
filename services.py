from typing import Optional, List
from io import BytesIO

import pandas as pd
import PyPDF2
from gtts import gTTS
import google.generativeai as genai

from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

from audio_utils import (
    is_wav,
    is_ogg,
    is_webm,
    wav_samplerate_from_bytes,
)

# Use a current Gemini model that supports text generation
GEMINI_MODEL = "gemini-2.0-flash"

# ---------- Clients ----------
def init_clients(gemini_api_key: str, gcp_sa_info: Optional[dict] = None):
    """
    - Configures Gemini with API key.
    - Returns a Google Speech-to-Text client (using SA info if provided).
    """
    genai.configure(api_key=gemini_api_key)

    if gcp_sa_info:
        credentials = service_account.Credentials.from_service_account_info(gcp_sa_info)
        speech_client = speech.SpeechClient(credentials=credentials)
    else:
        # Uses default app credentials (GOOGLE_APPLICATION_CREDENTIALS) if present
        speech_client = speech.SpeechClient()

    return speech_client

# ---------- Features ----------
def translate_with_gemini(text: str, target_lang_name: str) -> Optional[str]:
    """
    Translate text with Gemini. Returns translated text or None.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = (
            f"Translate the following text to {target_lang_name}. "
            f"Return only the translated text.\n\n{text}"
        )
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip() if resp else None
    except Exception:
        return None


def text_to_speech(text: str, lang_code: str) -> Optional[BytesIO]:
    """
    Generate MP3 audio with gTTS in the selected language code.
    """
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception:
        return None


def transcribe_with_google_stt(
    audio_bytes: bytes,
    speech_client: speech.SpeechClient,
    sample_rate_hz: Optional[int] = None,
    primary_lang: str = "en-US",
    alt_langs: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Try multiple configs so we still get text even if the bytes aren't LINEAR16 WAV.
    Order:
      1) If WAV -> LINEAR16 (with inferred or provided sample_rate)
      2) OGG_OPUS
      3) WEBM_OPUS
      4) ENCODING_UNSPECIFIED (let Google infer from container/headers)
    Returns transcript string or None.
    """
    def _config(encoding, sr=None):
        kw = dict(
            encoding=encoding,
            language_code=primary_lang,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        if sr:
            kw["sample_rate_hertz"] = sr
        if alt_langs:
            kw["alternative_language_codes"] = alt_langs
        return speech.RecognitionConfig(**kw)

    def _recognize(cfg) -> Optional[str]:
        try:
            audio = speech.RecognitionAudio(content=audio_bytes)
            resp = speech_client.recognize(config=cfg, audio=audio)
            texts = []
            for r in resp.results:
                if r.alternatives:
                    texts.append(r.alternatives[0].transcript)
            text = " ".join(texts).strip()
            return text or None
        except Exception:
            return None

    # Build attempts
    attempts = []
    if is_wav(audio_bytes):
        sr = sample_rate_hz or wav_samplerate_from_bytes(audio_bytes)
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.LINEAR16, sr))
    if is_ogg(audio_bytes):
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.OGG_OPUS, sample_rate_hz))
    if is_webm(audio_bytes):
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hz))
    # Always include an inference fallback
    attempts.append(_config(speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED, sample_rate_hz))

    for cfg in attempts:
        out = _recognize(cfg)
        if out:
            return out
    return None


def extract_text_from_file(file) -> Optional[str]:
    """
    Read text from supported file types: txt, pdf, csv, xlsx
    """
    ext = file.name.split(".")[-1].lower()
    try:
        if ext == "txt":
            return file.read().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            reader = PyPDF2.PdfReader(file)
            pages = []
            for page in reader.pages:
                try:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                except Exception:
                    continue
            return "\n".join(pages) if pages else None
        elif ext == "csv":
            df = pd.read_csv(file)
            return df.to_string(index=False)
        elif ext == "xlsx":
            df = pd.read_excel(file)
            return df.to_string(index=False)
        else:
            return None
    except Exception:
        return None
