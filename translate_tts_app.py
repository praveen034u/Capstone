import os
import io
import wave
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import streamlit as st
import pandas as pd
from gtts import gTTS
import PyPDF2
import google.generativeai as genai

# Google Cloud Speech-to-Text
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

# In-browser microphone
from streamlit_mic_recorder import mic_recorder


# =========================
# Page & Secrets
# =========================
st.set_page_config(page_title="Translate + TTS (Powered By Gemini AI)", layout="centered")
st.title("🎙️ Translate & Speak App (Powered By Gemini AI)")
st.caption("Microphone → Google STT → Gemini (Translate) → gTTS (Audio)")

# Prefer Streamlit Secrets, fallback to environment
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()

# Option A (recommended on Streamlit Cloud): service account pasted under a TOML table
# [gcp_service_account] with quoted keys in secrets.toml
GCP_SA_INFO = st.secrets.get("gcp_service_account", None)

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

# Configure Google Speech Client
try:
    if GCP_SA_INFO:
        credentials = service_account.Credentials.from_service_account_info(GCP_SA_INFO)
        speech_client = speech.SpeechClient(credentials=credentials)
    else:
        # Uses ADC if available (e.g., GOOGLE_APPLICATION_CREDENTIALS)
        speech_client = speech.SpeechClient()
except Exception as e:
    st.error(f"Failed to create Google Speech client: {e}")
    st.stop()

# Current Gemini model
GEMINI_MODEL = "gemini-2.0-flash"


# =========================
# Language Maps
# =========================
DISPLAY_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
}

# For Speech-to-Text detection, we provide a primary language and alternatives
PRIMARY_SPEECH_LANG = "en-US"
ALT_SPEECH_LANGS: List[str] = [
    "hi-IN",        # Hindi
    "es-ES",        # Spanish (Spain)
    "es-MX",        # Spanish (Mexico)
    "fr-FR",        # French
    "de-DE",        # German
    "cmn-Hans-CN",  # Chinese Mandarin (Simplified)
    "ja-JP",        # Japanese
]


# =========================
# Helpers
# =========================
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
            st.warning("Unsupported file type. Please upload TXT/PDF/CSV/XLSX.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


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
        return (resp.text or "").strip() if resp else None
    except Exception as e:
        st.error(f"Gemini translation error: {e}")
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
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None


# --- Audio sniffers ---
def _is_wav(bytes_: bytes) -> bool:
    # Basic RIFF/WAVE header check: 'RIFF'....'WAVE'
    return len(bytes_) >= 12 and bytes_[:4] == b"RIFF" and bytes_[8:12] == b"WAVE"

def _is_ogg(bytes_: bytes) -> bool:
    # OGG magic: "OggS"
    return len(bytes_) >= 4 and bytes_[:4] == b"OggS"

def _is_webm(bytes_: bytes) -> bool:
    # WebM/Matroska typically starts with EBML magic: 0x1A 45 DF A3
    return len(bytes_) >= 4 and bytes_[:4] == b"\x1A\x45\xDF\xA3"

def _wav_samplerate_from_bytes(wav_bytes: bytes) -> Optional[int]:
    """
    Extract WAV samplerate from header, if possible.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.getframerate()
    except Exception:
        return None


def transcribe_with_google_stt(
    audio_bytes: bytes,
    sample_rate_hz: Optional[int] = None,
    primary_lang: str = PRIMARY_SPEECH_LANG,
    alt_langs: Optional[List[str]] = None,
    debug: bool = False,
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
            if debug:
                st.write({"stt_encoding": cfg.encoding.name, "response_results": len(resp.results)})
            texts = []
            for r in resp.results:
                if r.alternatives:
                    texts.append(r.alternatives[0].transcript)
            text = " ".join(texts).strip()
            return text or None
        except Exception as e:
            if debug:
                st.warning(f"Google STT error with {cfg.encoding.name}: {e}")
            return None

    # Build attempts
    attempts = []
    if _is_wav(audio_bytes):
        sr = sample_rate_hz or _wav_samplerate_from_bytes(audio_bytes)
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.LINEAR16, sr))
    if _is_ogg(audio_bytes):
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.OGG_OPUS, sample_rate_hz))
    if _is_webm(audio_bytes):
        attempts.append(_config(speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, sample_rate_hz))
    # Always include an inference fallback
    attempts.append(_config(speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED, sample_rate_hz))

    for cfg in attempts:
        out = _recognize(cfg)
        if out:
            return out
    return None


# --- Mic stability helpers ---
def _safe_mic_recorder(**kwargs):
    """
    Wrap mic_recorder to survive cases where component returns a dict missing keys
    (e.g., 'sample_rate') and raises KeyError internally. Return None instead of crashing.
    """
    try:
        return mic_recorder(**kwargs)
    except KeyError:
        return None
    except Exception as e:
        st.warning(f"Mic component error: {type(e).__name__}")
        return None


# =========================
# UI State
# =========================
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "input_text_cache" not in st.session_state:
    st.session_state.input_text_cache = ""


# =========================
# Input Mode
# =========================
input_mode = st.radio("Input Type:", ["Type Text", "Upload File", "Use Microphone"], horizontal=True)
input_text = ""


# 1) Type
if input_mode == "Type Text":
    input_text = st.text_area(
        "✍️ Enter your text here:",
        height=150,
        value=st.session_state.get("input_text_cache", "")
    )

# 2) Upload
elif input_mode == "Upload File":
    file = st.file_uploader("📤 Upload TXT, PDF, CSV, or Excel", type=["txt", "pdf", "csv", "xlsx"])
    if file:
        input_text = extract_text_from_file(file)
        if input_text:
            st.success("✅ File content loaded.")
            st.text_area("📄 Extracted Text", input_text, height=150)

# 3) Microphone (Google STT)
elif input_mode == "Use Microphone":
    st.markdown("🎤 Click **Start recording** to speak, then click **Stop recording**.")
    debug_stt = st.toggle("Debug STT", value=False, help="Show raw STT responses and configs")

    audio_dict = _safe_mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=True
    )

    # audio_dict: may contain 'bytes' and sometimes 'sample_rate'. Be defensive.
    if audio_dict and isinstance(audio_dict, dict) and audio_dict.get("bytes"):
        sr = audio_dict.get("sample_rate") or _wav_samplerate_from_bytes(audio_dict["bytes"])

        transcript = transcribe_with_google_stt(
            audio_bytes=audio_dict["bytes"],
            sample_rate_hz=sr,                 # okay if None
            primary_lang=PRIMARY_SPEECH_LANG,  # bias to en-US
            alt_langs=ALT_SPEECH_LANGS,        # auto-detect among these
            debug=debug_stt,
        )
        if transcript:
            input_text = transcript
            st.success("📝 Transcribed text:")
            st.write(input_text)
        else:
            st.warning("Could not transcribe. Please try again. (Try speaking a bit longer and clearly.)")

    st.divider()
    st.caption("If your browser blocks mic access, allow permission or use the upload option.")
    fallback = st.file_uploader("Or upload a short audio clip (WAV/OGG/WebM)", type=["wav", "ogg", "webm"])
    if fallback:
        data = fallback.read()
        sr_up = _wav_samplerate_from_bytes(data) if _is_wav(data) else None

        transcript = transcribe_with_google_stt(
            audio_bytes=data,
            sample_rate_hz=sr_up,
            primary_lang=PRIMARY_SPEECH_LANG,
            alt_langs=ALT_SPEECH_LANGS,
            debug=debug_stt,
        )
        if transcript:
            input_text = transcript
            st.success("📝 Transcribed text (from file):")
            st.write(input_text)


# Persist current input text for later TTS
if input_text:
    st.session_state.input_text_cache = input_text


# =========================
# Target Language Selection
# =========================
target_language = st.selectbox("🌐 Target Language:", list(DISPLAY_LANGUAGES.keys()))
lang_code = DISPLAY_LANGUAGES[target_language]


# =========================
# Translate
# =========================
if st.button("🔄 Translate Text"):
    source = (input_text or "").strip()
    if not source:
        st.warning("No input to translate.")
    else:
        with st.spinner("Translating via Gemini..."):
            translated = translate_with_gemini(source, target_language)
        if translated:
            st.session_state.translated_text = translated
            st.success("✅ Translation complete")
            st.text_area("📘 Translated Text", translated, height=200)


# Show the last translated text if any
if st.session_state.translated_text:
    st.subheader("📘 Latest Translated Text")
    st.write(st.session_state.translated_text)


# =========================
# TTS
# =========================
if st.button("🔊 Generate Audio"):
    tts_text = (st.session_state.translated_text or st.session_state.input_text_cache or "").strip()
    if not tts_text:
        st.warning("Please enter or translate text before generating audio.")
    else:
        with st.spinner("Generating audio..."):
            audio_data = text_to_speech(tts_text, lang_code)
        if audio_data:
            st.audio(audio_data, format="audio/mp3")
            st.download_button(
                "⬇️ Download MP3",
                data=audio_data,
                file_name=f"tts_{lang_code}.mp3",
                mime="audio/mpeg"
            )


# =========================
# Help
# =========================
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **How it works (Google-only)**
    1. Choose input (Type / Upload / Microphone).
    2. Microphone uses **Google Cloud Speech-to-Text** with auto language detection among common languages.
    3. Translation uses **Gemini**.
    4. Audio is generated with **gTTS** (downloadable MP3).

    **Notes**
    - Mic records **WAV/PCM** in most browsers; we also try OGG/WebM variants.
    - If you paste a Google service account into `.streamlit/secrets.toml`, ensure keys are **quoted**:
      ```
      [gcp_service_account]
      "type" = "service_account"
      ...
      ```
    """)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Google Cloud STT + Gemini + gTTS")
