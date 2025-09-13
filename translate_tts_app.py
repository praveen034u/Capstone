import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
from gtts import gTTS
import PyPDF2

# Gemini (google-generativeai)
import google.generativeai as genai

# Whisper (OpenAI)
from openai import OpenAI

# In-browser mic capture
from streamlit_mic_recorder import mic_recorder


# =========================
# App & Secrets Setup
# =========================
st.set_page_config(page_title="Translate + TTS", layout="centered")
st.title("🎙️ Translate & Speak App")
st.caption("Microphone → Whisper (STT) → Gemini (Translate) → gTTS (Audio)")

# Secrets: prefer Streamlit secrets, fallback to environment
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI (for Whisper STT). Mic mode will be disabled if not provided.
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Use a current Gemini model that supports text generation
GEMINI_MODEL = "gemini-2.0-flash"  # update later if Google changes defaults


# =========================
# Language Map
# =========================
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja"
}


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


def transcribe_audio_bytes(audio_bytes: bytes, filename_hint: str = "speech.wav") -> Optional[str]:
    """
    Save audio bytes to a temp file and transcribe with OpenAI Whisper (whisper-1).
    Auto-detects spoken language. Returns transcript or None.

    Requires OPENAI_API_KEY and network access.
    """
    if client is None:
        st.error("OPENAI_API_KEY is not set. Microphone transcription requires OpenAI Whisper.")
        return None

    try:
        suffix = Path(filename_hint).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return tr.text.strip() if tr and getattr(tr, "text", None) else None
    except Exception as e:
        st.error(f"STT failed: {e}")
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
modes = ["Type Text", "Upload File"]
if client is not None:
    modes.append("Use Microphone")  # only show if OPENAI_API_KEY configured

input_mode = st.radio("Input Type:", modes, horizontal=True)
input_text = ""


# 1) Type
if input_mode == "Type Text":
    input_text = st.text_area("✍️ Enter your text here:", height=150, value=st.session_state.get("input_text_cache", ""))

# 2) Upload
elif input_mode == "Upload File":
    file = st.file_uploader("📤 Upload TXT, PDF, CSV, or Excel", type=["txt", "pdf", "csv", "xlsx"])
    if file:
        input_text = extract_text_from_file(file)
        if input_text:
            st.success("✅ File content loaded.")
            st.text_area("📄 Extracted Text", input_text, height=150)

# 3) Microphone (only if OPENAI_API_KEY present)
elif input_mode == "Use Microphone":
    st.markdown("🎤 Click **Start recording** to speak, then click **Stop recording**.")
    audio_dict = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=True
    )

    if audio_dict and isinstance(audio_dict, dict) and audio_dict.get("bytes"):
        transcript = transcribe_audio_bytes(audio_dict["bytes"], filename_hint="speech.wav")
        if transcript:
            input_text = transcript
            st.success("📝 Transcribed text:")
            st.write(input_text)
        else:
            st.warning("Could not transcribe. Please try again.")

    st.divider()
    st.caption("If your browser blocks microphone access, allow mic permission or use the upload option below.")
    fallback = st.file_uploader("Or upload an audio file (MP3/WebM/WAV/M4A)", type=["mp3", "webm", "wav", "m4a"])
    if fallback:
        transcript = transcribe_audio_bytes(fallback.read(), filename_hint=fallback.name)
        if transcript:
            input_text = transcript
            st.success("📝 Transcribed text (from file):")
            st.write(input_text)


# Cache what user typed or captured so it persists for TTS later
if input_text:
    st.session_state.input_text_cache = input_text


# =========================
# Language Selection
# =========================
target_language = st.selectbox("🌐 Target Language:", list(LANGUAGES.keys()))
lang_code = LANGUAGES[target_language]


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
    **How it works**
    1. Choose your input method (type, upload, or microphone — mic requires OPENAI Whisper).
    2. Translate your text using **Gemini**.
    3. Generate audio with **gTTS** and download the MP3.
    """)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Gemini + Whisper + gTTS")