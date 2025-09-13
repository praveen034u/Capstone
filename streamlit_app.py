import os
from typing import List, Optional

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from services import (
    init_clients,
    translate_with_gemini,
    transcribe_with_google_stt,
    text_to_speech,
    extract_text_from_file,
    GEMINI_MODEL,
)
from audio_utils import (
    safe_mic_recorder,
    wav_samplerate_from_bytes,
    is_wav,
)

# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="Translate + TTS (Google Only)", layout="centered")
st.title("🎙️ Translate & Speak App (Google Only)")
st.caption("Microphone → Google STT → Gemini (Translate) → gTTS (Audio)")

# ---------------------------
# Secrets / Config
# ---------------------------
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
GCP_SA_INFO = st.secrets.get("gcp_service_account", None)

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

try:
    speech_client = init_clients(GEMINI_API_KEY, GCP_SA_INFO)
except Exception as e:
    st.error(f"Client initialization failed: {e}")
    st.stop()

# ---------------------------
# Language maps
# ---------------------------
DISPLAY_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
}

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

# ---------------------------
# UI state
# ---------------------------
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "input_text_cache" not in st.session_state:
    st.session_state.input_text_cache = ""

# ---------------------------
# Input mode
# ---------------------------
input_mode = st.radio("Input Type:", ["Type Text", "Upload File", "Use Microphone"], horizontal=True)
input_text: Optional[str] = ""

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
    audio_dict = safe_mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=True
    )

    if audio_dict and isinstance(audio_dict, dict) and audio_dict.get("bytes"):
        sr = audio_dict.get("sample_rate") or wav_samplerate_from_bytes(audio_dict["bytes"])

        transcript = transcribe_with_google_stt(
            audio_bytes=audio_dict["bytes"],
            speech_client=speech_client,
            sample_rate_hz=sr,
            primary_lang=PRIMARY_SPEECH_LANG,
            alt_langs=ALT_SPEECH_LANGS,
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
        sr_up = wav_samplerate_from_bytes(data) if is_wav(data) else None

        transcript = transcribe_with_google_stt(
            audio_bytes=data,
            speech_client=speech_client,
            sample_rate_hz=sr_up,
            primary_lang=PRIMARY_SPEECH_LANG,
            alt_langs=ALT_SPEECH_LANGS,
        )
        if transcript:
            input_text = transcript
            st.success("📝 Transcribed text (from file):")
            st.write(input_text)

# Persist input for TTS later
if input_text:
    st.session_state.input_text_cache = input_text

# ---------------------------
# Target Language
# ---------------------------
target_language = st.selectbox("🌐 Target Language:", list(DISPLAY_LANGUAGES.keys()))
lang_code = DISPLAY_LANGUAGES[target_language]

# ---------------------------
# Translate
# ---------------------------
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

# Show result if present
if st.session_state.translated_text:
    st.subheader("📘 Latest Translated Text")
    st.write(st.session_state.translated_text)

# ---------------------------
# TTS
# ---------------------------
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

# ---------------------------
# Help
# ---------------------------
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **How it works (Google-only)**
    1. Choose input (Type / Upload / Microphone).
    2. Microphone uses **Google Cloud Speech-to-Text** with auto language detection among common languages.
    3. Translation uses **Gemini**.
    4. Audio is generated with **gTTS** (downloadable MP3).

    **Notes**
    - Mic records WAV/PCM in most browsers; app also tries OGG/WebM variants transparently.
    - If you paste a Google service account into `.streamlit/secrets.toml`, ensure keys are **quoted**:
      ```
      [gcp_service_account]
      "type" = "service_account"
      ...
      ```
    """)
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Google Cloud STT + Gemini + gTTS")