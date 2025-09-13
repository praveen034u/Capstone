import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
import PyPDF2
import tempfile
import google.generativeai as genai
import openai
from pydub import AudioSegment

# ---------------------
# API Keys from secrets
# ---------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------
# Language Map
# ---------------------
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja"
}

# ---------------------
# Helper Functions
# ---------------------
def extract_text_from_file(file):
    ext = file.name.split(".")[-1].lower()
    try:
        if ext == "txt":
            return file.read().decode("utf-8")
        elif ext == "pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif ext == "csv":
            df = pd.read_csv(file)
            return df.to_string(index=False)
        elif ext == "xlsx":
            df = pd.read_excel(file)
            return df.to_string(index=False)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
    return None


def translate_with_gemini(text, target_lang):
    prompt = f"Translate the following text to {target_lang}:\n\n{text}"
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini translation error: {e}")
        return None


def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None


def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        audio = open(tmp_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio)
        return transcript['text']
    except Exception as e:
        st.error(f"STT failed: {e}")
        return None


# ---------------------
# Streamlit App UI
# ---------------------
st.set_page_config(page_title="Translate + TTS", layout="centered")
st.title("🎙️ Translate & Speak App")
st.caption("Google Gemini + OpenAI Whisper + gTTS")

input_mode = st.radio("Input Type:", ["Type Text", "Upload File", "Use Microphone"])
input_text = ""

# SESSION STATE to hold translated text
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

# 1. Input Sources
if input_mode == "Type Text":
    input_text = st.text_area("✍️ Enter your text here:", height=150)

elif input_mode == "Upload File":
    file = st.file_uploader("📤 Upload TXT, PDF, CSV, or Excel", type=["txt", "pdf", "csv", "xlsx"])
    if file:
        input_text = extract_text_from_file(file)

elif input_mode == "Use Microphone":
    st.markdown("🎤 Upload your voice recording (MP3/WebM/WAV):")
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "webm", "wav", "m4a"])
    if audio_file:
        input_text = transcribe_audio(audio_file.read())
        if input_text:
            st.success("📝 Transcribed text:")
            st.write(input_text)

# 2. Language Selection
target_language = st.selectbox("🌐 Target Language:", list(LANGUAGES.keys()))
lang_code = LANGUAGES[target_language]

# 3. Translate Button
if st.button("🔄 Translate Text"):
    if not input_text.strip():
        st.warning("No input to translate.")
    else:
        with st.spinner("Translating via Gemini..."):
            translated = translate_with_gemini(input_text, target_language)
        if translated:
            st.session_state.translated_text = translated
            st.success("✅ Translation complete")
            st.text_area("📘 Translated Text", translated, height=200)

# 4. Show previously translated text if any
if st.session_state.translated_text:
    st.subheader("📘 Latest Translated Text")
    st.write(st.session_state.translated_text)

# 5. Text-to-Speech Button
if st.button("🔊 Generate Audio"):
    tts_text = st.session_state.translated_text.strip() or input_text.strip()
    if not tts_text:
        st.warning("Please enter or translate text before generating audio.")
    else:
        with st.spinner("Generating audio..."):
            audio_data = text_to_speech(tts_text, lang_code)
        if audio_data:
            st.audio(audio_data, format="audio/mp3")
            st.download_button("⬇️ Download MP3", data=audio_data, file_name=f"tts_{lang_code}.mp3", mime="audio/mpeg")

# 6. Info
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    - Choose your input method (type, upload, or voice).
    - Translate your text using **Gemini API**.
    - Then generate audio with **gTTS** or directly convert existing text.
    """)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Gemini + Whisper + gTTS")
