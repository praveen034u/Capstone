from typing import Optional
import io
import wave

from streamlit_mic_recorder import mic_recorder

# --- sniffers ---
def is_wav(bytes_: bytes) -> bool:
    # Basic RIFF/WAVE header check: 'RIFF'....'WAVE'
    return len(bytes_) >= 12 and bytes_[:4] == b"RIFF" and bytes_[8:12] == b"WAVE"

def is_ogg(bytes_: bytes) -> bool:
    # OGG magic: "OggS"
    return len(bytes_) >= 4 and bytes_[:4] == b"OggS"

def is_webm(bytes_: bytes) -> bool:
    # WebM/Matroska typically starts with EBML magic: 0x1A 45 DF A3
    return len(bytes_) >= 4 and bytes_[:4] == b"\x1A\x45\xDF\xA3"

def wav_samplerate_from_bytes(wav_bytes: bytes) -> Optional[int]:
    """Extract WAV samplerate from header, if possible."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.getframerate()
    except Exception:
        return None

# --- mic safety wrapper ---
def safe_mic_recorder(**kwargs):
    """
    Wrap mic_recorder to survive cases where component returns a dict missing keys
    (e.g., 'sample_rate') and raises KeyError internally. Return None instead of crashing.
    """
    try:
        return mic_recorder(**kwargs)
    except KeyError:
        return None
    except Exception:
        # Keep UI alive; caller can decide what to show
        return None
