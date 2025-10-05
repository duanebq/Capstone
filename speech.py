# speech.py — exposes: transcribe(path) -> str
# Uses OpenAI Whisper transcription (no local dependencies).

import os
import tempfile
from functools import lru_cache
from typing import Optional


# -------------------------------------------------------------------
#  LOCAL PIPELINE (DISABLED FOR DEPLOY)
#  You can re-enable later if you add a local 'speech_pipeline.py'
# -------------------------------------------------------------------

TranslationReadyWorkflow = None  # Disabled — avoids import errors on Streamlit Cloud


@lru_cache(maxsize=1)
def _wf() -> Optional["TranslationReadyWorkflow"]:
    # Always returns None since local pipeline is disabled
    return None


def _pipeline_transcribe(path: str) -> Optional[str]:
    # This function is a placeholder for local speech pipelines
    # Always returns None to skip to OpenAI fallback
    return None


# -------------------------------------------------------------------
#  OPENAI WHISPER FALLBACK
# -------------------------------------------------------------------

def _normalize_to_wav16k(path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV for Whisper."""
    import librosa, soundfile as sf
    y, sr = librosa.load(path, sr=16000, mono=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, y, 16000, subtype="PCM_16")
    return tmp.name


def _openai_transcribe(path: str) -> str:
    """Use OpenAI Whisper model to transcribe audio."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    client = OpenAI(api_key=api_key)
    wav = _normalize_to_wav16k(path)
    try:
        with open(wav, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (getattr(resp, "text", "") or "").strip()
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


# -------------------------------------------------------------------
#  MAIN ENTRY POINT
# -------------------------------------------------------------------

def transcribe(path: str) -> str:
    """
    Return a plaintext transcription for the given audio/video file.
    Uses OpenAI Whisper (since local pipeline is disabled for deployment).
    """
    # Try local pipeline (skipped)
    txt = _pipeline_transcribe(path)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # Fallback to OpenAI Whisper
    try:
        return _openai_transcribe(path)
    except Exception:
        return ""
