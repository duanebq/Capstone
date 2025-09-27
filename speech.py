# speech.py — exposes: transcribe(path) -> str
# Tries your local speech_pipeline. If missing, falls back to OpenAI Whisper.

import os, tempfile
from functools import lru_cache
from typing import Optional

# If ffmpeg isn't on PATH for MoviePy, set:
# os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ffmpeg\bin\ffmpeg.exe"

# ----- Try local pipeline first -----
try:
    from speech_pipeline import TranslationReadyWorkflow
except Exception:
    TranslationReadyWorkflow = None

@lru_cache(maxsize=1)
def _wf() -> Optional["TranslationReadyWorkflow"]:
    if TranslationReadyWorkflow is None:
        return None
    return TranslationReadyWorkflow()

def _pipeline_transcribe(path: str) -> Optional[str]:
    wf = _wf()
    if wf is None:
        return None
    audio = wf.load_existing_audio(path, target_sr=16000, export_wav_path=None)
    if audio is None:
        return ""
    res = wf.transcribe_with_confidence(audio)
    if not res:
        return ""
    return (res.get("text") or "").strip()

# ----- OpenAI fallback -----
def _normalize_to_wav16k(path: str) -> str:
    import librosa, soundfile as sf
    y, sr = librosa.load(path, sr=16000, mono=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, y, 16000, subtype="PCM_16")
    return tmp.name

def _openai_transcribe(path: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    wav = _normalize_to_wav16k(path)
    with open(wav, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return (getattr(resp, "text", "") or "").strip()

def transcribe(path: str) -> str:
    """
    Return a plaintext transcription for the given audio/video file.
    Uses local pipeline if available, else OpenAI Whisper.
    """
    try:
        txt = _pipeline_transcribe(path)
        if isinstance(txt, str) and txt:
            return txt
    except Exception:
        pass

    try:
        if os.getenv("OPENAI_API_KEY"):
            return _openai_transcribe(path)
    except Exception:
        pass

    return ""
