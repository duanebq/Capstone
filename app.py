# app.py — Translate • Summarize • Detect Tone (EN-only)
# Upload (PDF/DOCX/Audio/Video), Microphone (no FFmpeg), optional YouTube URL.
# Metrics are silently written to C:\Users\enhan\Desktop\Speech Recognition\results

import os, sys, time, io, tempfile, importlib.util
from typing import Optional, Tuple, Dict, Any
import streamlit as st

st.set_page_config(
    page_title="Translate • Summarize • Detect Tone (EN-only)",
    page_icon="🌐",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths (your exact project)
# ──────────────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Quamina Image\Desktop\Caps"
PDF_PATH       = rf"{BASE}\pdf.py"
WORD_PATH      = rf"{BASE}\word.py"
SPEECH_PATH    = rf"{BASE}\speech.py"
RAG_PATH       = rf"{BASE}\st_rag.py"              # optional
METRICS_PATH   = rf"{BASE}\metrics_silent.py"      # required for metrics
RESULTS_DIR    = rf"{BASE}\results"                # where metrics & artifacts go
os.makedirs(RESULTS_DIR, exist_ok=True)

# Safe imports from absolute paths
def _load_from_path(modname: str, abspath: str):
    try:
        spec = importlib.util.spec_from_file_location(modname, abspath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {abspath}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, e

pdf_mod,   _ = _load_from_path("pdf_local",    PDF_PATH)
word_mod,  _ = _load_from_path("word_local",   WORD_PATH)
speech_mod, _ = _load_from_path("speech_local", SPEECH_PATH)
rag_mod,   _ = _load_from_path("st_rag_local", RAG_PATH)
metrics_mod, metrics_err = _load_from_path("metrics_local", METRICS_PATH)

if metrics_err:
    st.error(f"Metrics module not found at:\n{METRICS_PATH}\n\n{metrics_err}")
    st.stop()

# Soft-guards (don’t block app)
if pdf_mod   and not hasattr(pdf_mod,   "extract_text"): st.warning("pdf.py is missing extract_text(path).")
if word_mod  and not hasattr(word_mod,  "extract_text"): st.warning("word.py is missing extract_text(path).")
if speech_mod and not hasattr(speech_mod, "transcribe"): st.info("speech.py transcribe(path) not found; Whisper fallback will be used.")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY missing. Put it in a .env file or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)

def _chat(prompt: str, model: str = "gpt-4o-mini") -> Tuple[str, Dict[str, Any]]:
    """Return (text, usage_dict)"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful assistant. Keep outputs concise and clear."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    usage_dict = {
        "model": getattr(resp, "model", model),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return text, usage_dict

def translate_from_english(text: str, target_lang: str):
    prompt = (
        f"Translate the following English text into {target_lang}. "
        "Keep technical terms accurate. Preserve bullets and basic formatting.\n\n"
        f"TEXT:\n{text}"
    )
    return _chat(prompt)

def summarize_text_in_lang(text: str, target_lang: str):
    prompt = (
        f"Summarize the following text in {target_lang} using 5–8 concise bullet points. "
        "Be factual and avoid adding new information.\n\n" + text
    )
    return _chat(prompt)

def detect_tone_labels(text: str, target_lang: str):
    prompt = (
        "Classify the dominant tone of the following English text. "
        "Pick 1–3 from: formal, neutral, casual, instructional, persuasive, excited, urgent, empathetic. "
        f"Return the labels translated into {target_lang}, comma-separated.\n\n{text}"
    )
    return _chat(prompt)

# ──────────────────────────────────────────────────────────────────────────────
# Language detect
# ──────────────────────────────────────────────────────────────────────────────
from langdetect import detect
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

# ──────────────────────────────────────────────────────────────────────────────
# Whisper size-safe transcription utilities (avoid 413)
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_FILE_LIMIT = 24_500_000  # ~24.5MB
TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".mp3", ".mpg", ".mpeg", ".mp4", ".mkv", ".avi", ".mov")

def _whisper_single_file(path: str) -> str:
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return (getattr(resp, "text", "") or "").strip()

def _extract_audio_to_wav_16k(src_path: str) -> Tuple[str, float]:
    import numpy as np, soundfile as sf
    from moviepy.editor import AudioFileClip, VideoFileClip
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    clip = None
    try:
        try:
            clip = AudioFileClip(src_path)
        except Exception:
            v = VideoFileClip(src_path)
            clip = v.audio
        if clip is None:
            raise RuntimeError("No audio stream in file")
        arr = clip.to_soundarray(fps=TARGET_SR)
        clip.close()
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            arr = arr.mean(axis=1)
        dur = len(arr) / float(TARGET_SR)
        sf.write(tmp_wav, arr.astype("float32"), TARGET_SR, subtype="PCM_16")
        return tmp_wav, dur
    except Exception as e:
        if clip is not None:
            try: clip.close()
            except Exception: pass
        raise e

def _chunk_and_whisper(wav_16k_path: str) -> Tuple[str, float]:
    import soundfile as sf
    data, sr = sf.read(wav_16k_path, dtype="float32")
    total_samples = len(data)
    dur_s = total_samples / float(sr)

    BYTES_PER_SEC = 32000  # 16k mono, 16-bit ≈ 32KB/s
    max_sec = max(30, int((OPENAI_FILE_LIMIT * 0.9) / BYTES_PER_SEC))
    hop = max_sec * sr

    texts = []
    for start in range(0, total_samples, hop):
        chunk = data[start:start + hop]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, chunk, sr, subtype="PCM_16")
            texts.append(_whisper_single_file(tmp.name))
    return " ".join([t for t in texts if t.strip()]), dur_s

def _openai_transcribe_file_size_safe(path: str) -> Tuple[str, float]:
    try:
        if os.path.getsize(path) <= OPENAI_FILE_LIMIT:
            return _whisper_single_file(path), 0.0
    except Exception:
        pass
    wav, dur = _extract_audio_to_wav_16k(path)
    try:
        if os.path.getsize(wav) <= OPENAI_FILE_LIMIT:
            return _whisper_single_file(wav), dur
        else:
            return _chunk_and_whisper(wav)
    finally:
        try: os.remove(wav)
        except Exception: pass

def transcribe_audio_with_fallback(path: str) -> Tuple[str, float]:
    # Try local speech.py first
    if speech_mod and hasattr(speech_mod, "transcribe"):
        try:
            local_txt = speech_mod.transcribe(path) or ""
            if local_txt.strip():
                return local_txt.strip(), 0.0
        except Exception:
            pass
    # Fallback: robust Whisper
    return _openai_transcribe_file_size_safe(path)

# ──────────────────────────────────────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def _save_audio_bytes_to_wav(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def get_text_from_upload_with_timing(file) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Return (text, audio_seconds, timings). For non-audio: (text, 0.0, timings)."""
    name = file.name.lower()
    path = _save_upload_to_temp(file)
    timings: Dict[str, float] = {}
    try:
        if name.endswith(".pdf"):
            if not pdf_mod:
                st.error("pdf.py not available.");  return None, 0.0, timings
            t0 = time.time()
            text = pdf_mod.extract_text(path)
            timings["extract_s"] = time.time() - t0
            return text, 0.0, timings

        elif name.endswith(".docx"):
            if not word_mod:
                st.error("word.py not available."); return None, 0.0, timings
            t0 = time.time()
            text = word_mod.extract_text(path)
            timings["extract_s"] = time.time() - t0
            return text, 0.0, timings

        elif name.endswith(AUDIO_EXTS):
            t0 = time.time()
            text, aud_sec = transcribe_audio_with_fallback(path)
            timings["transcribe_s"] = time.time() - t0
            return text, aud_sec, timings

        else:
            st.error("Unsupported type. Upload PDF, DOCX, WAV/MP3, or MPG/MP4.")
            return None, 0.0, timings
    finally:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Processing pipeline: translate → summarize → tone (+ timings/usage)
# ──────────────────────────────────────────────────────────────────────────────
def process_text_all(text: str, target_lang: str, show_intermediate: bool = False):
    if not text or not text.strip():
        st.error("No text could be extracted.")
        return None

    if show_intermediate:
        with st.expander("Raw extracted text"):
            st.write(text[:4000] + ("..." if len(text) > 4000 else ""))

    lang = detect_language(text)
    st.info(f"Detected language: **{lang}**")
    if lang != "en":
        st.error("Sorry, this app only supports English input.")
        return None

    timings: Dict[str, float] = {}
    usage: Dict[str, Any] = {}

    t0 = time.time()
    translated, u_tr = translate_from_english(text, target_lang)
    timings["translate_s"] = time.time() - t0
    usage["translation"] = u_tr

    t0 = time.time()
    summary, u_sum = summarize_text_in_lang(text, target_lang)
    timings["summarize_s"] = time.time() - t0
    usage["summary"] = u_sum

    t0 = time.time()
    tone, u_tone = detect_tone_labels(text, target_lang)
    timings["tone_s"] = time.time() - t0
    usage["tone"] = u_tone

    return {
        "translated": translated,
        "summary": summary,
        "tone": tone,
        "usage": usage,
        "timings": timings,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Optional: YouTube
# ──────────────────────────────────────────────────────────────────────────────
def _yt_download_to_temp(url: str) -> Optional[str]:
    import yt_dlp, os, tempfile
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = None
        for k in ("requested_downloads", "requested_formats"):
            rd = info.get(k)
            if rd:
                first = rd[0]
                path = first.get("filepath") or first.get("url")
                if path: break
        if not path:
            path = ydl.prepare_filename(info)
    return path

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🌐 Translate • Summarize • Detect Tone (English-only input)")

with st.sidebar:
    st.header("Settings")
    LANGS = [
        "Spanish","French","German","Italian","Portuguese",
        "Dutch","Russian","Arabic","Turkish","Polish",
        "Chinese (Simplified)","Chinese (Traditional)",
        "Japanese","Korean","Hindi","Bengali","Urdu",
        "Thai","Vietnamese","Indonesian","Malay","Filipino",
        "Swahili","Zulu",
    ]
    target_lang = st.selectbox("Pick your target language 🌍", LANGS, index=0)
    show_intermediate = st.checkbox("Show intermediate outputs", value=False)

input_method = st.radio(
    "Choose input method:",
    ["📁 Upload File", "🎤 Record from Microphone (no FFmpeg)", "📺 YouTube URL"],
    horizontal=True
)

text = None
aud_sec = 0.0
file_name = None
source_kind = None
step_timings: Dict[str, float] = {}

# ---- Upload ----
if input_method == "📁 Upload File":
    uploaded = st.file_uploader(
        "Upload PDF / DOCX / WAV / MP3 / MPG / MP4",
        type=["pdf", "docx", "wav", "mp3", "mpg", "mpeg", "mp4", "mkv", "avi", "mov"],
    )
    if uploaded:
        source_kind = "upload"
        file_name = uploaded.name
        with st.spinner("Extracting / Transcribing content..."):
            text, aud_sec, tdict = get_text_from_upload_with_timing(uploaded)
            step_timings.update(tdict)

# ---- Microphone (no playback; auto-process on stop) ----
elif input_method == "🎤 Record from Microphone (no FFmpeg)":
    source_kind = "mic"
    # Correct usage: st.session_state (no underscore!)
    if "mic_text" not in st.session_state:      st.session_state.mic_text = None
    if "mic_processed" not in st.session_state: st.session_state.mic_processed = False
    if "mic_filename" not in st.session_state:  st.session_state.mic_filename = None

    try:
        from streamlit_mic_recorder import mic_recorder
        st.info("Click Start, speak, then Stop.")
        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            key="mic",
        )
        wav_bytes = audio.get("bytes") if isinstance(audio, dict) else None

        if wav_bytes and not st.session_state.mic_processed:
            with st.spinner("Transcribing audio..."):
                wav_path = _save_audio_bytes_to_wav(wav_bytes)
                t0 = time.time()
                text_val, sec_val = transcribe_audio_with_fallback(wav_path)
                step_timings["transcribe_s"] = time.time() - t0
            if text_val:
                st.session_state.mic_text = text_val
                st.session_state.mic_filename = "microphone_recording.wav"
                st.session_state.mic_processed = True
                aud_sec = float(sec_val or 0.0)
                st.success("✅ Transcription complete. See Results below.")
            else:
                st.error("Transcription returned empty. Try speaking 3–5s and check mic volume.")

        if st.button("🔁 Record again"):
            st.session_state.mic_text = None
            st.session_state.mic_processed = False
            st.session_state.mic_filename = None
            st.rerun()

        if st.session_state.mic_text:
            text = st.session_state.mic_text
            file_name = st.session_state.mic_filename

    except ImportError:
        st.error("Mic widget missing. Install with: pip install streamlit-mic-recorder")

# ---- YouTube ----
elif input_method == "📺 YouTube URL":
    source_kind = "youtube"
    url = st.text_input("Paste a YouTube URL (single video)")
    if st.button("Fetch audio & transcribe") and url:
        try:
            with st.spinner("Downloading audio..."):
                path = _yt_download_to_temp(url)
            if not path:
                st.error("Could not download audio.")
            else:
                with st.spinner("Transcribing..."):
                    t0 = time.time()
                    text, aud_sec = transcribe_audio_with_fallback(path)
                    step_timings["transcribe_s"] = time.time() - t0
                file_name = os.path.basename(path)
        except Exception as e:
            st.error(f"YouTube error: {e}")

# ---- Common results ----
if text:
    result = process_text_all(text, target_lang, show_intermediate)
    if result:
        translated = result["translated"]
        summary    = result["summary"]
        tone       = result["tone"]
        usage_pack = result["usage"]
        step_timings.update(result["timings"])

        st.subheader("✅ Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**📝 Summary ({target_lang})**")
            st.write(summary)
        with c2:
            st.markdown(f"**🎭 Tone ({target_lang})**")
            st.write(tone)

        st.markdown(f"**🌍 Translation (English → {target_lang})**")
        st.write(translated)

        st.divider()
        base_name = os.path.splitext(file_name or "input")[0]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download Translation (.txt)",
                data=translated.encode("utf-8"),
                file_name=f"translation_{base_name}_{target_lang}.txt",
                mime="text/plain",
            )
        with c2:
            st.download_button(
                "📥 Download Summary (.txt)",
                data=summary.encode("utf-8"),
                file_name=f"summary_{base_name}_{target_lang}.txt",
                mime="text/plain",
            )

        # ---- Optional RAG (quietly available if present) ----
        try:
            HAS_RAG = bool(rag_mod and hasattr(rag_mod, "BestRAG"))
        except Exception:
            HAS_RAG = False

        if HAS_RAG:
            with st.expander("🔎 Ask a question about this content (RAG)"):
                q = st.text_input("Your question")
                if q:
                    try:
                        Rag = rag_mod.BestRAG
                        rag = Rag.from_texts([text], metadatas=[{"section": 1}])
                        ans = rag.ask(q)
                        st.markdown("**Answer**")
                        st.write(ans)
                    except Exception as e:
                        st.info(f"RAG unavailable: {e}")

        # ---- Metrics (silent write) ----
        try:
            if metrics_mod and hasattr(metrics_mod, "save_metrics_silently"):
                run_inputs = {
                    "source": source_kind or "unknown",
                    "filename": file_name or "",
                    "target_lang": target_lang,
                    "audio_seconds": float(aud_sec or 0.0),
                }
                usage = {
                    "translation": usage_pack.get("translation", {}),
                    "summary":    usage_pack.get("summary", {}),
                    "tone":       usage_pack.get("tone", {}),
                    "whisper_seconds": float(aud_sec or 0.0),
                }
                paths = metrics_mod.save_metrics_silently(
                    run=run_inputs,
                    transcript=text,
                    translation=translated,
                    summary=summary,
                    tone=tone,
                    usage=usage,
                    out_dir=RESULTS_DIR,
                    reference=None,
                    timings=step_timings,
                )
                st.caption(f"📁 Metrics and artifacts saved to: {RESULTS_DIR} (run id: {paths.get('run_id','?')})")
        except Exception as e:
            st.caption(f"⚙️ Metrics note: {e}")

else:
    st.info("Provide input via Upload, Microphone, or YouTube to begin.")

with st.expander("ℹ️ How to use"):
    st.markdown("""
1) Choose **Upload File**, **Record from Microphone (no FFmpeg)**, or **YouTube URL**.  
2) Input must be **English**; non-English is blocked.  
3) The app translates to your selected language, summarizes in that language, and detects tone.  
4) Optional: ask questions about the transcript (RAG).  
5) Download your translation and summary as `.txt`.  
""")
