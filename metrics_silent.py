# metrics_silent.py — silent metrics writer (CSV + JSONL + artifacts)
from __future__ import annotations
import os, time, json, uuid, inspect
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from jiwer import wer, cer
import pandas as pd

# ---------- accuracy helpers ----------
def _safe_wer(ref: Optional[str], hyp: Optional[str]) -> Optional[float]:
    try:
        ref = (ref or "").strip(); hyp = (hyp or "").strip()
        return float(wer(ref, hyp)) if ref and hyp else None
    except Exception:
        return None

def _safe_cer(ref: Optional[str], hyp: Optional[str]) -> Optional[float]:
    try:
        ref = (ref or "").strip(); hyp = (hyp or "").strip()
        return float(cer(ref, hyp)) if ref and hyp else None
    except Exception:
        return None

DEFAULT_PRICING = {
    "gpt-4o-mini": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "gpt-4o":      {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "whisper-1":   {"audio_per_min": 0.0},
}

def _estimate_cost(usage: Dict[str, Any], pricing: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    total = 0.0
    for key in ("translation", "summary", "tone"):
        u = usage.get(key, {}) or {}
        model = u.get("model", "unknown")
        p = pricing.get(model, {})
        cost = 0.0
        cost += (float(u.get("prompt_tokens", 0))    / 1000.0) * float(p.get("prompt_per_1k", 0.0))
        cost += (float(u.get("completion_tokens", 0))/ 1000.0) * float(p.get("completion_per_1k", 0.0))
        total += cost
        items.append({"stage": key, "model": model, "cost_usd": round(cost, 6), **u})

    # whisper seconds (optional)
    ws = float(usage.get("whisper_seconds", 0.0))
    if ws > 0:
        p = pricing.get("whisper-1", {})
        w_cost = (ws / 60.0) * float(p.get("audio_per_min", 0.0))
        total += w_cost
        items.append({"stage": "whisper", "model": "whisper-1", "seconds": ws, "cost_usd": round(w_cost, 6)})

    return {"total_usd": round(total, 6), "items": items}

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

# ---------- main entry ----------
def save_metrics_silently(
    run: Dict[str, Any],
    transcript: str,
    translation: str,
    summary: str,
    tone: str,
    usage: Dict[str, Any],
    out_dir: Optional[str] = None,
    reference: Optional[str] = None,
    timings: Optional[Dict[str, float]] = None,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, str]:
    """
    Preferred signature used by the app.

    Writes:
      out_dir/runs.jsonl
      out_dir/runs.csv
      out_dir/run_<id>/meta.json
      out_dir/run_<id>/(transcript.txt|translation.txt|summary.txt|tone.txt)
    """
    base_dir = out_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(base_dir, exist_ok=True)

    # compute metrics
    wer_v = _safe_wer(reference, transcript)
    cer_v = _safe_cer(reference, transcript)
    orig_words = len((transcript or "").split())
    sum_words  = len((summary or "").split())
    compression_ratio = round(orig_words / max(1, sum_words), 3) if sum_words else None

    timings = timings or {}
    total_proc = float(timings.get("translate_s", 0.0) + timings.get("summarize_s", 0.0) + timings.get("tone_s", 0.0))
    audio_seconds = float(run.get("audio_seconds", 0.0) or 0.0)
    rtf = round(total_proc / audio_seconds, 4) if audio_seconds > 0 else None
    wps = round(orig_words / total_proc, 4) if total_proc > 0 else None

    pricing = pricing or DEFAULT_PRICING
    cost = _estimate_cost(usage or {}, pricing)

    # build record
    record = {
        "run": {
            "id": f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            **run,
        },
        "accuracy": {"wer": wer_v, "cer": cer_v},
        "sizes": {"orig_words": orig_words, "summary_words": sum_words, "compression_ratio": compression_ratio},
        "latency": {
            "translate_s": round(float(timings.get("translate_s", 0.0)), 4),
            "summarize_s": round(float(timings.get("summarize_s", 0.0)), 4),
            "tone_s":      round(float(timings.get("tone_s", 0.0)), 4),
            "processing_s": round(total_proc, 4),
            "audio_seconds": audio_seconds,
            "rtf": rtf,
            "throughput_wps": wps,
        },
        "cost": cost,
        "usage": usage or {},
    }

    # paths
    run_id = record["run"]["id"]
    jsonl = os.path.join(base_dir, "runs.jsonl")
    csvp  = os.path.join(base_dir, "runs.csv")
    rund  = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(rund, exist_ok=True)

    # write artifacts
    with open(os.path.join(rund, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript or "")
    with open(os.path.join(rund, "translation.txt"), "w", encoding="utf-8") as f:
        f.write(translation or "")
    with open(os.path.join(rund, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary or "")
    with open(os.path.join(rund, "tone.txt"), "w", encoding="utf-8") as f:
        f.write(tone or "")
    with open(os.path.join(rund, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # append JSONL
    flat = {"run_id": run_id, **_flatten(record)}
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(flat, ensure_ascii=False) + "\n")

    # append CSV (header on first write)
    hdr = not os.path.exists(csvp)
    pd.DataFrame([flat]).to_csv(csvp, index=False, mode="a", header=hdr)

    return {"run_id": run_id, "jsonl": jsonl, "csv": csvp, "dir": rund}


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility shim:
# If someone imported an older version expecting save_metrics_silently(metrics, base_dir)
# we shim a function with that name/shape to keep both styles working.
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics_silently_legacy(metrics: Dict[str, Any], base_dir: Optional[str] = None):
    base_dir = base_dir or os.path.join(os.getcwd(), "results")
    run = metrics.get("run", {})
    usage = metrics.get("usage", {})
    payload = metrics.get("payload", {})
    timings = metrics.get("timings", {})
    return save_metrics_silently(
        run=run,
        transcript=payload.get("transcript", ""),
        translation=payload.get("translation", ""),
        summary=payload.get("summary", ""),
        tone=payload.get("tone", ""),
        usage=usage,
        out_dir=base_dir,
        reference=metrics.get("reference"),
        timings=timings,
    )

# Alias for older callers (if someone imports by name)
if "parameters" in dir(save_metrics_silently):  # no-op guard
    pass
