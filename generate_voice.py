"""Step 2: voiceover with Gemini TTS (English / Bangla).

Usage: python generate_voice.py --lang en|bn
Reads: output/script_<lang>.json            (made by generate_script.py)
Writes: output/voice_<lang>.wav             (whole narration, one file)
        output/timings_<lang>.json          (start/end second of every segment, for the video step)
        output/voice_<lang>/seg_XX.wav      (one file per segment, also used as a retry cache)
Env:   GEMINI_API_KEY        (required)
       GEMINI_TTS_MODEL      (optional, comma list; default = auto-discover TTS models)
       GEMINI_VOICE          (optional prebuilt voice name, default Charon)
       TTS_STYLE             (optional style prefix; default empty so nothing but the script is spoken)
       VOICE_CLEAN           (optional, 0 = keep raw audio; default 1 = trim tail hiss + fades)
       VOICE_TAIL_THRESHOLD  (optional, default 0.12; raise to 0.2 to cut more of a quiet hiss)
       VOICE_GATE_THRESHOLD  (optional, default 0 = off; whole-segment hiss gate)
       VOICE_TAIL_GATE       (optional, default 0.2; hiss gate for the last 0.8 s of each segment only; 0 = off)
       GEN_BUDGET_SECONDS    (optional total time budget, default 900)

Robustness: busy/limited/unavailable models are skipped, rounds repeat with waiting,
finished segments are cached so a retry only redoes what is missing, text is split so
no request exceeds the input-size limit (Bangla uses ~3 bytes per character).
"""
import argparse
import base64
import json
import math
import os
import re
import sys
import time
import wave
from array import array
from pathlib import Path

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models"
OUT = Path("output")
BUDGET = int(os.environ.get("GEN_BUDGET_SECONDS", "900"))
VOICE = os.environ.get("GEMINI_VOICE", "Charon").strip() or "Charon"
STYLE = os.environ.get("TTS_STYLE", "")  # empty on purpose: ONLY the script text is sent, nothing extra can be read aloud
FALLBACK_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash-lite-preview-tts", "gemini-2.5-pro-preview-tts"]
MAX_CHUNK_BYTES = 3000   # service limit is about 4000 bytes per text field
GAP_SECONDS = 0.35       # silence between segments
MIN_AUDIO_SECONDS = 0.3
CLEAN = os.environ.get("VOICE_CLEAN", "1") != "0"          # trim tail hiss + fades
TAIL_RATIO = float(os.environ.get("VOICE_TAIL_THRESHOLD", "0.12"))  # raise (e.g. 0.2) to cut more
GATE_RATIO = float(os.environ.get("VOICE_GATE_THRESHOLD", "0"))     # whole-segment gate; OFF by default (it can make speech sound rough)
TAIL_GATE_RATIO = float(os.environ.get("VOICE_TAIL_GATE", "0.2"))   # gate applied ONLY to the last 0.8 s of each segment
GATE_FLOOR = float(os.environ.get("VOICE_GATE_FLOOR", "0.08"))      # gain used in those quiet parts (0.08 = about -22 dB)


# ----------------------------------------------------------------- text
def split_chunks(text: str, max_bytes: int = MAX_CHUNK_BYTES) -> list[str]:
    """Split on sentence ends so each chunk stays under max_bytes (UTF-8)."""
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
    chunks, cur = [], ""
    for s in sentences:
        pieces = [s]
        if len(s.encode("utf-8")) > max_bytes:  # very long sentence: split on words
            pieces, buf = [], ""
            for w in s.split():
                if len((buf + " " + w).encode("utf-8")) > max_bytes and buf:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                pieces.append(buf)
        for p in pieces:
            if cur and len((cur + " " + p).encode("utf-8")) > max_bytes:
                chunks.append(cur)
                cur = p
            else:
                cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.lower())


def build_segments(script: dict) -> list[dict]:
    scenes = script["scenes"]
    segs = []
    hook = script["hook"].strip()
    first = scenes[0]["narration"]
    if norm(first).startswith(norm(hook)[:40]):  # scene 1 already starts with the hook
        pass
    else:
        segs.append({"kind": "hook", "text": hook, "image_query": scenes[0].get("image_query", "")})
    for i, s in enumerate(scenes, 1):
        segs.append({"kind": "scene", "text": s["narration"].strip(), "image_query": s.get("image_query", "")})
    return segs


# ----------------------------------------------------------------- gemini
def tts_rank(name: str):
    lite = "lite" in name
    pro = "pro" in name
    group = 2 if pro else (1 if lite else 0)
    ver = tuple(int(x) for x in re.findall(r"\d+", name))
    return (group, tuple(-v for v in ver))


def list_candidates(headers: dict) -> list[str]:
    forced = os.environ.get("GEMINI_TTS_MODEL", "").strip()
    if forced:
        return [m.strip() for m in forced.split(",") if m.strip()]
    try:
        r = requests.get(f"{API}?pageSize=200", headers=headers, timeout=30)
        r.raise_for_status()
        names = [
            m["name"].split("/")[-1]
            for m in r.json().get("models", [])
            if "tts" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    except Exception as e:  # noqa: BLE001 - discovery must never crash the run
        print(f"model discovery failed ({type(e).__name__}); using built-in list")
        return FALLBACK_MODELS
    names.sort(key=tts_rank)
    return names or FALLBACK_MODELS


def synth(text: str, headers: dict, candidates: list[str], deadline: float):
    """Returns (pcm_bytes, sample_rate) or None when the time budget ran out."""
    body = {
        "contents": [{"parts": [{"text": (STYLE + text) if STYLE.strip() else text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}},
        },
    }
    rnd = 0
    while True:
        rnd += 1
        daily = 0
        for model in candidates:
            remaining = deadline - time.time()
            if remaining <= 2:
                return None
            try:
                r = requests.post(
                    f"{API}/{model}:generateContent", headers=headers, json=body, timeout=min(120, remaining)
                )
            except requests.RequestException as e:
                print(f"[round {rnd}] {model}: {type(e).__name__}, next")
                continue
            if r.status_code != 200:
                if r.status_code in (400, 401, 403) and ("API_KEY_INVALID" in r.text or "API key not valid" in r.text):
                    print("The Gemini API key is invalid. Check the GEMINI_API_KEY secret.")
                    sys.exit(3)
                if r.status_code == 429 and re.search(r"per\s*day|PerDay", r.text, re.IGNORECASE):
                    daily += 1
                print(f"[round {rnd}] {model}: HTTP {r.status_code}, next")
                continue
            try:
                part = r.json()["candidates"][0]["content"]["parts"][0]
                inline = part.get("inlineData") or part.get("inline_data")
                pcm = base64.b64decode(inline["data"])
                mime = inline.get("mimeType") or inline.get("mime_type") or ""
            except (KeyError, IndexError, TypeError, ValueError):
                print(f"[round {rnd}] {model}: no audio in answer, next")
                continue
            m = re.search(r"rate=(\d+)", mime)
            rate = int(m.group(1)) if m else 24000
            if len(pcm) < rate * 2 * MIN_AUDIO_SECONDS:
                print(f"[round {rnd}] {model}: audio too short, next")
                continue
            print(f"voice ok with {model}")
            return pcm, rate
        if daily and daily == len(candidates):
            print("Daily Gemini TTS quota is used up on every model. Run again tomorrow or raise the limit.")
            sys.exit(3)
        wait = min(20 * rnd, 60, max(0.0, deadline - time.time() - 2))
        if wait <= 0:
            return None
        print(f"all models busy/limited; waiting {int(wait)}s (round {rnd} done)")
        time.sleep(wait)


# ----------------------------------------------------------------- cleanup
def gate_quiet(b: array, rate: int, thr: float) -> None:
    """In place: lower very quiet stretches (hiss between words/sentences) by GATE_FLOOR.
    Frames within 60 ms of real speech are left untouched so word edges are not clipped;
    the gain changes smoothly from frame to frame (no clicks)."""
    frame = max(1, int(rate * 0.01))
    nf = len(b) // frame
    if nf < 3:
        return
    loud = []
    for i in range(nf):
        chunk = b[i * frame:(i + 1) * frame]
        loud.append(math.sqrt(sum(x * x for x in chunk) / frame) > thr)
    hold = 6
    target = []
    for i in range(nf):
        near = any(loud[max(0, i - hold):i + hold + 1])
        target.append(1.0 if near else GATE_FLOOR)
    for i in range(nf):
        g0 = target[i]
        g1 = target[i + 1] if i + 1 < nf else target[i]
        if g0 == 1.0 and g1 == 1.0:
            continue
        base = i * frame
        for j in range(frame):
            b[base + j] = int(b[base + j] * (g0 + (g1 - g0) * j / frame))
    # samples after the last full frame keep the last gain
    last = target[-1]
    if last != 1.0:
        for k in range(nf * frame, len(b)):
            b[k] = int(b[k] * last)


def clean_pcm(pcm: bytes, rate: int) -> bytes:
    """Cut the quiet hiss/noise after the last spoken word (and any at the start),
    then apply short fades so there are no clicks. 16-bit mono PCM in, same out."""
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        a.byteswap()
    frame = max(1, int(rate * 0.02))
    n = len(a) // frame
    if n < 5:
        return pcm
    rms = []
    for i in range(n):
        chunk = a[i * frame:(i + 1) * frame]
        rms.append(math.sqrt(sum(x * x for x in chunk) / frame))
    p90 = sorted(rms)[int(0.9 * (n - 1))]
    thr = max(200.0, TAIL_RATIO * p90)
    loud = [i for i, v in enumerate(rms) if v > thr]
    if not loud:
        return pcm
    start = max(0, loud[0] * frame - int(rate * 0.05))
    end = min(len(a), (loud[-1] + 1) * frame + int(rate * 0.06))
    b = a[start:end]
    if GATE_RATIO > 0:
        gate_quiet(b, rate, max(120.0, GATE_RATIO * p90))
    if TAIL_GATE_RATIO > 0 and len(b) > rate:
        k = len(b) - int(rate * 0.8)
        tail = b[k:]
        gate_quiet(tail, rate, max(150.0, TAIL_GATE_RATIO * p90))
        b[k:] = tail
    fade_in, fade_out = int(rate * 0.01), int(rate * 0.10)
    ln = len(b)
    for i in range(min(fade_in, ln)):
        b[i] = int(b[i] * i / fade_in)
    for i in range(min(fade_out, ln)):
        b[ln - 1 - i] = int(b[ln - 1 - i] * i / fade_out)
    if sys.byteorder == "big":
        b.byteswap()
    return b.tobytes()


# ----------------------------------------------------------------- wav
def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "bn"], required=True)
    lang = ap.parse_args().lang

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    script_path = OUT / f"script_{lang}.json"
    if not script_path.exists():
        sys.exit(f"{script_path} not found. Run generate_script.py first.")
    script = json.loads(script_path.read_text(encoding="utf-8"))
    segments = build_segments(script)

    seg_dir = OUT / f"voice_{lang}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    candidates = list_candidates(headers)
    print("tts models to try:", ", ".join(candidates), "| voice:", VOICE)
    deadline = time.time() + BUDGET

    rate = None
    pcms: list[bytes] = []
    for i, seg in enumerate(segments):
        f = seg_dir / f"seg_{i:02d}.wav"
        if f.exists():
            pcm, r = read_wav(f)
            print(f"segment {i}: cached")
        else:
            parts = []
            for chunk in split_chunks(seg["text"]):
                res = synth(chunk, headers, candidates, deadline)
                if res is None:
                    sys.exit(f"Time budget used up at segment {i}. Finished segments are kept; run again.")
                parts.append(res)
            r = parts[0][1]
            if any(p[1] != r for p in parts):
                sys.exit("Inconsistent sample rates between chunks.")
            pcm = b"".join(p[0] for p in parts)
            write_wav(f, pcm, r)
            print(f"segment {i}: {len(pcm) / (2 * r):.1f}s")
        if rate is None:
            rate = r
        elif r != rate:
            sys.exit("Inconsistent sample rates between segments.")
        if CLEAN:
            cleaned = clean_pcm(pcm, rate)
            print(f"segment {i}: cleaned {len(pcm) / (2 * rate):.2f}s -> {len(cleaned) / (2 * rate):.2f}s")
            pcm = cleaned
        pcms.append(pcm)

    gap = b"\x00\x00" * int(rate * GAP_SECONDS)
    timeline, t, joined = [], 0.0, []
    for i, (seg, pcm) in enumerate(zip(segments, pcms)):
        dur = len(pcm) / (2 * rate)
        timeline.append({
            "index": i, "kind": seg["kind"], "text": seg["text"], "image_query": seg["image_query"],
            "start": round(t, 3), "end": round(t + dur, 3), "duration": round(dur, 3),
        })
        joined.append(pcm)
        if i < len(segments) - 1:
            joined.append(gap)
        t += dur + GAP_SECONDS

    write_wav(OUT / f"voice_{lang}.wav", b"".join(joined), rate)
    total = timeline[-1]["end"]
    (OUT / f"timings_{lang}.json").write_text(
        json.dumps({"lang": lang, "voice": VOICE, "sample_rate": rate, "total_seconds": total, "segments": timeline},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK [{lang}] {len(segments)} segments, {total:.1f}s total")


if __name__ == "__main__":
    main()
