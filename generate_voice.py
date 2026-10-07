"""Step 2: voiceover with Gemini TTS (English / Bangla).

Usage: python generate_voice.py --lang en|bn
Reads: output/script_<lang>.json            (made by generate_script.py)
Writes: output/voice_<lang>.wav             (whole narration, one file)
        output/timings_<lang>.json          (start/end second of every segment, for the video step)
        output/voice_<lang>/seg_XX.wav      (one file per segment, also used as a retry cache)
Env:   GEMINI_API_KEY        (required)
       GEMINI_TTS_MODEL      (optional, comma list; default = auto-discover TTS models)
"""Step 2 (v3): voiceover with Gemini TTS (English / Bangla).

WHY v3: Gemini TTS sometimes leaves a hiss after the last word of EVERY audio it generates.
Older versions made one request per scene (9 scenes = up to 9 hisses). Now the scenes are
packed into as few requests as the input limit allows (usually ONE for English), so there is
at most one tail to clean (usually at the very end). One continuous reading also keeps the
voice consistent. Scene boundaries are found from the pauses in the audio, so the video
step still gets start/end seconds for every scene.

Usage: python generate_voice.py --lang en|bn
Reads: output/script_<lang>.json
Writes: output/voice_<lang>.wav, output/timings_<lang>.json, output/voice_<lang>/req_XX.wav (raw audio, also a retry cache)

Env:   GEMINI_API_KEY       (required)
       GEMINI_TTS_MODEL     (optional, comma list; default = auto-discover TTS models)
       GEMINI_VOICE         (optional prebuilt voice name, default Charon)
       TTS_STYLE            (optional style prefix; default empty so nothing but the script is spoken)
       VOICE_MODE           (optional; "scene" = one request per scene like the old version; default packed)
       VOICE_MAX_BYTES      (optional, default 3400 text bytes per request; service limit is ~4000)
       VOICE_CLEAN          (0 = keep raw audio; default 1 = trim tail hiss + fades)
       VOICE_FADE_OUT / VOICE_TAIL_THRESHOLD / VOICE_TAIL_GATE / VOICE_GATE_THRESHOLD  (fine tuning)
       GEN_BUDGET_SECONDS   (optional total time budget, default 900)
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
STYLE = os.environ.get("TTS_STYLE", "")  # empty on purpose: ONLY the script text is sent
FALLBACK_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash-lite-preview-tts", "gemini-2.5-pro-preview-tts"]
MAX_REQ_BYTES = int(os.environ.get("VOICE_MAX_BYTES", "3400"))
PACK = os.environ.get("VOICE_MODE", "packed").strip().lower() != "scene"
GAP_SECONDS = 0.35       # silence between requests (only when the script needs more than one)
MIN_AUDIO_SECONDS = 0.3
ATTEMPTS = 3             # regenerate up to 3 times if the audio looks wrong or has a noisy tail
NOISY_DB = -35.0         # tail louder than this (dB below speech) counts as noisy

CLEAN = os.environ.get("VOICE_CLEAN", "1") != "0"
TAIL_RATIO = float(os.environ.get("VOICE_TAIL_THRESHOLD", "0.2"))
FADE_OUT_SEC = float(os.environ.get("VOICE_FADE_OUT", "0.18"))
GATE_RATIO = float(os.environ.get("VOICE_GATE_THRESHOLD", "0"))
TAIL_GATE_RATIO = float(os.environ.get("VOICE_TAIL_GATE", "0.3"))
GATE_FLOOR = float(os.environ.get("VOICE_GATE_FLOOR", "0.08"))


# ----------------------------------------------------------------- text
def split_chunks(text: str, max_bytes: int) -> list[str]:
    """Split on sentence ends so each chunk stays under max_bytes (UTF-8)."""
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
    chunks, cur = [], ""
    for s in sentences:
        pieces = [s]
        if len(s.encode("utf-8")) > max_bytes:
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


def sentences_of(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.lower())


def build_segments(script: dict) -> list[dict]:
    scenes = script["scenes"]
    segs = []
    hook = script["hook"].strip()
    if not norm(scenes[0]["narration"]).startswith(norm(hook)[:40]):  # scene 1 does not already start with the hook
        segs.append({"kind": "hook", "text": hook, "image_query": scenes[0].get("image_query", "")})
    for s in scenes:
        segs.append({"kind": "scene", "text": s["narration"].strip(), "image_query": s.get("image_query", "")})
    return segs


def plan_requests(segments: list[dict]) -> list[list[dict]]:
    """Pack consecutive scenes into requests of at most MAX_REQ_BYTES of text."""
    units = []
    for si, seg in enumerate(segments):
        t = seg["text"]
        pieces = [t] if len(t.encode("utf-8")) <= MAX_REQ_BYTES else split_chunks(t, MAX_REQ_BYTES)
        for p in pieces:
            units.append({"scene": si, "text": p})
    reqs, cur, cur_bytes = [], [], 0
    for u in units:
        b = len(u["text"].encode("utf-8")) + 2
        if cur and (not PACK or cur_bytes + b > MAX_REQ_BYTES):
            reqs.append(cur)
            cur, cur_bytes = [], 0
        cur.append(u)
        cur_bytes += b
    if cur:
        reqs.append(cur)
    return reqs


# ----------------------------------------------------------------- gemini
def tts_rank(name: str):
    group = 2 if "pro" in name else (1 if "lite" in name else 0)
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
    """One TTS request. Returns (pcm_bytes, sample_rate) or None when the time budget ran out."""
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
                    f"{API}/{model}:generateContent", headers=headers, json=body, timeout=min(180, remaining)
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


# ----------------------------------------------------------------- audio helpers
def to_array(pcm: bytes) -> array:
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        a.byteswap()
    return a


def frame_rms(a: array, frame: int) -> list[float]:
    n = len(a) // frame
    return [math.sqrt(sum(x * x for x in a[i * frame:(i + 1) * frame]) / frame) for i in range(n)]


def p90_of(values: list[float]) -> float:
    return sorted(values)[int(0.9 * (len(values) - 1))] if values else 1.0


def noise_after_speech(pcm: bytes, rate: int) -> float:
    """Level (dB below speech) of whatever follows the last spoken word. -99 = clean ending."""
    a = to_array(pcm)
    frame = max(1, int(rate * 0.01))
    rms = frame_rms(a, frame)
    if len(rms) < 20:
        return -99.0
    p90 = max(p90_of(rms), 1.0)
    loud = [i for i, v in enumerate(rms) if v > 0.2 * p90]
    if not loud:
        return -99.0
    region = rms[loud[-1] + 12:]
    if len(region) < 10:
        return -99.0
    v = math.sqrt(sum(x * x for x in region) / len(region))
    return round(20 * math.log10(max(v, 1.0) / p90), 1)


def verify(pcm: bytes, rate: int, text: str) -> bool:
    """Is the audio length plausible for this much text? (catches skipped or cut-off speech)"""
    words = max(1, len(text.split()))
    secs = len(pcm) / (2 * rate)
    return words / 5.0 <= secs <= words / 1.0


def gate_quiet(b: array, rate: int, thr: float) -> None:
    """In place: lower very quiet stretches by GATE_FLOOR (smooth, 60 ms hold around speech)."""
    frame = max(1, int(rate * 0.01))
    nf = len(b) // frame
    if nf < 3:
        return
    loud = [math.sqrt(sum(x * x for x in b[i * frame:(i + 1) * frame]) / frame) > thr for i in range(nf)]
    hold = 6
    target = [1.0 if any(loud[max(0, i - hold):i + hold + 1]) else GATE_FLOOR for i in range(nf)]
    for i in range(nf):
        g0 = target[i]
        g1 = target[i + 1] if i + 1 < nf else target[i]
        if g0 == 1.0 and g1 == 1.0:
            continue
        base = i * frame
        for j in range(frame):
            b[base + j] = int(b[base + j] * (g0 + (g1 - g0) * j / frame))
    if target[-1] != 1.0:
        for k in range(nf * frame, len(b)):
            b[k] = int(b[k] * target[-1])


def clean_pcm(pcm: bytes, rate: int) -> bytes:
    """Cut hiss after the last spoken word (and any at the start), gate the last 0.8 s, add fades."""
    a = to_array(pcm)
    frame = max(1, int(rate * 0.02))
    rms = frame_rms(a, frame)
    n = len(rms)
    if n < 5:
        return pcm
    p90 = p90_of(rms)
    thr = max(200.0, TAIL_RATIO * p90)
    loud = [i for i, v in enumerate(rms) if v > thr]
    if not loud:
        return pcm
    start = max(0, loud[0] * frame - int(rate * 0.05))
    end = min(len(a), (loud[-1] + 1) * frame + int(rate * 0.02))
    b = a[start:end]
    if GATE_RATIO > 0:
        gate_quiet(b, rate, max(120.0, GATE_RATIO * p90))
    if TAIL_GATE_RATIO > 0 and len(b) > rate:
        k = len(b) - int(rate * 0.8)
        tail = b[k:]
        gate_quiet(tail, rate, max(150.0, TAIL_GATE_RATIO * p90))
        b[k:] = tail
    fade_in, fade_out = int(rate * 0.01), int(rate * FADE_OUT_SEC)
    ln = len(b)
    for i in range(min(fade_in, ln)):
        b[i] = int(b[i] * i / fade_in)
    for i in range(min(fade_out, ln)):
        b[ln - 1 - i] = int(b[ln - 1 - i] * 0.5 * (1 - math.cos(math.pi * i / fade_out)))
    if sys.byteorder == "big":
        b.byteswap()
    return b.tobytes()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


# ----------------------------------------------------------------- aligning scenes inside one audio
def find_pauses(a: array, rate: int) -> list[tuple[int, int]]:
    """Interior quiet stretches (>= 150 ms) as (start_sample, end_sample)."""
    frame = max(1, int(rate * 0.01))
    rms = frame_rms(a, frame)
    n = len(rms)
    if n < 10:
        return []
    thr = max(60.0, 0.08 * p90_of(rms))
    pauses, i = [], 0
    while i < n:
        if rms[i] < thr:
            j = i
            while j < n and rms[j] < thr:
                j += 1
            if i > 0 and j < n and (j - i) * 0.01 >= 0.15:
                pauses.append((i * frame, j * frame))
            i = j
        else:
            i += 1
    return pauses


def dp_sentence_pauses(pauses: list[tuple[int, int]], sent_chars: list[int], total: int, rate: int):
    """Choose which pause is the end of each sentence (S-1 of them) so that sentence lengths
    best match their text lengths. Returns a list of pause indexes, or None."""
    S, M = len(sent_chars), len(pauses)
    need = S - 1
    if need <= 0 or M < need:
        return None
    tc = sum(sent_chars)
    exp = [total * c / tc for c in sent_chars]
    cent = [(p[0] + p[1]) // 2 for p in pauses]
    plen = [(p[1] - p[0]) / rate for p in pauses]
    bonus = [0.25 * min(l, 0.8) for l in plen]

    def seg_cost(d: float, e: float) -> float:
        return ((d - e) / (0.6 * e + rate * 0.5)) ** 2

    INF = float("inf")
    dp = [[INF] * M for _ in range(need)]
    back = [[-1] * M for _ in range(need)]
    for p in range(M):
        dp[0][p] = seg_cost(cent[p], exp[0]) - bonus[p]
    for j in range(1, need):
        for p in range(j, M):
            best, arg = INF, -1
            for q in range(j - 1, p):
                if dp[j - 1][q] == INF:
                    continue
                v = dp[j - 1][q] + seg_cost(cent[p] - cent[q], exp[j]) - bonus[p]
                if v < best:
                    best, arg = v, q
            dp[j][p], back[j][p] = best, arg
    best, last = INF, -1
    for p in range(need - 1, M):
        if dp[need - 1][p] == INF:
            continue
        v = dp[need - 1][p] + seg_cost(total - cent[p], exp[need])
        if v < best:
            best, last = v, p
    if last < 0:
        return None
    chosen = [last]
    for j in range(need - 1, 0, -1):
        chosen.append(back[j][chosen[-1]])
    return chosen[::-1]


def align_units(pcm: bytes, rate: int, units: list[dict]) -> list[tuple[int, str]]:
    """Boundaries (sample index, method) between the units of one request."""
    if len(units) < 2:
        return []
    a = to_array(pcm)
    total = len(a)
    unit_sents = [sentences_of(u["text"]) or [u["text"]] for u in units]
    sent_chars = [max(1, len(re.sub(r"\s+", "", s))) for ss in unit_sents for s in ss]
    cum = []  # sentence index (1-based) at which each unit ends
    run = 0
    for ss in unit_sents[:-1]:
        run += len(ss)
        cum.append(run)
    chars = [max(1, len(re.sub(r"\s+", "", u["text"]))) for u in units]
    est = [int(total * sum(chars[:k + 1]) / sum(chars)) for k in range(len(units) - 1)]
    pauses = find_pauses(a, rate)
    centers = [(p[0] + p[1]) // 2 for p in pauses]
    tol = max(int(rate * 3.0), int(0.35 * total / max(1, len(units))))
    chosen = dp_sentence_pauses(pauses, sent_chars, total, rate)

    bounds = []
    for k in range(len(units) - 1):
        pick, method = None, "estimate"
        if chosen is not None:
            c = centers[chosen[cum[k] - 1]]
            if abs(c - est[k]) <= tol:
                pick, method = c, "pause"
        if pick is None and centers:
            c = min(centers, key=lambda x: abs(x - est[k]))
            if abs(c - est[k]) <= tol:
                pick, method = c, "nearest-pause"
        if pick is None:
            pick = est[k]
        lower = (bounds[-1][0] if bounds else 0) + int(rate * 0.3)
        bounds.append((max(pick, lower), method))
    return bounds


# ----------------------------------------------------------------- one request, best of a few tries
def synth_best(text: str, headers: dict, candidates: list[str], deadline: float):
    best = None
    for attempt in range(1, ATTEMPTS + 1):
        res = synth(text, headers, candidates, deadline)
        if res is None:
            break
        pcm, rate = res
        ok = verify(pcm, rate, text)
        noise = noise_after_speech(pcm, rate)
        print(f"attempt {attempt}: {len(pcm) / (2 * rate):.1f}s, plausible={ok}, tail noise={noise} dB")
        key = (0 if ok else 1, noise)
        if best is None or key < best[0]:
            best = (key, pcm, rate, ok, noise, attempt)
        if ok and noise <= NOISY_DB:
            break
    if best is None:
        return None
    _, pcm, rate, ok, noise, tries = best
    return pcm, rate, ok, noise, tries


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
    reqs = plan_requests(segments)
    print(f"{len(segments)} scenes -> {len(reqs)} TTS request(s)")

    seg_dir = OUT / f"voice_{lang}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    candidates = list_candidates(headers)
    print("tts models to try:", ", ".join(candidates), "| voice:", VOICE)
    deadline = time.time() + BUDGET

    rate = None
    parts = []  # (cleaned_pcm, units, bounds, debug)
    for ri, units in enumerate(reqs):
        text = "\n\n".join(u["text"] for u in units)
        f = seg_dir / f"req_{ri:02d}.wav"
        if f.exists():
            raw, r = read_wav(f)
            dbg = {"cached": True}
            print(f"request {ri}: cached")
        else:
            res = synth_best(text, headers, candidates, deadline)
            if res is None:
                sys.exit(f"Time budget used up at request {ri}. Finished requests are kept; run again.")
            raw, r, ok, noise, tries = res
            write_wav(f, raw, r)
            dbg = {"attempts": tries, "plausible": ok, "tail_noise_db_raw": noise}
        if rate is None:
            rate = r
        elif r != rate:
            sys.exit("Inconsistent sample rates between requests.")
        pcm = clean_pcm(raw, r) if CLEAN else raw
        dbg["tail_noise_db_clean"] = noise_after_speech(pcm, r)
        bounds = align_units(pcm, r, units)
        dbg["boundaries"] = [m for _, m in bounds]
        print(f"request {ri}: {len(raw) / (2 * r):.1f}s raw -> {len(pcm) / (2 * r):.1f}s clean, boundaries: {dbg['boundaries']}")
        parts.append((pcm, units, bounds, dbg))

    gap = b"\x00\x00" * int(rate * GAP_SECONDS)
    joined, t_off, scene_end = [], 0, {}
    for ri, (pcm, units, bounds, _) in enumerate(parts):
        n = len(pcm) // 2
        edges = [0] + [b for b, _ in bounds] + [n]
        for ui, u in enumerate(units):
            scene_end[u["scene"]] = t_off + edges[ui + 1]
        joined.append(pcm)
        t_off += n
        if ri < len(parts) - 1:
            joined.append(gap)
            t_off += len(gap) // 2

    timeline, prev = [], 0.0
    for si, seg in enumerate(segments):
        end = max(scene_end[si] / rate, prev + 0.05)
        timeline.append({
            "index": si, "kind": seg["kind"], "text": seg["text"], "image_query": seg["image_query"],
            "start": round(prev, 3), "end": round(end, 3), "duration": round(end - prev, 3),
        })
        prev = end
    total = t_off / rate
    timeline[-1]["end"] = round(total, 3)
    timeline[-1]["duration"] = round(total - timeline[-1]["start"], 3)

    write_wav(OUT / f"voice_{lang}.wav", b"".join(joined), rate)
    (OUT / f"timings_{lang}.json").write_text(
        json.dumps({"lang": lang, "voice": VOICE, "sample_rate": rate, "total_seconds": round(total, 3),
                    "requests": [p[3] for p in parts], "segments": timeline},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK [{lang}] {len(segments)} scenes, {len(parts)} request(s), {total:.1f}s total")


if __name__ == "__main__":
    main()
