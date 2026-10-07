"""Step 3: video (16:9 and 9:16) from free stock photos + the voiceover + captions.

Usage: python generate_video.py --lang en|bn --kind long|short
Reads:  output/timings_<lang>_<kind>.json, output/voice_<lang>_<kind>.wav
Writes: output/video_<lang>_<kind>_<fmt>.mp4, output/credits_<lang>_<kind>.txt
        long -> 16x9 (horizontal), short -> 9x16 (vertical) unless VIDEO_FORMATS says otherwise
Needs:  ffmpeg (with libass) + fonts-noto-core, and at least one free image key:
        PIXABAY_API_KEY (https://pixabay.com/api/docs/ - shown after you log in) and/or
        PEXELS_API_KEY  (https://www.pexels.com/api/new/ - issuing new keys was paused when this was written)

Env:   PIXABAY_API_KEY / PEXELS_API_KEY  (at least one; Pixabay is tried first)
       VIDEO_FORMATS       (optional; default 16x9 for long, 9x16 for short)
       VIDEO_PRESET / VIDEO_CRF  (optional x264 settings, default veryfast / 23)

Each scene gets one stock photo (search = the scene's image_query) with a slow zoom for exactly
the scene's duration from timings_<lang>.json, so pictures change when the narration moves on.
Captions are burned in. If a photo cannot be found or downloaded, a plain gradient is used so the
run still finishes. Photographers are credited in credits_<lang>.txt (paste it in the description).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

OUT = Path("output")
WORK = Path("work")
FPS = 25
SIZES = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
ORIENT = {"16x9": "landscape", "9x16": "portrait"}
PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "").strip()
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY", "").strip()
PRESET = os.environ.get("VIDEO_PRESET", "veryfast")
CRF = os.environ.get("VIDEO_CRF", "23")
HEADROOM = 1.25  # source images are fetched 25% larger than the frame so the zoom stays sharp


# ----------------------------------------------------------------- image providers
def pexels_search(query: str, orientation: str) -> list[dict]:
    headers = {"Authorization": PEXELS_KEY}
    params = {"query": query, "orientation": orientation, "per_page": 15, "size": "large"}
    for attempt in range(4):
        try:
            r = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"pexels search error ({type(e).__name__}), retry")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 401:
            print("The Pexels API key is invalid. Check the PEXELS_API_KEY secret.")
            sys.exit(3)
        if r.status_code == 429:
            print("Pexels rate limit reached; waiting")
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code != 200:
            time.sleep(5 * (attempt + 1))
            continue
        try:
            return r.json().get("photos", [])
        except ValueError:
            return []
    return []


def pixabay_search(query: str, fmt: str) -> list[dict]:
    """Pixabay free key: largest image offered is 1280 px (bigger sizes need Pixabay's approval)."""
    vertical = fmt == "9x16"
    params = {"key": PIXABAY_KEY, "q": query[:100], "image_type": "photo", "safesearch": "true",
              "orientation": "vertical" if vertical else "horizontal", "per_page": 30}
    for attempt in range(4):
        try:
            r = requests.get("https://pixabay.com/api/", params=params, timeout=30)
        except requests.RequestException as e:
            print(f"pixabay search error ({type(e).__name__}), retry")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            print("Pixabay rate limit reached; waiting 60s")
            time.sleep(60)
            continue
        if r.status_code in (400, 401, 403) and re.search(r"api key|invalid key|key", r.text, re.IGNORECASE):
            print("The Pixabay API key is invalid. Check the PIXABAY_API_KEY secret.")
            sys.exit(3)
        if r.status_code != 200:
            time.sleep(5 * (attempt + 1))
            continue
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            return []
        need = ("imageHeight", 1280) if vertical else ("imageWidth", 1280)
        hits.sort(key=lambda hh: 0 if hh.get(need[0], 0) >= need[1] else 1)  # sharp ones first (stable)
        return [{"id": f"pixabay-{hh['id']}", "urls": [u for u in (hh.get("largeImageURL"), hh.get("webformatURL")) if u],
                 "credit": f"Image by {hh.get('user', 'Pixabay user')} from Pixabay: {hh.get('pageURL', '')}"}
                for hh in hits if hh.get("id")]
    return []


def pexels_candidates(query: str, fmt: str) -> list[dict]:
    out = []
    for photo in pexels_search(query, ORIENT[fmt]):
        w, h = SIZES[fmt]
        src = photo.get("src", {})
        urls = []
        if src.get("original"):
            urls.append(f"{src['original']}?auto=compress&cs=tinysrgb&fit=crop&w={int(w * HEADROOM)}&h={int(h * HEADROOM)}")
        urls += [src[k] for k in ("large2x", "large") if src.get(k)]
        out.append({"id": f"pexels-{photo.get('id')}", "urls": urls,
                    "credit": f"Photo by {photo.get('photographer', '')} on Pexels: {photo.get('url', '')}"})
    return out


def search_candidates(query: str, fmt: str) -> list[dict]:
    """Candidates from every configured provider (Pixabay first, then Pexels)."""
    out = []
    if PIXABAY_KEY:
        out += pixabay_search(query, fmt)
    if PEXELS_KEY:
        out += pexels_candidates(query, fmt)
    return out


def download(url: str, path: Path) -> bool:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 5000:
                path.write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return False


def query_variants(q: str) -> list[str]:
    words = q.split()
    out = [q]
    for n in (3, 2, 1):
        v = " ".join(words[:n])
        if v and v not in out:
            out.append(v)
    return out


def gradient(path: Path, w: int, h: int, idx: int) -> None:
    colors = ["0x1b2a49", "0x2d1b49", "0x0f3d3e", "0x3d2b0f", "0x1b3a2a"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c={colors[idx % len(colors)]}:s={w}x{h}", "-frames:v", "1", "-update", "1", str(path)],
        check=True,
    )


def fetch_images(segments: list[dict], fmt: str, tag: str, credits: dict) -> list[Path]:
    w, h = SIZES[fmt]
    dw, dh = int(w * HEADROOM) // 2 * 2, int(h * HEADROOM) // 2 * 2
    folder = WORK / f"images_{tag}_{fmt}"
    folder.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()
    found: dict[str, list] = {}
    paths = []
    for i, seg in enumerate(segments):
        path = folder / f"scene_{i:02d}.jpg"
        meta = folder / f"scene_{i:02d}.json"
        if path.exists():
            if meta.exists():
                m = json.loads(meta.read_text(encoding="utf-8"))
                if "credit" in m:
                    credits[m["id"]] = m
                    used.add(m["id"])
            paths.append(path)
            continue
        done = False
        for allow_reuse in (False, True):  # prefer a photo not used yet; reuse one before falling back to a gradient
            for q in query_variants(segments[i].get("image_query") or "abstract background"):
                if q not in found:
                    found[q] = search_candidates(q, fmt)
                    time.sleep(0.8)
                for cand in found[q]:
                    if cand["id"] in used and not allow_reuse:
                        continue
                    if any(download(u, path) for u in cand["urls"]):
                        used.add(cand["id"])
                        info = {"id": cand["id"], "credit": cand["credit"]}
                        credits[cand["id"]] = info
                        meta.write_text(json.dumps(info), encoding="utf-8")
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if not done:
            print(f"scene {i} [{fmt}]: no usable photo for '{seg.get('image_query')}', using a gradient")
            gradient(path, dw, dh, i)
        print(f"scene {i} [{fmt}]: image ready")
        paths.append(path)
    return paths


# ----------------------------------------------------------------- captions
def ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def caption_chunks(text: str, max_words: int) -> list[str]:
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
    chunks = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            chunks.append(s)
            continue
        n = -(-len(words) // max_words)       # pieces
        size = -(-len(words) // n)            # even pieces
        chunks += [" ".join(words[k:k + size]) for k in range(0, len(words), size)]
    return chunks or [text]


def write_ass(path: Path, segments: list[dict], fmt: str, lang: str) -> None:
    w, h = SIZES[fmt]
    wide = fmt == "16x9"
    font = "Noto Sans Bengali" if lang == "bn" else "Noto Sans"
    size, outline = (60, 4) if wide else (68, 5)
    mv, mlr = (80, 140) if wide else (420, 70)
    max_words = 9 if wide else 6
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{outline},2,2,{mlr},{mlr},{mv},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for seg in segments:
        chunks = caption_chunks(seg["text"], max_words)
        weights = [max(1, len(re.sub(r"\s+", "", c))) for c in chunks]
        total_w = sum(weights)
        span = seg["end"] - seg["start"]
        t = seg["start"]
        for c, wt in zip(chunks, weights):
            d = span * wt / total_w
            text = c.replace("{", "(").replace("}", ")").replace("\n", " ")
            lines.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Default,,0,0,0,,{text}")
            t += d
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- ffmpeg
def render(fmt: str, tag: str, lang: str, segments: list[dict], images: list[Path], voice: Path, total: float) -> Path:
    w, h = SIZES[fmt]
    dw, dh = int(w * HEADROOM) // 2 * 2, int(h * HEADROOM) // 2 * 2
    ass = WORK / f"captions_{tag}_{fmt}.ass"
    write_ass(ass, segments, fmt, lang)

    bounds = [0] + [int(round(s["end"] * FPS)) for s in segments]
    bounds[-1] = int(round(total * FPS))
    frames = [max(1, bounds[i + 1] - bounds[i]) for i in range(len(segments))]

    inputs, chains = [], []
    for i, img in enumerate(images):
        inputs += ["-i", str(img.resolve())]
        n = frames[i]
        z = f"1+0.10*on/{n}" if i % 2 == 0 else f"1.10-0.10*on/{n}"
        chains.append(
            f"[{i}:v]scale={dw}:{dh}:flags=lanczos:force_original_aspect_ratio=increase,crop={dw}:{dh},setsar=1,"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={w}x{h}:fps={FPS},"
            f"format=yuv420p[v{i}]"
        )
    k = len(images)
    inputs += ["-i", str(voice.resolve())]
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[vc];"
    graph += f"[vc]subtitles=filename={ass.name}[vout];[{k}:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]"

    out = (OUT / f"video_{tag}_{fmt}.mp4").resolve()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", PRESET, "-crf", CRF, "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed:\n" + "\n".join(r.stderr.strip().splitlines()[-25:]))
        sys.exit(1)
    print(f"[{fmt}] rendered {out.name} in {time.time() - t0:.0f}s, {out.stat().st_size / 1e6:.1f} MB")
    return out


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "bn"], required=True)
    ap.add_argument("--kind", choices=["long", "short"], required=True)
    args = ap.parse_args()
    lang, kind = args.lang, args.kind
    tag = f"{lang}_{kind}"

    if not (PIXABAY_KEY or PEXELS_KEY):
        print("No image key found. Get a free Pixabay key (pixabay.com/api/docs, after logging in) and add it as the "
              "GitHub secret PIXABAY_API_KEY.")
        sys.exit(3)
    for need in (f"timings_{tag}.json", f"voice_{tag}.wav"):
        if not (OUT / need).exists():
            sys.exit(f"{OUT / need} not found. Run the earlier steps first.")
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        sys.exit("ffmpeg is not installed")

    timings = json.loads((OUT / f"timings_{tag}.json").read_text(encoding="utf-8"))
    segments = timings["segments"]
    total = float(timings["total_seconds"])
    WORK.mkdir(exist_ok=True)
    default_fmt = "16x9" if kind == "long" else "9x16"
    formats = [f.strip() for f in os.environ.get("VIDEO_FORMATS", default_fmt).split(",") if f.strip() in SIZES]

    credits: dict = {}
    for fmt in formats:
        images = fetch_images(segments, fmt, tag, credits)
        render(fmt, tag, lang, segments, images, OUT / f"voice_{tag}.wav", total)

    lines = []
    if any(c["id"].startswith("pixabay") for c in credits.values()):
        lines.append("Images from Pixabay (https://pixabay.com)")
    if any(c["id"].startswith("pexels") for c in credits.values()):
        lines.append("Photos provided by Pexels (https://www.pexels.com)")
    lines += [c["credit"] for c in credits.values()]
    (OUT / f"credits_{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK [{tag}] {len(formats)} video(s), {len(credits)} photos credited")


if __name__ == "__main__":
    main()
