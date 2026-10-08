"""Step 3: video (16:9 and 9:16) from free stock photos + the voiceover + captions.

Usage: python generate_video.py --lang en|bn --kind long|short
Reads:  output/timings_<lang>_<kind>.json, output/voice_<lang>_<kind>.wav
Writes: output/video_<lang>_<kind>_<fmt>.mp4, output/credits_<lang>_<kind>.txt
        long -> 16x9 (horizontal), short -> 9x16 (vertical) unless VIDEO_FORMATS says otherwise
Needs:  ffmpeg (with libass) + fonts-noto-core, and at least one free image key:
        PIXABAY_API_KEY (https://pixabay.com/api/docs/ - shown after you log in) and/or
        PEXELS_API_KEY  (https://www.pexels.com/api/new/ - issuing new keys was paused when this was written)

"""Step 3 (v2): video from stock FOOTAGE (photos as fallback) + voiceover + background music + captions.

Usage: python generate_video.py --lang en|bn --kind long|short
Reads:  output/timings_<lang>_<kind>.json, output/voice_<lang>_<kind>.wav, music/* (optional)
Writes: output/video_<lang>_<kind>_<fmt>.mp4, output/credits_<lang>_<kind>.txt
        long -> 16x9 (horizontal), short -> 9x16 (vertical) unless VIDEO_FORMATS says otherwise
Needs:  ffmpeg (with libass) + fonts-noto-core, and a free PIXABAY_API_KEY (also used for video clips).

How it works
  * Every scene is cut into shots of about 7 s. Each shot is a different Pixabay video clip found with the
    scene's image_query. If no clip is found (or one fails to render) a photo with a slow zoom is used instead,
    and if even that fails a plain gradient, so the run always finishes.
  * Vertical (9:16) videos show the clip in the middle over a blurred, enlarged copy of itself, nothing is cropped away.
  * Music: put royalty-free tracks (mp3, m4a, wav, ogg, opus, webm ...) in the repo folder music/ . One is picked per
    video, played quietly, lowered automatically while the narrator speaks, with fade-in and fade-out.
    Optional music/credits.txt is added to credits_<lang>_<kind>.txt. No tracks = no music.
  * File size is capped (long ~3.5 Mbps, short ~4 Mbps) so the GitHub artifact stays small.

Env:   PIXABAY_API_KEY / PEXELS_API_KEY (at least one for photos; clips need Pixabay)
       VIDEO_FORMATS (default 16x9 for long, 9x16 for short), VIDEO_CLIPS=0 (photos only), SHOT_SECONDS (default 7),
       VIDEO_WORKERS (parallel shot renders, default 2), VIDEO_MUSIC=0 (no music), MUSIC_VOLUME_LUFS (default -30)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
SHOT_SEC = float(os.environ.get("SHOT_SECONDS", "7"))
USE_CLIPS = os.environ.get("VIDEO_CLIPS", "1") != "0" and bool(PIXABAY_KEY)
WORKERS = max(1, int(os.environ.get("VIDEO_WORKERS", "2")))
USE_MUSIC = os.environ.get("VIDEO_MUSIC", "1") != "0"
MUSIC_LUFS = os.environ.get("MUSIC_VOLUME_LUFS", "-30")
MUSIC_DIR = Path("music")
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".webm", ".aac", ".flac"}
MAXRATE = {"16x9": "3500k", "9x16": "4000k"}


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



# ----------------------------------------------------------------- video clips (Pixabay)
def pixabay_video_search(query: str) -> list[dict]:
    params = {"key": PIXABAY_KEY, "q": query[:100], "safesearch": "true", "per_page": 20}
    for attempt in range(3):
        try:
            r = requests.get("https://pixabay.com/api/videos/", params=params, timeout=30)
        except requests.RequestException as e:
            print(f"pixabay video search error ({type(e).__name__}), retry")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            print("Pixabay rate limit reached; waiting 60s")
            time.sleep(60)
            continue
        if r.status_code in (400, 401, 403) and "key" in r.text.lower():
            print("The Pixabay API key is invalid. Check the PIXABAY_API_KEY secret.")
            sys.exit(3)
        if r.status_code != 200:
            time.sleep(5 * (attempt + 1))
            continue
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            return []
        out = []
        for h in hits:
            best = None
            for name in ("large", "medium", "small"):
                v = (h.get("videos") or {}).get(name) or {}
                try:
                    wd, size = int(v.get("width", 0)), int(v.get("size", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if v.get("url") and 640 <= wd <= 1920 and size <= 80_000_000 and (best is None or wd > best[1]):
                    best = (v["url"], wd)
            try:
                dur = float(h.get("duration", 0) or 0)
            except (TypeError, ValueError):
                dur = 0.0
            if best and dur >= 3 and h.get("id"):
                out.append({"id": f"pixabayvideo-{h['id']}", "url": best[0], "duration": dur,
                            "user": h.get("user", "Pixabay user"), "page": h.get("pageURL", "")})
        return out
    return []


def download_file(url: str, path: Path) -> bool:
    tmp = path.with_suffix(".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as r:
                if r.status_code == 200:
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    if tmp.stat().st_size > 50_000:
                        tmp.replace(path)
                        return True
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return False


# ----------------------------------------------------------------- shots
def photo_for_scene(i: int, seg: dict, fmt: str, tag: str, state: dict) -> Path:
    """One photo for scene i (cached; falls back to a gradient). Used when no clip is available."""
    w, h = SIZES[fmt]
    dw, dh = int(w * HEADROOM) // 2 * 2, int(h * HEADROOM) // 2 * 2
    folder = WORK / f"images_{tag}_{fmt}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"scene_{i:02d}.jpg"
    if path.exists():
        return path
    for allow_reuse in (False, True):
        for q in query_variants(seg.get("image_query") or "abstract background"):
            if q not in state["pfound"]:
                state["pfound"][q] = search_candidates(q, fmt)
                time.sleep(0.8)
            for cand in state["pfound"][q]:
                if cand["id"] in state["used_photos"] and not allow_reuse:
                    continue
                if any(download(u, path) for u in cand["urls"]):
                    state["used_photos"].add(cand["id"])
                    state["credits"][cand["id"]] = {"id": cand["id"], "credit": cand["credit"]}
                    return path
    print(f"scene {i} [{fmt}]: no usable photo for '{seg.get('image_query')}', using a gradient")
    gradient(path, dw, dh, i)
    return path


def plan_shots(segments: list[dict], fmt: str, tag: str, total: float, state: dict) -> list[dict]:
    """Cut every scene into shots; attach a clip to each shot where possible."""
    clip_dir = WORK / f"clips_{tag}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for i, seg in enumerate(segments):
        t0, t1 = float(seg["start"]), float(seg["end"])
        if i == len(segments) - 1:
            t1 = total
        n = max(1, round((t1 - t0) / SHOT_SEC))
        clips = []
        if USE_CLIPS:
            for allow_reuse in (False, True):
                for q in query_variants(seg.get("image_query") or "nature"):
                    if q not in state["vfound"]:
                        state["vfound"][q] = pixabay_video_search(q)
                        time.sleep(0.8)
                    for c in state["vfound"][q]:
                        if len(clips) >= n:
                            break
                        if (c["id"] in state["used_clips"] and not allow_reuse) or any(c["id"] == x[0]["id"] for x in clips):
                            continue
                        path = clip_dir / f"{c['id']}.mp4"
                        if path.exists() or download_file(c["url"], path):
                            clips.append((c, path))
                            state["used_clips"].add(c["id"])
                            state["credits"][c["id"]] = {"id": c["id"], "user": c["user"], "credit": f"Video by {c['user']} from Pixabay: {c['page']}"}
                    if len(clips) >= n:
                        break
                if clips:
                    break
        k = max(1, len(clips))
        cuts = [t0 + (t1 - t0) * j / k for j in range(k + 1)]
        for j in range(k):
            f0, f1 = int(round(cuts[j] * FPS)), int(round(cuts[j + 1] * FPS))
            shots.append({"scene": i, "frames": max(1, f1 - f0), "clip": clips[j][1] if clips else None})
        print(f"scene {i} [{fmt}]: {len(clips)} clip(s) for {k} shot(s), {t1 - t0:.1f}s")
    return shots


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "50", "-an", "-threads", "2"]


def run_ffmpeg(cmd: list[str], cwd=None) -> bool:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed: " + " | ".join(r.stderr.strip().splitlines()[-4:]))
    return r.returncode == 0


def render_clip_shot(src: Path, frames: int, fmt: str, out: Path) -> bool:
    w, h = SIZES[fmt]
    if fmt == "16x9":
        graph = f"[0:v]fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,format=yuv420p[v]"
    else:  # blurred enlarged background + the whole clip in the middle
        sw, sh = w // 8, h // 8  # blur a tiny copy and enlarge it: looks the same, costs far less CPU
        graph = (f"[0:v]fps={FPS},split[a][b];[a]scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},boxblur=4:2,"
                 f"scale={w}:{h}:flags=bilinear[bg];"
                 f"[b]scale={w}:-2:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(src.resolve()), "-filter_complex", graph,
           "-map", "[v]", "-frames:v", str(frames), *ENC, str(out.resolve())]
    return run_ffmpeg(cmd)


def render_photo_shot(photo: Path, frames: int, fmt: str, idx: int, out: Path) -> bool:
    w, h = SIZES[fmt]
    dw, dh = int(w * HEADROOM) // 2 * 2, int(h * HEADROOM) // 2 * 2
    z = f"1+0.10*on/{frames}" if idx % 2 == 0 else f"1.10-0.10*on/{frames}"
    graph = (f"[0:v]scale={dw}:{dh}:flags=lanczos:force_original_aspect_ratio=increase,crop={dw}:{dh},setsar=1,"
             f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={FPS},format=yuv420p[v]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(photo.resolve()), "-filter_complex", graph,
           "-map", "[v]", "-frames:v", str(frames), *ENC, str(out.resolve())]
    return run_ffmpeg(cmd)


# ----------------------------------------------------------------- music
def audio_ok(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0 and "audio" in r.stdout


def pick_music(tag: str) -> Path | None:
    if not USE_MUSIC or not MUSIC_DIR.is_dir():
        return None
    files = sorted(p for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        return None
    import datetime as dt
    start = (dt.date.today().toordinal() + (0 if tag.endswith("long") else 1) + (0 if tag.startswith("en") else 2)) % len(files)
    for k in range(len(files)):
        p = files[(start + k) % len(files)]
        if audio_ok(p):
            return p
        print(f"music file {p.name}: no audio stream, skipped")
    return None


def audio_graph(total: float, with_music: bool) -> str:
    voice = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
    if not with_music:
        return f"[1:a]{voice}[aout]"
    fade_out = max(0.0, total - 3)
    return (f"[1:a]{voice},asplit=2[va][vb];"
            f"[2:a]aresample=48000,atrim=duration={total:.2f},asetpts=PTS-STARTPTS,loudnorm=I={MUSIC_LUFS}:TP=-2:LRA=7,"
            f"afade=t=in:d=2,afade=t=out:st={fade_out:.2f}:d=3[mus];"
            f"[mus][vb]sidechaincompress=threshold=0.04:ratio=8:attack=25:release=500[duck];"
            f"[va][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")


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


# ----------------------------------------------------------------- render
def render(fmt: str, tag: str, lang: str, segments: list[dict], voice: Path, total: float, state: dict) -> Path:
    shots = plan_shots(segments, fmt, tag, total, state)
    shot_dir = WORK / f"shots_{tag}_{fmt}"
    if shot_dir.exists():
        for f in shot_dir.glob("*.mp4"):
            f.unlink()
    shot_dir.mkdir(parents=True, exist_ok=True)
    outs = [shot_dir / f"shot_{n:03d}.mp4" for n in range(len(shots))]

    t0 = time.time()
    def do(n: int) -> bool:
        s = shots[n]
        return bool(s["clip"]) and render_clip_shot(s["clip"], s["frames"], fmt, outs[n])
    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(do, range(len(shots))))
    for n, s in enumerate(shots):   # photo (or gradient) for scenes without a clip and for failed clip shots
        if not ok[n]:
            photo = photo_for_scene(s["scene"], segments[s["scene"]], fmt, tag, state)
            if not render_photo_shot(photo, s["frames"], fmt, n, outs[n]):
                sys.exit(f"could not render shot {n}")
    print(f"[{fmt}] {len(shots)} shots rendered in {time.time() - t0:.0f}s ({sum(1 for x in ok if x)} from clips)")

    ass = WORK / f"captions_{tag}_{fmt}.ass"
    write_ass(ass, segments, fmt, lang)
    listing = WORK / f"concat_{tag}_{fmt}.txt"
    listing.write_text("".join(f"file '{o.relative_to(WORK).as_posix()}'\n" for o in outs), encoding="utf-8")
    music = pick_music(tag)
    out = (OUT / f"video_{tag}_{fmt}.mp4").resolve()

    def final(with_music: bool) -> bool:
        inputs = ["-f", "concat", "-safe", "0", "-i", listing.name, "-i", str(voice.resolve())]
        if with_music:
            inputs += ["-stream_loop", "-1", "-i", str(music.resolve())]
        graph = f"[0:v]subtitles=filename={ass.name}[vout];" + audio_graph(total, with_music)
        rate = MAXRATE[fmt]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", CRF, "-maxrate", rate, "-bufsize", f"{int(rate[:-1]) * 2}k",
               "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
               "-t", f"{total:.3f}", str(out)]
        return run_ffmpeg(cmd, cwd=WORK)

    t1 = time.time()
    if music:
        print(f"[{fmt}] background music: {music.name}")
        if not final(True):
            print("final render with music failed; retrying without music")
            music = None
            if not final(False):
                sys.exit(1)
    elif not final(False):
        sys.exit(1)
    print(f"[{fmt}] final video {out.name}: {time.time() - t1:.0f}s, {out.stat().st_size / 1e6:.1f} MB"
          + (f", music {music.name}" if music else ", no music"))
    for f in outs:      # free disk space
        f.unlink(missing_ok=True)
    return out


# ----------------------------------------------------------------- credits + main
def credits_text(credits: dict) -> str:
    lines = []
    users = sorted({c["user"] for c in credits.values() if c["id"].startswith("pixabay") and c.get("user")})
    photo_users = []
    for c in credits.values():
        if c["id"].startswith("pixabay-"):
            m = re.match(r"Image by (.*) from Pixabay", c.get("credit", ""))
            if m:
                photo_users.append(m.group(1))
    users = sorted(set(users) | set(photo_users))
    if users:
        lines.append("Images and video clips from Pixabay (https://pixabay.com). Contributors: " + ", ".join(users[:60]))
    pex = sorted({m.group(1) for c in credits.values() if c["id"].startswith("pexels")
                  for m in [re.match(r"Photo by (.*) on Pexels", c.get("credit", ""))] if m})
    if pex:
        lines.append("Photos provided by Pexels (https://www.pexels.com). Photographers: " + ", ".join(pex[:40]))
    return "\n".join(lines)


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
    state = {"vfound": {}, "pfound": {}, "used_clips": set(), "used_photos": set(), "credits": credits}
    for fmt in formats:
        render(fmt, tag, lang, segments, OUT / f"voice_{tag}.wav", total, state)

    text = credits_text(credits)
    mc = MUSIC_DIR / "credits.txt"
    if USE_MUSIC and mc.exists():
        text = (text + "\n" + mc.read_text(encoding="utf-8").strip()).strip()
    (OUT / f"credits_{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"OK [{tag}] {len(formats)} video(s), {len(credits)} stock items credited")


if __name__ == "__main__":
    main()
