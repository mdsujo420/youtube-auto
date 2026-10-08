"""Step 4: thumbnail + upload kit (title, description, chapters, tags) for one video.

Usage: python generate_package.py --lang en|bn --kind long|short
Reads:  output/script_<lang>_<kind>.json, output/timings_<lang>_<kind>.json, output/credits_<lang>_<kind>.txt,
        work/images_<lang>_<kind>_<fmt>/scene_XX.jpg   (the stock photos used by the video step)
Writes: output/thumbnail_<lang>_<kind>.jpg      1280x720 (long) or 1080x1920 cover (short)
        output/upload_kit_<lang>_<kind>.txt     copy-paste text for YouTube Studio (title, description, tags, checklist)
        output/upload_kit_<lang>_<kind>.json    same data for a future API upload

Needs ffmpeg + fonts-noto-core. Env: THUMB_SCENE (which scene's photo to use, default 0)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("output")
WORK = Path("work")
SIZES = {"long": (1280, 720), "short": (1080, 1920)}
PHOTO_DIR = {"long": "16x9", "short": "9x16"}


# ----------------------------------------------------------------- text kit
def fmt_ts(sec: float) -> str:
    s = int(sec)
    h, m, s = s // 3600, s // 60 % 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_chapters(script: dict, timings: dict) -> list[tuple[float, str]]:
    scenes, segs = script["scenes"], timings["segments"]
    heads = [s.get("heading") for s in script.get("outline", [])]
    offset = len(segs) - len(scenes)  # the voice step adds one extra segment if the hook was not in scene 1
    first_scene_of: dict[int, int] = {}
    for i, sc in enumerate(scenes):
        first_scene_of.setdefault(sc.get("section", 0), i)
    chapters = []
    for sec_idx, head in enumerate(heads):
        i = first_scene_of.get(sec_idx)
        if i is None or i + offset >= len(segs) or not head:
            continue
        chapters.append((float(segs[i + offset]["start"]), str(head)))
    if not chapters:
        return []
    chapters[0] = (0.0, chapters[0][1])
    kept = [chapters[0]]
    for start, head in chapters[1:]:
        if start - kept[-1][0] >= 10:      # YouTube needs chapters of at least 10 seconds
            kept.append((start, head))
    return kept if len(kept) >= 3 else []


def hashtags(script: dict, kind: str) -> list[str]:
    tags = []
    for t in script.get("tags", []):
        h = re.sub(r"[^\w]+", "", t, flags=re.UNICODE)
        if h and f"#{h}" not in tags:
            tags.append(f"#{h}")
        if len(tags) == 2:
            break
    return (["#Shorts"] if kind == "short" else []) + tags


def build_kit(script: dict, timings: dict, credits: str, lang: str, kind: str) -> dict:
    title = script["title"].strip()[:100]
    parts = [script["description"].strip()]
    chapters = build_chapters(script, timings) if kind == "long" else []
    if chapters:
        parts.append("\n".join(f"{fmt_ts(t)} {h}" for t, h in chapters))
    if script.get("sources"):
        parts.append("Sources to check: " + "; ".join(script["sources"]))
    if credits.strip():
        parts.append(credits.strip())
    tags = hashtags(script, kind)
    if tags:
        parts.append(" ".join(tags))
    keywords, total = [], 0
    for t in script.get("tags", []):
        if total + len(t) + 1 > 450:
            break
        keywords.append(t)
        total += len(t) + 1
    return {"title": title, "description": "\n\n".join(parts), "tags": keywords, "language": lang, "kind": kind,
            "categoryId": "27", "madeForKids": False, "chapters": [(fmt_ts(t), h) for t, h in chapters],
            "duration_seconds": timings.get("total_seconds"), "warnings": script.get("warnings", []) + timings.get("warnings", [])}


def kit_text(kit: dict, tag: str) -> str:
    checklist = [
        "Watch the whole video once (voice, pictures, captions, facts against the sources).",
        "Check the length: short must be under 3 minutes; long should be over 8 minutes for mid-roll ads.",
        "Set 'Made for kids' = No.",
        "Answer YouTube's 'altered or synthetic content' question honestly (AI voice, stock photos).",
        "Upload the thumbnail (long videos); for a short, pick a cover frame or upload the cover image.",
        "Paste title, description and tags exactly as below, then publish or schedule.",
    ]
    lines = [f"UPLOAD KIT - {tag}", "", "TITLE", kit["title"], "", "DESCRIPTION", kit["description"], "",
             "TAGS (comma separated)", ", ".join(kit["tags"]), "", "CHECKLIST"]
    lines += [f"[ ] {c}" for c in checklist]
    if kit["warnings"]:
        lines += ["", "WARNINGS FROM THE PIPELINE"] + [f"- {w}" for w in kit["warnings"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- thumbnail
def find_font(lang: str) -> str:
    family = "Noto Sans Bengali" if lang == "bn" else "Noto Sans"
    for q in (f"{family}:bold", "Noto Sans:bold", "DejaVu Sans:bold"):
        try:
            f = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True).stdout.strip()
        except OSError:
            f = ""
        if f and Path(f).exists():
            return f
    return ""


def wrap_two_lines(text: str) -> list[str]:
    words = text.split()
    if len(words) <= 1 or len(text) <= 11:
        return [text]
    best, best_max = None, 10 ** 9
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if max(len(a), len(b)) < best_max:
            best, best_max = [a, b], max(len(a), len(b))
    return best


def make_thumbnail(script: dict, lang: str, kind: str, tag: str) -> Path:
    w, h = SIZES[kind]
    idx = int(os.environ.get("THUMB_SCENE", "0"))
    photo = WORK / f"images_{tag}_{PHOTO_DIR[kind]}" / f"scene_{idx:02d}.jpg"
    if not photo.exists():
        cands = sorted((WORK / f"images_{tag}_{PHOTO_DIR[kind]}").glob("scene_*.jpg"))
        if not cands:
            sys.exit(f"No photo found in work/images_{tag}_{PHOTO_DIR[kind]}. Run the video step first.")
        photo = cands[0]
    text = script.get("thumbnail_text") or script["title"]
    text = text.upper() if lang == "en" else text
    lines = wrap_two_lines(text.strip())
    widest = max(len(l) for l in lines)
    size = int(min(h * (0.20 if kind == "long" else 0.085), (w * 0.9) / (max(widest, 4) * 0.76)))
    font = find_font(lang)
    fontopt = f"fontfile={font}:" if font else ""
    WORK.mkdir(exist_ok=True)
    chain = [f"scale={w}:{h}:force_original_aspect_ratio=increase", f"crop={w}:{h}",
             "eq=saturation=1.3:contrast=1.08:brightness=-0.05",
             f"drawbox=x=0:y=ih*0.42:w=iw:h=ih*0.58:color=black@0.5:t=fill"]
    gap = int(size * 1.18)
    base_y = int(h * (0.93 if kind == "long" else 0.80)) - gap * len(lines)
    colors = ["white", "0xFFD400"]
    for i, line in enumerate(lines):
        tf = WORK / f"thumb_{tag}_{i}.txt"
        tf.write_text(line, encoding="utf-8")
        chain.append(
            f"drawtext={fontopt}textfile={tf.name}:fontsize={size}:fontcolor={colors[i % 2]}:borderw={max(4, size // 14)}:"
            f"bordercolor=black:x=(w-text_w)/2:y={base_y + i * gap}"
        )
    out = (OUT / f"thumbnail_{tag}.jpg").resolve()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(photo.resolve()), "-vf", ",".join(chain),
           "-frames:v", "1", "-update", "1", "-q:v", "3", str(out)]
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed:\n" + "\n".join(r.stderr.strip().splitlines()[-15:]))
        sys.exit(1)
    return out


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "bn"], required=True)
    ap.add_argument("--kind", choices=["long", "short"], required=True)
    args = ap.parse_args()
    lang, kind = args.lang, args.kind
    tag = f"{lang}_{kind}"
    for need in (f"script_{tag}.json", f"timings_{tag}.json"):
        if not (OUT / need).exists():
            sys.exit(f"{OUT / need} not found. Run the earlier steps first.")
    script = json.loads((OUT / f"script_{tag}.json").read_text(encoding="utf-8"))
    timings = json.loads((OUT / f"timings_{tag}.json").read_text(encoding="utf-8"))
    cf = OUT / f"credits_{tag}.txt"
    credits = cf.read_text(encoding="utf-8") if cf.exists() else ""

    kit = build_kit(script, timings, credits, lang, kind)
    (OUT / f"upload_kit_{tag}.json").write_text(json.dumps(kit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"upload_kit_{tag}.txt").write_text(kit_text(kit, tag), encoding="utf-8")
    thumb = make_thumbnail(script, lang, kind, tag)
    print(f"OK [{tag}] thumbnail {thumb.name} ({thumb.stat().st_size / 1e3:.0f} KB), {len(kit['chapters'])} chapters, title: {kit['title']}")


if __name__ == "__main__":
    main()
