"""Step 1 (v2): topic + script generator for LONG (8+ min, 16:9) and SHORT (2-3 min, 9:16) videos,
English / Bangla, using Gemini.

Usage: python generate_script.py --lang en|bn --kind long|short
Out:   output/script_<lang>_<kind>.json     history/topics_<lang>.json (long only)

How: models write about 55-60% of the words you ask for in one go, so the script is built in pieces:
  1. an OUTLINE call (topic, title, hook, N sections with facts + photo ideas)
  2. one call per SECTION (each ~100 words), each checked and retried until its length fits
  3. if the total is still too short, the shortest sections are rewritten longer
The SHORT is built from the LONG's topic (run long first) so both videos cover the same subject.
Progress is cached in work/, so a retry continues instead of starting over.

Env:   GEMINI_API_KEY (required), GEMINI_MODEL (optional), GEN_BUDGET_SECONDS (default 1800)
       LONG_TARGET_SEC (default 540 = 9 min, keeps a safe margin above the 8-minute mid-roll limit)
       SHORT_TARGET_SEC (default 150 = 2.5 min), WPS_EN / WPS_BN (speaking pace, words per second;
       learned automatically from history/pace_<lang>.json after the first voice run)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models"
LANGS = {"en": "English", "bn": "Bengali (বাংলা)"}
OUT, HISTORY, WORK = Path("output"), Path("history"), Path("work")
BUDGET = int(os.environ.get("GEN_BUDGET_SECONDS", "1800"))
DEFAULT_WPS = {"en": float(os.environ.get("WPS_EN", "2.5")), "bn": float(os.environ.get("WPS_BN", "2.0"))}
KINDS = {
    "long": {"target": int(os.environ.get("LONG_TARGET_SEC", "540")), "min": 505, "max": 640, "sections": 12, "per_scene": 3},
    "short": {"target": int(os.environ.get("SHORT_TARGET_SEC", "150")), "min": 120, "max": 175, "sections": 4, "per_scene": 2},
}
BAD_OPENERS = (
    "welcome", "hello", "hi guys", "hey guys", "today we", "in this video",
    "স্বাগতম", "হ্যালো", "আসসালামু", "নমস্কার", "আজকে আমরা", "আজ আমরা", "এই ভিডিওতে",
)
HOOK_SIGNAL = re.compile(
    r"[0-9০-৯]|\?|কেন|কীভাবে|কেউ|কখনো|কখনও|\bwhy\b|\bhow\b|\bnever\b|\bnobody\b|\bno one\b"
    r"|\bonly\b|\bfirst\b|\blargest\b|\bsmallest\b|\boldest\b|\bfastest\b|\bthan\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------- helpers
def words_of(text: str) -> int:
    return len(text.split())


def sentences_of(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.lower())


def lint_hook(hook: str) -> list[str]:
    issues = []
    n = words_of(hook)
    if n < 5:
        issues.append("hook is too short (min 5 words)")
    if n > 22:
        issues.append("hook is too long (max 22 words)")
    if hook.lower().strip().startswith(BAD_OPENERS):
        issues.append("hook starts with a greeting/intro; open with the surprising fact")
    if not HOOK_SIGNAL.search(hook):
        issues.append("hook has no number, question or curiosity word")
    return issues


def load_wps(lang: str) -> float:
    f = HISTORY / f"pace_{lang}.json"
    try:
        v = float(json.loads(f.read_text(encoding="utf-8"))["wps"])
        if 1.0 <= v <= 4.5:
            return v
    except (OSError, ValueError, KeyError):
        pass
    return DEFAULT_WPS[lang]


# ----------------------------------------------------------------- gemini
def list_candidates(headers: dict) -> list[str]:
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return [m.strip() for m in forced.split(",") if m.strip()]
    fallback = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-3.5-flash"]
    try:
        r = requests.get(f"{API}?pageSize=200", headers=headers, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa: BLE001 - discovery must never crash the run
        print(f"model discovery failed ({type(e).__name__}); using built-in list")
        return fallback
    skip = ("image", "tts", "preview", "exp", "live", "thinking", "audio", "native", "robotics", "computer", "lite")
    flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(x in n for x in skip)]
    flash.sort(key=lambda n: tuple(int(x) for x in re.findall(r"\d+", n)), reverse=True)
    lite = [n for n in names if "flash-lite" in n and "preview" not in n and "exp" not in n]
    return (flash + [n for n in lite if n not in flash]) or fallback


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for cand in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return None


def call_gemini(prompt: str, headers: dict, candidates: list[str], deadline: float):
    """Parsed JSON, or None when the time budget ran out."""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0, "maxOutputTokens": 16384}}
    rnd = 0
    while True:
        rnd += 1
        for model in candidates:
            remaining = deadline - time.time()
            if remaining <= 2:
                return None
            try:
                r = requests.post(f"{API}/{model}:generateContent", headers=headers, json=body, timeout=min(120, remaining))
            except requests.RequestException as e:
                print(f"[round {rnd}] {model}: {type(e).__name__}, next")
                continue
            if r.status_code != 200:
                if r.status_code in (400, 401, 403) and ("API_KEY_INVALID" in r.text or "API key not valid" in r.text):
                    print("The Gemini API key is invalid. Check the GEMINI_API_KEY secret.")
                    sys.exit(3)
                print(f"[round {rnd}] {model}: HTTP {r.status_code}, next")
                continue
            try:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, ValueError):
                print(f"[round {rnd}] {model}: empty/blocked answer, next")
                continue
            parsed = parse_json(text)
            if parsed is None:
                print(f"[round {rnd}] {model}: not valid JSON, next")
                continue
            return parsed
        wait = min(15 * rnd, 90, max(0.0, deadline - time.time() - 2))
        if wait <= 0:
            return None
        print(f"all models busy; waiting {int(wait)}s (round {rnd} done)")
        time.sleep(wait)


# ----------------------------------------------------------------- prompts
def outline_prompt(lang: str, kind: str, n: int, used: list[str], base: dict | None, feedback: str, chosen: dict | None = None) -> str:
    name = LANGS[lang]
    fb = f"\nFix this from the previous attempt: {feedback}\n" if feedback else ""
    if base:
        topic_part = (
            f"This is the SHORT version of an existing long video. Keep the SAME topic: \"{base['topic']}\" "
            f"(long video title: \"{base['title']}\"). Its sections and facts:\n"
            + "\n".join(f"- {s['heading']}: {'; '.join(s['points'])}" for s in base["sections"])
            + f"\nPick only the {n - 2} most surprising facts for the middle sections. Use a NEW hook, different from the long video."
        )
    elif chosen:
        topic_part = (f"Today's topic was chosen from trending signals: \"{chosen['topic']}\"."
                      + (f" Suggested angle: {chosen['angle']}." if chosen.get("angle") else "")
                      + "\nUse exactly this subject; take an original angle and never copy other videos' titles or wording.")
    else:
        topic_part = ("Pick ONE fresh topic (science, history, nature, space, human body, world records, geography).\n"
                      "Topics already used (do NOT repeat or paraphrase): " + ("; ".join(used[-80:]) or "none yet"))
    shape = "a vertical Short of about 2.5 minutes" if kind == "short" else "a long YouTube video of about 9 minutes"
    return f"""You plan a faceless YouTube "amazing facts" video in {name}: {shape}.
{topic_part}
{fb}
Rules:
- Use only well-established, verifiable facts. If you are not sure of a number, leave it out.
- Exactly {n} sections. Section 1 opens with the hook; the last section wraps up and ends with a question inviting comments.
- hook: ONE spoken line (5-22 words) that opens with the most surprising fact, ideally with a number or a question. No greeting.
- title: max 70 characters. thumbnail_text: max 4 words. Write title, hook, thumbnail_text, description, headings and points in {name}.
- Each section: heading, 2-4 'points' (the facts to cover), and exactly 3 'image_queries'.
- image_queries are in English, 3-5 words, something a stock PHOTO can show (places, animals, objects, nature, science equipment).
  No abstract ideas, no text, no brand or person names.
- sources: 2 reputable places a human can check the facts (e.g. NASA, Britannica, Nature).

Return ONLY JSON:
{{"topic": str, "title": str, "hook": str, "thumbnail_text": str, "description": str, "tags": [str], "sources": [str],
  "sections": [{{"heading": str, "points": [str], "image_queries": [str, str, str]}}]}}"""


def section_prompt(lang: str, outline: dict, i: int, words: int, prev_tail: str, feedback: str) -> str:
    name = LANGS[lang]
    n = len(outline["sections"])
    sec = outline["sections"][i]
    lo, hi = int(words * 0.85), int(words * 1.2)
    rules = []
    if i == 0:
        rules.append(f"Begin with EXACTLY this line, word for word: {outline['hook']}")
    else:
        rules.append("Do NOT greet and do NOT introduce the video. Continue naturally from the previous section.")
    rules.append("End by asking the viewers one question about the topic, inviting comments." if i == n - 1
                 else "Do NOT wrap up or say goodbye; more sections follow.")
    fb = f"\nPrevious attempt problem: {feedback}\n" if feedback else ""
    earlier = "; ".join(s["heading"] for s in outline["sections"][:i]) or "none"
    return f"""You write the spoken narration for one section of a faceless YouTube facts video in {name}.
Video title: {outline['title']}
All sections: {" | ".join(s['heading'] for s in outline['sections'])}
Sections already narrated (do not repeat their facts): {earlier}
Previous section ended with: "{prev_tail}"
{fb}
Now write section {i + 1} of {n}: "{sec['heading']}"
Cover these points: {"; ".join(sec['points'])}

Rules:
- Length: about {words} words (must be between {lo} and {hi}). Count them.
- Natural spoken {name}, short sentences, concrete detail. Only well-established facts; skip any number you are unsure of.
- {rules[0]}
- {rules[1]}
- Plain text only: no headings, no bullet points, no markdown, no stage directions.

Return ONLY JSON: {{"narration": str}}"""


# ----------------------------------------------------------------- building blocks
def get_outline(lang, kind, n, used, base, headers, cands, deadline, chosen=None):
    feedback = ""
    for attempt in range(1, 4):
        data = call_gemini(outline_prompt(lang, kind, n, used, base, feedback, chosen), headers, cands, deadline)
        if data is None:
            return None
        secs = data.get("sections") if isinstance(data, dict) else None
        ok = (isinstance(secs, list) and len(secs) >= max(3, n - 2) and data.get("title") and data.get("hook")
              and all(isinstance(s, dict) and s.get("heading") for s in secs))
        if ok:
            for s in secs:
                pts = s.get("points") or [s["heading"]]
                s["points"] = [str(p) for p in (pts if isinstance(pts, list) else [pts])]
                q = s.get("image_queries") or [s["heading"]]
                s["image_queries"] = [str(x) for x in (q if isinstance(q, list) else [q])] or [s["heading"]]
            data["sections"] = secs[:n]
            for k in ("topic", "thumbnail_text", "description"):
                data[k] = str(data.get(k) or data["title"])
            for k in ("tags", "sources"):
                v = data.get(k) or []
                data[k] = [str(x) for x in (v if isinstance(v, list) else [v])]
            return data
        feedback = f"return exactly {n} sections, each with heading, points and image_queries, plus title and hook"
        print(f"outline attempt {attempt} unusable")
    return None


def write_section(lang, outline, i, words, prev_tail, headers, cands, deadline):
    best, best_gap, feedback = None, 10 ** 9, ""
    lo, hi = int(words * 0.8), int(words * 1.3)
    for attempt in range(1, 4):
        data = call_gemini(section_prompt(lang, outline, i, words, prev_tail, feedback), headers, cands, deadline)
        if data is None:
            return best
        text = str(data.get("narration", "")).replace("\n", " ").strip() if isinstance(data, dict) else ""
        text = re.sub(r"[*#_`]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if i == 0 and text and not norm(text).startswith(norm(outline["hook"])[:30]):
            text = outline["hook"].strip() + " " + text
        n = words_of(text)
        if n and abs(n - words) < best_gap:
            best, best_gap = text, abs(n - words)
        print(f"section {i + 1}: attempt {attempt}: {n} words (target {words})")
        if lo <= n <= hi:
            return text
        feedback = f"you wrote {n} words; write about {words} words ({'longer, add concrete detail' if n < lo else 'shorter'})"
    return best


def split_scenes(narration: str, queries: list[str], k: int, section: int) -> list[dict]:
    sents = sentences_of(narration) or [narration]
    k = max(1, min(k, len(sents)))
    total = words_of(narration)
    groups, cur, acc = [], [], 0
    for s in sents:
        cur.append(s)
        acc += words_of(s)
        if len(groups) < k - 1 and acc >= total * (len(groups) + 1) / k:
            groups.append(" ".join(cur))
            cur = []
    if cur:
        groups.append(" ".join(cur))
    return [{"narration": g, "image_query": queries[j % len(queries)], "section": section} for j, g in enumerate(groups)]


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    ap.add_argument("--kind", choices=KINDS, required=True)
    args = ap.parse_args()
    lang, kind = args.lang, args.kind
    cfg = KINDS[kind]

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    for d in (OUT, HISTORY, WORK):
        d.mkdir(exist_ok=True)

    hist_file = HISTORY / f"topics_{lang}.json"
    try:
        used = json.loads(hist_file.read_text(encoding="utf-8")) if hist_file.exists() else []
    except ValueError:
        used = []

    base = None
    if kind == "short":
        lp = OUT / f"script_{lang}_long.json"
        if lp.exists():
            try:
                lj = json.loads(lp.read_text(encoding="utf-8"))
                if lj.get("outline"):
                    base = {"topic": lj["topic"], "title": lj["title"], "sections": lj["outline"]}
            except ValueError:
                pass
        if base is None:
            print("no long script found: the short will pick its own topic")

    chosen = None
    tf = OUT / f"topic_{lang}.json"
    if kind == "long" and tf.exists():
        try:
            chosen = json.loads(tf.read_text(encoding="utf-8"))
            print(f"using the trending topic: {chosen['topic']}")
        except (ValueError, KeyError):
            chosen = None

    wps = load_wps(lang)
    n = cfg["sections"]
    target_words = int(cfg["target"] * wps)
    per = max(40, target_words // n)
    print(f"[{lang}/{kind}] pace {wps:.2f} words/s -> target {target_words} words in {n} sections (~{per} each)")

    cache_f = WORK / f"script_cache_{lang}_{kind}.json"
    cache = {}
    if cache_f.exists():
        try:
            cache = json.loads(cache_f.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    def save_cache():
        cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    cands = list_candidates(headers)
    print("models to try:", ", ".join(cands))
    deadline = time.time() + BUDGET

    outline = cache.get("outline")
    if not outline:
        outline = get_outline(lang, kind, n, used, base, headers, cands, deadline, chosen)
        if outline is None:
            sys.exit("Could not get an outline from Gemini within the time budget. Run again later.")
        cache["outline"] = outline
        save_cache()
    n = len(outline["sections"])
    narr: dict = cache.setdefault("narr", {})

    for i in range(n):
        if str(i) in narr:
            continue
        prev_tail = " ".join(sentences_of(narr.get(str(i - 1), ""))[-1:]) if i else ""
        text = write_section(lang, outline, i, per, prev_tail, headers, cands, deadline)
        if not text:
            sys.exit(f"Time budget used up at section {i + 1}. Progress is kept; run again.")
        narr[str(i)] = text
        save_cache()

    # make it longer if still short of the minimum duration
    for pass_no in range(2):
        total = sum(words_of(narr[str(i)]) for i in range(n))
        if total / wps >= cfg["min"]:
            break
        weakest = sorted(range(n), key=lambda i: words_of(narr[str(i)]))[: max(2, n // 3)]
        print(f"expand pass {pass_no + 1}: {total} words = {total / wps:.0f}s, need {cfg['min']}s; rewriting {len(weakest)} sections longer")
        for i in weakest:
            prev_tail = " ".join(sentences_of(narr.get(str(i - 1), ""))[-1:]) if i else ""
            text = write_section(lang, outline, i, int(per * 1.4), prev_tail, headers, cands, deadline)
            if text and words_of(text) > words_of(narr[str(i)]):
                narr[str(i)] = text
        save_cache()

    scenes = []
    for i in range(n):
        sec = outline["sections"][i]
        scenes += split_scenes(narr[str(i)], sec["image_queries"], cfg["per_scene"], i)
    total_words = sum(words_of(s["narration"]) for s in scenes)
    est = total_words / wps

    warnings = lint_hook(outline["hook"])
    if len(outline["title"]) > 70:
        warnings.append("title longer than 70 characters")
    if not (cfg["min"] <= est <= cfg["max"]):
        warnings.append(f"estimated length {est:.0f}s is outside {cfg['min']}-{cfg['max']}s")
    if outline["topic"].strip().lower() in {u.strip().lower() for u in used} and not base:
        warnings.append("topic was already used")

    result = {
        "kind": kind, "topic": outline["topic"], "title": outline["title"], "hook": outline["hook"],
        "scenes": scenes, "thumbnail_text": outline["thumbnail_text"], "description": outline["description"],
        "tags": outline["tags"], "sources": outline["sources"], "estimated_seconds": round(est),
        "outline": outline["sections"],
    }
    if warnings:
        result["warnings"] = warnings
        print("warnings: " + "; ".join(warnings))
    tmp = OUT / f"script_{lang}_{kind}.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / f"script_{lang}_{kind}.json")
    if not base:
        tmp = hist_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(used + [outline["topic"]], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(hist_file)
    cache_f.unlink(missing_ok=True)
    print(f"OK [{lang}/{kind}] {outline['title']} | {len(scenes)} scenes, {total_words} words, ~{est:.0f}s")


if __name__ == "__main__":
    main()
