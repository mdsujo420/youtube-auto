"""Step 1 (hardened): facts topic + script generator (English / Bangla) using Gemini.

Usage: python generate_script.py --lang en|bn
Env:   GEMINI_API_KEY      (required)
       GEMINI_MODEL        (optional, comma list; default = auto-discover, newest flash first)
       GEN_BUDGET_SECONDS  (optional, total time budget per run, default 900)
Out:   output/script_<lang>.json   and   history/topics_<lang>.json

Design goal: almost never fail. Busy servers, timeouts, bad JSON, unavailable models,
failed model discovery and imperfect scripts are all handled by retrying, failing over,
or saving the best candidate with a "warnings" field instead of throwing it away.
It only exits with an error if Gemini produced nothing usable for the whole time budget,
or the API key is invalid.
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
OUT = Path("output")
HISTORY = Path("history")
BUDGET = int(os.environ.get("GEN_BUDGET_SECONDS", "900"))
MAX_ATTEMPTS = 6
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-3.5-flash"]

BAD_OPENERS = (
    "welcome", "hello", "hi guys", "hey guys", "today we", "in this video",
    "স্বাগতম", "হ্যালো", "আসসালামু", "নমস্কার", "আজকে আমরা", "আজ আমরা", "এই ভিডিওতে",
)
HOOK_SIGNAL = re.compile(
    r"[0-9০-৯]|\?|কেন|কীভাবে|কেউ|কখনো|কখনও|\bwhy\b|\bhow\b|\bnever\b|\bnobody\b|\bno one\b"
    r"|\bonly\b|\bfirst\b|\blargest\b|\bsmallest\b|\boldest\b|\bfastest\b|\bthan\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------- checks
def lint_hook(hook: str) -> list[str]:
    issues = []
    n = len(hook.split())
    if n < 5:
        issues.append("hook is too short (min 5 words)")
    if n > 22:
        issues.append("hook is too long (max 22 words)")
    if hook.lower().strip().startswith(BAD_OPENERS):
        issues.append("hook starts with a greeting/intro; open with the surprising fact")
    if not HOOK_SIGNAL.search(hook):
        issues.append("hook has no number, question or curiosity word")
    return issues


def validate(data: dict, used: list[str]) -> list[str]:
    issues = []
    n_scenes = len(data["scenes"])
    if not (7 <= n_scenes <= 9):
        issues.append(f"{n_scenes} scenes; need 7-9")
    words = sum(len(s["narration"].split()) for s in data["scenes"])
    if not (180 <= words <= 260):
        issues.append(f"narration is {words} words; need 200-260")
    if len(data["title"]) > 70:
        issues.append("title longer than 70 characters")
    if data["topic"].strip().lower() in {u.strip().lower() for u in used}:
        issues.append("topic was already used; pick a different one")
    return issues + lint_hook(data["hook"])


# ----------------------------------------------------------------- prompt
def build_prompt(lang: str, used: list[str], feedback: str) -> str:
    name = LANGS[lang]
    avoid = "; ".join(used[-80:]) or "none yet"
    fb = f"\nFix this from the previous attempt: {feedback}\n" if feedback else ""
    return f"""You write scripts for a faceless YouTube "amazing facts" channel in {name}.
Pick ONE fresh topic (science, history, nature, space, human body, world records, geography).
Topics already used (do NOT repeat or paraphrase): {avoid}
{fb}
Rules:
- Use only well-established, verifiable facts. If you are not sure of a number, do not use it.
- The hook is ONE spoken line (5-22 words) that opens with the most surprising fact, ideally with a number or a question. No greeting.
- 7 to 9 scenes. Total narration MUST be 200-260 words (count them). Short, spoken sentences.
- Last scene ends with a one-line question that invites a comment.
- Write narration, title, hook, thumbnail_text and description in {name}.
- Title max 70 characters.
- image_query must be in English (used to find stock images), 3-6 words, visual and concrete.
- thumbnail_text: max 4 words.
- sources: 2 reputable places a human can check the facts (e.g. NASA, Britannica, Nature).

Return ONLY JSON with this shape:
{{"topic": str, "title": str, "hook": str,
  "scenes": [{{"narration": str, "image_query": str}}],
  "thumbnail_text": str, "description": str, "tags": [str], "sources": [str]}}"""


# ----------------------------------------------------------------- gemini
def list_candidates(headers: dict) -> list[str]:
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return [m.strip() for m in forced.split(",") if m.strip()]
    try:
        r = requests.get(f"{API}?pageSize=200", headers=headers, timeout=30)
        r.raise_for_status()
        names = [
            m["name"].split("/")[-1]
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    except Exception as e:  # noqa: BLE001 - discovery must never crash the run
        print(f"model discovery failed ({type(e).__name__}); using built-in list")
        return FALLBACK_MODELS
    skip = ("image", "tts", "preview", "exp", "live", "thinking", "audio", "native", "robotics", "computer", "lite")
    flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(x in n for x in skip)]
    flash.sort(key=lambda n: tuple(int(x) for x in re.findall(r"\d+", n)), reverse=True)
    lite = [n for n in names if "flash-lite" in n and "preview" not in n and "exp" not in n]
    out = flash + [n for n in lite if n not in flash]
    return out or FALLBACK_MODELS


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def call_gemini(prompt: str, headers: dict, candidates: list[str], deadline: float):
    """Returns parsed JSON, or None if the time budget ran out."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
    }
    rnd = 0
    while True:
        rnd += 1
        for model in candidates:
            remaining = deadline - time.time()
            if remaining <= 2:
                return None
            try:
                r = requests.post(
                    f"{API}/{model}:generateContent", headers=headers, json=body, timeout=min(90, remaining)
                )
            except requests.RequestException as e:
                print(f"[round {rnd}] {model}: {type(e).__name__}, next")
                continue
            if r.status_code != 200:
                if r.status_code in (400, 401, 403) and ("API_KEY_INVALID" in r.text or "API key not valid" in r.text):
                    sys.exit("The Gemini API key is invalid. Check the GEMINI_API_KEY secret.")
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
            print(f"using model {model}")
            return parsed
        wait = min(15 * rnd, 90, max(0.0, deadline - time.time() - 2))
        if wait <= 0:
            return None
        print(f"all models busy; waiting {int(wait)}s (round {rnd} done)")
        time.sleep(wait)


def normalize(raw):
    """Turn whatever the model returned into a clean dict, or None if unusable."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        return None
    scenes = []
    for s in raw.get("scenes") or []:
        if isinstance(s, str):
            s = {"narration": s}
        if isinstance(s, dict) and str(s.get("narration", "")).strip():
            scenes.append({
                "narration": str(s["narration"]).strip(),
                "image_query": str(s.get("image_query") or "").strip() or "abstract background",
            })
    title = str(raw.get("title") or "").strip()
    hook = str(raw.get("hook") or "").strip()
    if len(scenes) < 3 or not title or not hook:
        return None

    def as_list(v):
        if isinstance(v, str):
            v = [x.strip() for x in re.split(r"[,\n]", v) if x.strip()]
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    return {
        "topic": str(raw.get("topic") or title).strip(),
        "title": title,
        "hook": hook,
        "scenes": scenes,
        "thumbnail_text": str(raw.get("thumbnail_text") or " ".join(title.split()[:4])).strip(),
        "description": str(raw.get("description") or hook).strip(),
        "tags": as_list(raw.get("tags")),
        "sources": as_list(raw.get("sources")),
    }


# ----------------------------------------------------------------- main
def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    lang = ap.parse_args().lang

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    HISTORY.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    hist_file = HISTORY / f"topics_{lang}.json"
    try:
        used = json.loads(hist_file.read_text(encoding="utf-8")) if hist_file.exists() else []
    except ValueError:
        used = []

    candidates = list_candidates(headers)
    print("models to try:", ", ".join(candidates))
    deadline = time.time() + BUDGET

    best = None  # (data, issues)
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = call_gemini(build_prompt(lang, used, feedback), headers, candidates, deadline)
        if raw is None:
            print("time budget used up")
            break
        data = normalize(raw)
        if data is None:
            feedback = "return complete JSON with topic, title, hook and 7-9 scenes"
            print(f"attempt {attempt}: unusable structure")
            continue
        issues = validate(data, used)
        if best is None or len(issues) < len(best[1]):
            best = (data, issues)
        if not issues:
            break
        feedback = "; ".join(issues)
        print(f"attempt {attempt} has issues: {feedback}")

    if best is None:
        sys.exit("Gemini produced nothing usable within the time budget. Run again later.")

    data, issues = best
    if issues:
        data["warnings"] = issues
        print("saved best attempt with warnings: " + "; ".join(issues))
    write_atomic(OUT / f"script_{lang}.json", data)
    write_atomic(hist_file, used + [data["topic"]])
    print(f"OK [{lang}] {data['title']}")


if __name__ == "__main__":
    main()
