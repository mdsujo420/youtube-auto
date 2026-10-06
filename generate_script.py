"""Step 1: facts topic + script generator (English / Bangla) using Gemini.

Usage: python generate_script.py --lang en|bn
Env:   GEMINI_API_KEY (required), GEMINI_MODEL (optional; if empty the newest available flash model is auto-picked)
Out:   output/script_<lang>.json   and   history/topics_<lang>.json
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

BAD_OPENERS = (
    "welcome", "hello", "hi guys", "hey guys", "today we", "in this video",
    "স্বাগতম", "হ্যালো", "আসসালামু", "নমস্কার", "আজকে আমরা", "আজ আমরা", "এই ভিডিওতে",
)
HOOK_SIGNAL = re.compile(
    r"[0-9০-৯]|\?|কেন|কীভাবে|কেউ|কখনো|কখনও|\bwhy\b|\bhow\b|\bnever\b|\bnobody\b|\bno one\b",
    re.IGNORECASE,
)


def lint_hook(hook: str) -> list[str]:
    """Cheap sanity checks for the first line. Returns a list of problems."""
    problems = []
    words = hook.split()
    if len(words) < 5:
        problems.append("hook is too short (min 5 words)")
    if len(words) > 22:
        problems.append("hook is too long (max 22 words)")
    if hook.lower().strip().startswith(BAD_OPENERS):
        problems.append("hook starts with a generic greeting/intro; start with the surprising fact")
    if not HOOK_SIGNAL.search(hook):
        problems.append("hook has no number, question or curiosity word")
    return problems


def build_prompt(lang: str, used: list[str], feedback: str) -> str:
    name = LANGS[lang]
    avoid = "; ".join(used[-60:]) or "none yet"
    fb = f"\nFix this from the previous attempt: {feedback}\n" if feedback else ""
    return f"""You write scripts for a faceless YouTube "amazing facts" channel in {name}.
Pick ONE fresh topic (science, history, nature, space, human body, world records, geography).
Topics already used (do NOT repeat or paraphrase): {avoid}
{fb}
Rules:
- Use only well-established, verifiable facts. If you are not sure of a number, do not use it.
- The hook is ONE spoken line (5-22 words) that opens with the most surprising fact. No greeting.
- 7 to 9 scenes. Total narration 180-260 words. Short, spoken sentences.
- Last scene ends with a one-line question that invites a comment.
- Write narration, title, hook, thumbnail_text and description in {name}.
- image_query must be in English (used to find stock images), 3-6 words, visual and concrete.
- thumbnail_text: max 4 words.
- sources: 2 reputable places a human can check the facts (e.g. NASA, Britannica, Nature).

Return ONLY JSON with this shape:
{{"topic": str, "title": str, "hook": str,
  "scenes": [{{"narration": str, "image_query": str}}],
  "thumbnail_text": str, "description": str, "tags": [str], "sources": [str]}}"""


def list_candidates(headers: dict) -> list[str]:
    """Model names to try, newest flash first. GEMINI_MODEL overrides."""
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return [forced]
    r = requests.get(f"{API}?pageSize=200", headers=headers, timeout=60)
    r.raise_for_status()
    skip = ("lite", "image", "tts", "preview", "exp", "live", "thinking", "audio", "native", "robotics", "computer")
    names = [
        m["name"].split("/")[-1]
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(x in n for x in skip)]
    flash.sort(key=lambda n: tuple(int(x) for x in re.findall(r"\d+", n)), reverse=True)
    lite = [n for n in names if "flash-lite" in n and "preview" not in n and "exp" not in n]
    out = flash + [n for n in lite if n not in flash]
    if not out:
        sys.exit("No usable Gemini model found for this API key. Models seen: " + ", ".join(names[:20]))
    return out


def call_gemini(prompt: str) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
    }
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    for model in list_candidates(headers):
        for attempt in range(3):
            r = requests.post(f"{API}/{model}:generateContent", headers=headers, json=body, timeout=120)
            if r.status_code in (500, 502, 503, 504) or (r.status_code == 429 and attempt < 2):
                time.sleep(10 * (attempt + 1))
                continue
            break
        if r.status_code in (404, 429, 403):
            print(f"model {model}: HTTP {r.status_code}, trying next")
            continue
        r.raise_for_status()
        print(f"using model {model}")
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        return json.loads(text)
    sys.exit("No Gemini model worked (404/quota). Check the API key and free-tier limits.")


def validate(data: dict) -> list[str]:
    problems = []
    for k in ("topic", "title", "hook", "scenes", "thumbnail_text", "description", "tags", "sources"):
        if not data.get(k):
            problems.append(f"missing field: {k}")
    if problems:
        return problems
    if not (7 <= len(data["scenes"]) <= 9):
        problems.append("scenes must be 7-9")
    words = sum(len(s.get("narration", "").split()) for s in data["scenes"])
    if not (150 <= words <= 300):
        problems.append(f"narration is {words} words; need 180-260")
    if len(data["title"]) > 70:
        problems.append("title longer than 70 characters")
    return problems + lint_hook(data["hook"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    lang = ap.parse_args().lang

    HISTORY.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    hist_file = HISTORY / f"topics_{lang}.json"
    used = json.loads(hist_file.read_text(encoding="utf-8")) if hist_file.exists() else []

    feedback = ""
    for attempt in range(3):
        data = call_gemini(build_prompt(lang, used, feedback))
        problems = validate(data)
        if not problems:
            break
        feedback = "; ".join(problems)
        print(f"attempt {attempt + 1} rejected: {feedback}")
    else:
        sys.exit("Could not get a script that passes the checks. Not saving.")

    (OUT / f"script_{lang}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    used.append(data["topic"])
    hist_file.write_text(json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK [{lang}] {data['title']}")


if __name__ == "__main__":
    main()
