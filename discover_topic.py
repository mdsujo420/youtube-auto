"""Step 0: find a trending, evergreen topic for today's video.

Usage: python discover_topic.py --lang en|bn
Writes: output/topic_<lang>.json   (generate_script.py picks it up automatically for the long video)

Signals (all free):
  1. YouTube Data API: videos about "amazing facts" from the last 7 days whose views are high compared with the
     channel's size (outliers). Needs YOUTUBE_API_KEY (read-only key; no upload permission, no audit needed).
  2. Wikipedia: the most-read articles yesterday (no key).
  3. Gemini turns the signals into ONE evergreen, photographable, safe topic that is not used yet.
The signals are only inspiration for the SUBJECT: titles and wording are never copied.
If anything fails the script exits 0 without a topic file, and generate_script.py picks a topic itself.

Env: YOUTUBE_API_KEY (optional but recommended), GEMINI_API_KEY (required), GEN_BUDGET_SECONDS (default 300)
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

from generate_script import LANGS, call_gemini, list_candidates  # same repo folder

OUT, HISTORY = Path("output"), Path("history")
YT_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()
BUDGET = int(os.environ.get("GEN_BUDGET_SECONDS", "300"))
UA = {"User-Agent": "youtube-auto-topic-finder/1.0 (personal automation)"}

# where to look: (relevance language, search phrases, wikipedia project)
SOURCES = {
    "en": [("en", ["amazing facts", "mind blowing facts", "facts you never knew"], "en.wikipedia")],
    "bn": [("bn", ["অবাক করা তথ্য", "মজার তথ্য", "আশ্চর্য তথ্য"], "bn.wikipedia"),
           ("en", ["amazing facts", "mind blowing facts"], "en.wikipedia")],
}
WIKI_SKIP = ("Main_Page", "Special:", "Wikipedia:", "Portal:", "File:", "Help:", "Category:", "Template:", "প্রধান_পাতা", "বিশেষ:")


def youtube_outliers(rel_lang: str, queries: list[str]) -> list[dict]:
    if not YT_KEY:
        return []
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids: dict[str, dict] = {}
    for q in queries:
        try:
            r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=30, params={
                "part": "snippet", "q": q, "type": "video", "order": "viewCount", "publishedAfter": since,
                "maxResults": 25, "relevanceLanguage": rel_lang, "safeSearch": "moderate", "key": YT_KEY})
        except requests.RequestException as e:
            print(f"youtube search failed ({type(e).__name__})")
            continue
        if r.status_code != 200:
            print(f"youtube search HTTP {r.status_code}: {r.text[:160]}")
            continue
        for it in r.json().get("items", []):
            vid = it.get("id", {}).get("videoId")
            if vid:
                ids[vid] = {"title": it["snippet"]["title"], "channel": it["snippet"]["channelId"]}
    if not ids:
        return []
    stats, chan = {}, {}
    try:
        vids = list(ids)
        for i in range(0, len(vids), 50):
            r = requests.get("https://www.googleapis.com/youtube/v3/videos", timeout=30,
                             params={"part": "statistics", "id": ",".join(vids[i:i + 50]), "key": YT_KEY})
            for it in r.json().get("items", []):
                stats[it["id"]] = int(it.get("statistics", {}).get("viewCount", 0))
        chs = sorted({v["channel"] for v in ids.values()})
        for i in range(0, len(chs), 50):
            r = requests.get("https://www.googleapis.com/youtube/v3/channels", timeout=30,
                             params={"part": "statistics", "id": ",".join(chs[i:i + 50]), "key": YT_KEY})
            for it in r.json().get("items", []):
                s = it.get("statistics", {})
                chan[it["id"]] = None if s.get("hiddenSubscriberCount") else int(s.get("subscriberCount", 0))
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"youtube stats failed ({type(e).__name__})")
        return []
    out = []
    for vid, meta in ids.items():
        views, subs = stats.get(vid, 0), chan.get(meta["channel"])
        if subs is None or views < 3000 or subs > 1_000_000:
            continue
        out.append({"title": meta["title"], "views": views, "subs": subs, "ratio": round(views / max(subs, 1000), 2)})
    out.sort(key=lambda x: x["ratio"], reverse=True)
    return out[:12]


def wikipedia_top(project: str) -> list[str]:
    for back in (1, 2, 3):
        d = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=back)
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/all-access/{d:%Y/%m/%d}"
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                arts = r.json()["items"][0]["articles"]
                return [a["article"].replace("_", " ") for a in arts if not a["article"].startswith(WIKI_SKIP)][:25]
        except (requests.RequestException, ValueError, KeyError, IndexError):
            pass
    return []


def choose_prompt(lang: str, yt: list[dict], wiki: list[str], used: list[str]) -> str:
    name = LANGS[lang]
    yt_lines = "\n".join(f"- {v['title']} | {v['views']} views | channel {v['subs']} subs | x{v['ratio']}" for v in yt) or "- (none available)"
    return f"""You choose today's topic for a faceless YouTube "amazing facts" channel in {name}.

Signals of what people are curious about right now. Use them ONLY as inspiration for the SUBJECT; never copy a title or wording.
YouTube facts videos that did unusually well for their channel size this week:
{yt_lines}
Most-read Wikipedia articles yesterday: {"; ".join(wiki) or "(none available)"}

Topics already used (do not repeat or paraphrase): {"; ".join(used[-80:]) or "none yet"}

Rules:
- ONE evergreen subject with 8+ surprising, verifiable facts that can be shown with stock PHOTOS
  (nature, animals, space, science, history, geography, the human body, records, everyday objects).
- Prefer a subject linked to a signal above, but take an original angle.
- Avoid: politics, breaking news, disasters and tragedies, crime, living people, celebrities, brands,
  copyrighted franchises, medical advice, anything sensitive or adult.
- Write topic, angle and candidates in {name}.

Return ONLY JSON: {{"topic": str, "angle": str, "reason": str, "candidates": [str, str, str, str]}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    lang = ap.parse_args().lang
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set; skipping topic discovery")
        return
    OUT.mkdir(exist_ok=True)
    hist = HISTORY / f"topics_{lang}.json"
    try:
        used = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else []
    except ValueError:
        used = []

    yt, wiki = [], []
    for rel, queries, project in SOURCES[lang]:
        yt += youtube_outliers(rel, queries)
        wiki += [w for w in wikipedia_top(project) if w not in wiki][:15]
    yt.sort(key=lambda x: x["ratio"], reverse=True)
    yt = yt[:15]
    print(f"signals: {len(yt)} YouTube outliers, {len(wiki)} Wikipedia articles" + ("" if YT_KEY else " (no YOUTUBE_API_KEY)"))
    if not yt and not wiki:
        print("no signals at all; the script step will pick a topic itself")
        return

    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    data = call_gemini(choose_prompt(lang, yt, wiki, used), headers, list_candidates(headers), time.time() + BUDGET)
    if not isinstance(data, dict) or not str(data.get("topic", "")).strip():
        print("Gemini gave no topic; the script step will pick one itself")
        return
    topic = str(data["topic"]).strip()
    used_l = {u.strip().lower() for u in used}
    if topic.lower() in used_l:
        alt = [str(c).strip() for c in data.get("candidates", []) if str(c).strip().lower() not in used_l]
        if not alt:
            print("chosen topic already used and no alternative; the script step will pick one itself")
            return
        topic = alt[0]
        data["angle"] = ""
    result = {"topic": topic, "angle": str(data.get("angle", "")), "reason": str(data.get("reason", "")),
              "candidates": data.get("candidates", []),
              "signals": {"youtube": yt, "wikipedia": wiki}}
    (OUT / f"topic_{lang}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK topic [{lang}]: {topic}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - discovery must never break the pipeline
        print(f"topic discovery failed ({type(e).__name__}: {e}); the script step will pick a topic itself")
