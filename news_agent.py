"""
24/7 Macro News Sentiment + Burst Detector — news_agent.py
==========================================================
The free first stage of the live-news layer. Runs on its own cloud cron, pulls
macro/market headlines from free RSS feeds (no API key, no cost), scores them with
VADER, and writes news_state.json — which the forward logger reads to stamp signals
and to STAND ASIDE when news is breaking (the same "trade the calm, not the chaos"
thesis the RVOL filter proved).

Two outputs:
  • news_state.json  — current snapshot {avg_sentiment, burst, mood, top headlines}
  • news_log.csv     — append-only history (so you can study news vs. your trades)

Zero LLM tokens (VADER is rule-based). Upgrading the SCORER to a Claude agent later
is a one-function swap (score_headline).

Run:  python3 ~/Desktop/news_agent.py
"""

import os, csv, json, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_HERE  = os.path.dirname(os.path.abspath(__file__))
STATE  = os.path.join(_HERE, "news_state.json")
LOG    = os.path.join(_HERE, "news_log.csv")

# Free, no-auth RSS feeds focused on macro / markets (what moves an index).
FEEDS = [
    "http://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",      # CNBC top news
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",       # CNBC economy
    "https://www.federalreserve.gov/feeds/press_all.xml",         # Fed press releases
    "https://www.investing.com/rss/news_25.rss",                  # economic indicators
]

BURST_MINUTES = 90     # window for "is news breaking right now"
BURST_COUNT   = 8      # >= this many fresh headlines in the window = a burst
STRONG_SENT   = 0.4    # |avg compound| above this = a clear risk-on/off tilt

UA = {"User-Agent": "Mozilla/5.0 (news-agent; macro sentiment)"}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def parse(raw):
    """Return list of (title, datetime-utc-or-None) from an RSS/Atom feed."""
    out = []
    root = ET.fromstring(raw)
    for item in root.iter():
        if item.tag.lower().endswith("item") or item.tag.lower().endswith("entry"):
            title = pub = None
            for c in item:
                tag = c.tag.lower()
                if tag.endswith("title"):
                    title = (c.text or "").strip()
                elif tag.endswith("pubdate") or tag.endswith("published") or tag.endswith("updated"):
                    try:
                        pub = parsedate_to_datetime(c.text)
                        if pub and pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                    except Exception:
                        pub = None
            if title:
                out.append((title, pub))
    return out


def score_headline(analyzer, text):
    """VADER compound score in [-1, 1]. Swap this for a Claude call to upgrade."""
    return analyzer.polarity_scores(text)["compound"]


def notify(msg):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    hook = os.environ.get("WEBHOOK_URL")
    try:
        if tok and chat:
            url, body = (f"https://api.telegram.org/bot{tok}/sendMessage",
                         json.dumps({"chat_id": chat, "text": msg}).encode())
        elif hook:
            url, body = hook, json.dumps({"content": msg, "text": msg}).encode()
        else:
            return
        urllib.request.urlopen(urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}), timeout=10)
    except Exception as e:
        print(f"(notify failed: {e})")


def main():
    now = datetime.now(timezone.utc)
    analyzer = SentimentIntensityAnalyzer()

    items, live_feeds = [], 0
    for url in FEEDS:
        try:
            items.extend(parse(fetch(url))); live_feeds += 1
        except Exception as e:
            print(f"  feed skipped ({url.split('/')[2]}): {e}")

    if not items:
        print("No headlines fetched — all feeds failed."); return

    # Dedupe by title; score each.
    seen, scored = set(), []
    for title, pub in items:
        if title in seen:
            continue
        seen.add(title)
        scored.append((title, pub, score_headline(analyzer, title)))

    recent = [s for s in scored if s[1] and (now - s[1]).total_seconds() <= BURST_MINUTES*60]
    basis  = recent if recent else scored
    avg    = sum(s[2] for s in basis) / len(basis)
    burst  = len(recent) >= BURST_COUNT
    mood   = "risk-on" if avg > STRONG_SENT else "risk-off" if avg < -STRONG_SENT else "neutral"

    top = sorted(basis, key=lambda s: abs(s[2]), reverse=True)[:5]
    state = {
        "ts": now.strftime("%Y-%m-%d %H:%M UTC"),
        "feeds_live": live_feeds, "headlines": len(scored),
        "recent": len(recent), "avg_sentiment": round(avg, 3),
        "burst": burst, "mood": mood,
        "top": [{"t": t, "s": round(sc, 2)} for t, _, sc in top],
    }
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)

    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["ts", "headlines", "recent", "avg_sentiment", "burst", "mood"])
        w.writerow([state["ts"], len(scored), len(recent), round(avg, 3), burst, mood])

    print(f"News: {len(scored)} headlines ({live_feeds} feeds) | "
          f"recent {len(recent)} | avg {avg:+.3f} | {mood}"
          f"{' | ⚠️ BURST' if burst else ''}")
    for t, _, sc in top[:3]:
        print(f"   {sc:+.2f}  {t[:70]}")

    if burst or abs(avg) > STRONG_SENT:
        notify(f"📰 Macro news: {mood} (avg {avg:+.2f}, {len(recent)} recent"
               f"{', BURST' if burst else ''})\n• " +
               "\n• ".join(t[:80] for t, _, _ in top[:3]))


if __name__ == "__main__":
    main()
