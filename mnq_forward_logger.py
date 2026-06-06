"""
MNQ Forward Signal Logger — mnq_forward_logger.py
=================================================
The only test the morning deep-VWAP MACD-hook edge hasn't passed is LIVE data.
Backtests can be overfit; forward data cannot. This script accumulates that
forward record so you fund the system on proof, not hope.

It uses the EXACT gates, scoring, and conviction sizing from backtest7/backtest8,
so every signal it logs is identical to what we validated. Run it daily (or on a
loop during 8:30–10:30am / 1:00–2:30pm CT). Each run does three things:

  1. RESOLVE  — walks every still-open journal entry forward through real
                subsequent price action and records WIN / LOSS / EXPIRED plus
                PnL net of slippage + commission.
  2. SCAN     — logs any NEW qualifying setup from the last ~2 trading days
                (deduped by bar timestamp). No 60-day backfill — the journal is
                a pure forward record from the day you start running it.
  3. REPORT   — journal summary; once ≥10 trades have CLOSED, it bootstraps the
                forward results so you can see P(profitable) on real data.

Journal: ~/Desktop/mnq_forward_journal.csv   (safe to open in Excel/Numbers)

Run:  python3 ~/Desktop/mnq_forward_logger.py
"""

import os
import random
import pandas as pd

from backtest7_final import (
    load_data, add_indicators, get_slow_conf, score,
    MIN_BARS, MAX_HOLD_BARS, CONF_MIN, CONF_MAX, MACD_FLOOR, MACD_CEIL,
)
from backtest8_robust import (
    contracts, POINT_VALUE, TICK, SLIP_TICKS, COMMISSION_RT, stats,
)

# Journal/state live next to this script — works locally AND on a cloud runner
# (GitHub Actions checks the repo out, so these land in the repo and get committed).
_HERE   = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(_HERE, "mnq_forward_journal.csv")
STATE   = os.path.join(_HERE, ".mnq_forward_state")           # last-scanned bar ts
SCAN_LOOKBACK_BARS = 576    # ~2 trading days; used ONLY on the very first run
BOOT_SEED = 42
RVOL_WINDOW = 20            # bars for the institutional-volume baseline
RVOL_SKIP   = 1.5          # PROVISIONAL stand-aside filter: skip signals during a
                           # volume spike (news/breakout blows through VWAP). Backtest
                           # showed WR 57→70% avoiding these — forward data confirms it.

COLUMNS = [
    "entry_ts", "status", "session", "zone", "near_vwap", "confidence",
    "contracts", "risk_pct", "entry", "entry_fill", "stop", "target",
    "macd", "rsi", "vix", "rvol", "news_sent", "news_burst",
    "exit_ts", "exit_price", "bars_held", "pnl",
]


# ── JOURNAL I/O ─────────────────────────────────────────────────────────────────
def load_journal():
    if os.path.exists(JOURNAL):
        return pd.read_csv(JOURNAL, dtype={"entry_ts": str, "exit_ts": str})
    return pd.DataFrame(columns=COLUMNS)


def save_journal(j):
    j.to_csv(JOURNAL, index=False)


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return f.read().strip() or None
    return None


def save_state(ts):
    with open(STATE, "w") as f:
        f.write(ts)


def ts_str(row):
    return row["ts"].strftime("%Y-%m-%d %H:%M")


# ── SIGNAL (identical to backtest7/8 gates + scoring) ───────────────────────────
def signal_for_row(row, slow):
    bt   = int(row["btime"])
    morn = (8*60+30) <= bt <= (10*60+30)
    aftn = (13*60)   <= bt <= (14*60+30)

    if row["bias"] != 1:                                       return None
    if row["VIX"]  > 30:                                       return None
    if not (morn or aftn):                                     return None
    if row["FRESH"] == 1:                                      return None
    if row["EMA9"]  > row["EMA21"]:                            return None
    mh = row["MACD_H"]
    if not (mh > row["MACD_P"]):                               return None
    if mh < MACD_FLOOR or mh > MACD_CEIL:                      return None
    px = row["Close"]
    if not pd.isna(row["PDL"]) and px < row["PDL"]:            return None
    if row.get("RVOL", 1.0) >= RVOL_SKIP:                      return None   # volume spike → stand aside

    conf, sigs = score(mh, bool(row["BB_C"]), row["RSI"], px, row["VWAP"],
                       row["ATR"], row["VIX"], morn, aftn, slow)
    if not (CONF_MIN <= conf < CONF_MAX):                      return None

    sl   = px - 1.5 * row["ATR"]
    risk = px - sl
    if risk <= 0:                                              return None
    tgt  = px + 2.5 * risk
    c    = contracts(px, sl, conf)
    risk_pct = 0.06 if conf >= 85 else 0.03

    return {
        "entry_ts":   ts_str(row),
        "status":     "OPEN",
        "session":    "AM" if morn else "PM",
        "zone":       "DEEP" if sigs["macd_deep"] else "MID",
        "near_vwap":  bool(sigs["near_vwap"]),
        "confidence": conf,
        "contracts":  c,
        "risk_pct":   risk_pct,
        "entry":      round(px, 2),
        "entry_fill": round(px + SLIP_TICKS * TICK, 2),
        "stop":       round(sl, 2),
        "target":     round(tgt, 2),
        "macd":       round(mh, 3),
        "rsi":        round(row["RSI"], 1),
        "vix":        round(row["VIX"], 1),
        "rvol":       round(float(row.get("RVOL", 1.0)), 2),
        "exit_ts":    "", "exit_price": "", "bars_held": "", "pnl": "",
    }


# ── SCAN: log every new signal since the last run (gap-proof) ───────────────────
def scan(df, slow, journal, pos_of, fresh_ts):
    known = set(journal["entry_ts"].astype(str))

    # Resume from the bar after the last one we scanned. On the first ever run
    # (or if that bar has aged out of the 60d window) fall back to ~2 days so we
    # don't backfill 60 days of history into the forward record.
    last = load_state()
    start = (pos_of[last] + 1) if (last and last in pos_of) \
            else max(MIN_BARS, len(df) - 1 - SCAN_LOOKBACK_BARS)

    new = []
    for i in range(start, len(df) - 1):       # skip last (possibly-forming) bar
        sig = signal_for_row(df.iloc[i], slow)
        if sig and sig["entry_ts"] not in known:
            sig["fresh"] = (sig["entry_ts"] == fresh_ts)
            known.add(sig["entry_ts"])
            new.append(sig)
    return new


# ── RESOLVE: walk open trades forward through real price action ──────────────────
def resolve(df, journal, pos_of):
    slip   = SLIP_TICKS * TICK
    closed = []                               # newly-resolved trades, for notifications

    for idx, e in journal[journal["status"] == "OPEN"].iterrows():
        p = pos_of.get(str(e["entry_ts"]))
        if p is None:
            continue                          # entry bar aged out of the 60d window
        c, entry_fill = int(e["contracts"]), float(e["entry_fill"])
        stop, target  = float(e["stop"]), float(e["target"])
        comm = c * COMMISSION_RT
        outcome = None

        for k in range(1, MAX_HOLD_BARS + 1):
            j = p + k
            if j >= len(df):
                break                         # not enough future bars yet — stays OPEN
            r = df.iloc[j]
            if r["Low"] <= stop:
                exit_fill = stop - slip
                outcome = ("LOSS", exit_fill, k)
            elif r["High"] >= target:
                exit_fill = target
                outcome = ("WIN", exit_fill, k)
            elif k == MAX_HOLD_BARS:
                exit_fill = r["Close"] - slip
                pnl_chk   = (exit_fill - entry_fill) * c * POINT_VALUE - comm
                outcome = ("EXPIRED_WIN" if pnl_chk > 0 else "EXPIRED_LOSS", exit_fill, k)
            if outcome:
                status, xfill, bars = outcome
                pnl = round((xfill - entry_fill) * c * POINT_VALUE - comm, 2)
                journal.loc[idx, ["status", "exit_ts", "exit_price", "bars_held", "pnl"]] = \
                    [status, ts_str(r), round(xfill, 2), bars, pnl]
                closed.append({"entry_ts": e["entry_ts"], "status": status, "pnl": pnl})
                break
    return closed


# ── REPORT ───────────────────────────────────────────────────────────────────────
def bootstrap_forward(pnls, n_boot=5000):
    rng, n = random.Random(BOOT_SEED), len(pnls)
    totals = []
    for _ in range(n_boot):
        totals.append(sum(pnls[rng.randrange(n)] for _ in range(n)))
    totals.sort()
    prob = sum(1 for t in totals if t > 0) / n_boot * 100
    p5   = totals[int(n_boot * 0.05)]
    p95  = totals[int(n_boot * 0.95)]
    return prob, p5, p95


def report(journal):
    print("\n" + "=" * 70)
    print("FORWARD JOURNAL — REAL OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    closed = journal[~journal["status"].isin(["OPEN"])].copy()
    open_  = journal[journal["status"] == "OPEN"]

    print(f"  Total logged: {len(journal)}   "
          f"Closed: {len(closed)}   Still open: {len(open_)}")

    if len(closed):
        pnls    = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        gross_w = pnls[pnls > 0].sum()
        gross_l = pnls[pnls < 0].sum()
        wr  = (pnls > 0).mean() * 100
        pf  = abs(gross_w / gross_l) if gross_l != 0 else float("inf")
        print(f"\n  Closed performance (net of costs):")
        print(f"    Win rate: {wr:.1f}%   PF: {pf:.2f}   "
              f"Net PnL: ${pnls.sum():.2f}   Exp: ${pnls.mean():.2f}/trade")

        if len(closed) >= 10:
            prob, p5, p95 = bootstrap_forward(pnls.tolist())
            print(f"\n  Bootstrap on {len(closed)} FORWARD trades:")
            print(f"    P(profitable): {prob:.1f}%   "
                  f"5th–95th: ${p5:.0f} to ${p95:.0f}")
            verdict = ("✅ FUND-READY — forward edge confirmed." if prob >= 90 else
                       "🟡 Promising but keep paper-trading." if prob >= 75 else
                       "❌ Forward edge not holding — do not fund.")
            print(f"    {verdict}")
        else:
            print(f"\n  Need {10 - len(closed)} more closed trades before the "
                  f"fund/don't-fund bootstrap kicks in.")

    if len(open_):
        print(f"\n  Open trades being tracked:")
        for _, e in open_.iterrows():
            print(f"    {e['entry_ts']} | {e['session']} {e['zone']} "
                  f"conf {e['confidence']} | {int(e['contracts'])}c | "
                  f"entry {e['entry']} stop {e['stop']} tgt {e['target']}")

    print(f"\n  Journal: {JOURNAL}")


# ── PHONE NOTIFICATION (optional) ────────────────────────────────────────────────
def notify(msg):
    """Pushes to your phone IF a channel is configured via env vars / repo secrets.
    Telegram (TELEGRAM_TOKEN + TELEGRAM_CHAT) takes priority; otherwise a generic
    webhook (WEBHOOK_URL — Discord/Slack-style). Silent no-op if neither is set."""
    import json, urllib.request
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    hook = os.environ.get("WEBHOOK_URL")
    try:
        if tok and chat:
            url  = f"https://api.telegram.org/bot{tok}/sendMessage"
            body = json.dumps({"chat_id": chat, "text": msg}).encode()
        elif hook:
            url  = hook
            body = json.dumps({"content": msg, "text": msg}).encode()  # Discord & Slack
        else:
            return
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"(notify failed: {e})")


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, bias_map = load_data()
    df["ts"]     = df.index                  # preserve CT datetime before merge drops it
    df           = add_indicators(df, bias_map)
    df["RVOL"]   = (df["Volume"] / df["Volume"].rolling(RVOL_WINDOW).mean()).fillna(1.0)
    slow         = get_slow_conf()

    journal = load_journal()

    # Shared bar-timestamp → position map; last completed bar is the "fresh" one.
    pos_of   = {t.strftime("%Y-%m-%d %H:%M"): p for p, t in enumerate(df["ts"])}
    fresh_ts = ts_str(df.iloc[len(df) - 2])   # last completed (non-forming) bar

    closed = resolve(df, journal, pos_of)
    new    = scan(df, slow, journal, pos_of, fresh_ts)

    fresh = [s for s in new if s.pop("fresh", False)]   # strip flag before persisting
    _news = {}                                          # stamp concurrent macro news
    try:
        import json as _json
        with open(os.path.join(_HERE, "news_state.json")) as _f:
            _news = _json.load(_f)
    except Exception:
        pass
    for _s in new:
        _s["news_sent"]  = _news.get("avg_sentiment", "")
        _s["news_burst"] = _news.get("burst", "")
    if new:
        journal = pd.concat([journal, pd.DataFrame(new)], ignore_index=True)
    save_journal(journal)
    save_state(fresh_ts)                       # remember how far we scanned

    print(f"\nResolved {len(closed)} open trade(s) | Logged {len(new)} new signal(s)")
    for s in fresh:
        print(f"  ⚡ FRESH SIGNAL NOW: {s['entry_ts']} {s['session']} {s['zone']} "
              f"conf {s['confidence']} — {int(s['contracts'])}c, "
              f"entry {s['entry']}, stop {s['stop']}, target {s['target']}")
    report(journal)

    # ── Push a phone summary only when something actually happened ────────────────
    if new or closed:
        lines = ["📊 MNQ forward update"]
        for c in closed:
            lines.append(f"Closed {c['entry_ts']} → {c['status']} ${c['pnl']:+.0f}")
        for s in new:
            tag = "⚡FRESH " if s in fresh else ""
            lines.append(f"{tag}Signal {s['entry_ts']} {s['session']} {s['zone']} "
                         f"conf {s['confidence']} ({int(s['contracts'])}c)")
        notify("\n".join(lines))
