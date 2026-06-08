"""
ICT FVG Forward Logger — ict_forward_logger.py
==============================================
Forward paper-trade journal for the validated ICT fair-value-gap strategy
(retrace into the gap, uptrend, morning kill-zone, long-only) across the 4 index
micros. Same gap-proof scan + self-resolving design as the main forward logger,
so it builds an un-overfittable live record on its own.

ACCOUNTS — profits kept SEPARATE from origin capital:
  • Position sizing always uses the fixed ORIGIN ($10k) — no compounding.
  • Net losses draw the origin down; once recovered it sits at baseline.
  • Everything above baseline accrues to a separate PROFIT account.

Reuses the validated signal (backtest13.sig_fvg) and data layer (backtest10.load).
Cloud-ready (script-relative paths). Run: python3 ~/Desktop/ict_forward_logger.py
"""

import os, json, random, urllib.request
import pandas as pd

from backtest10_institutional import load, INSTRUMENTS, MAX_CONTRACTS, SLIP_TICKS
from backtest13_ict_orb import sig_fvg, MAX_HOLD

_HERE   = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(_HERE, "ict_forward_journal.csv")
STATE   = os.path.join(_HERE, ".ict_forward_state")
ORIGIN  = 10_000
RISK    = 0.03                    # SINGLE shared account: 3%/trade — quarter-Kelly
                                  # (full Kelly ≈ 19%), the risk-adjusted optimum after
                                  # the MNQ/MES correlation + estimation-error haircut.
                                  # One position per market (max 2 concurrent now).
MARKETS = {"MES (S&P 500)", "MNQ (Nasdaq)"}   # focus: 2 most-liquid index micros only
SCAN_LOOKBACK_BARS = 576          # ~2 days; first run only (no historical backfill)

COLUMNS = ["entry_ts", "instrument", "status", "contracts", "entry", "entry_fill",
           "stop", "target", "exit_ts", "exit_price", "bars_held", "pnl"]


def ts_of(row):
    m = int(row["btime"]); return f"{row['Date']} {m // 60:02d}:{m % 60:02d}"


def load_journal():
    if os.path.exists(JOURNAL):
        return pd.read_csv(JOURNAL, dtype={"entry_ts": str, "exit_ts": str, "instrument": str})
    return pd.DataFrame(columns=COLUMNS)


def load_state():
    return open(STATE).read().strip() if os.path.exists(STATE) else None


def save_state(s):
    with open(STATE, "w") as f:
        f.write(s)


def size(stop_pts, pt):
    if stop_pts <= 0:
        return 1
    return max(1, min(int((ORIGIN * RISK) / (stop_pts * pt)), MAX_CONTRACTS))


# ── SCAN (gap-proof, no backfill) ───────────────────────────────────────────────
def scan(mkts, journal, last_scanned):
    known = set(zip(journal["instrument"].astype(str), journal["entry_ts"].astype(str)))
    new, latest = [], last_scanned or ""
    for name, (df, inst) in mkts.items():
        pt, slip = inst["point"], SLIP_TICKS * inst["tick"]
        if len(df) >= 2:
            latest = max(latest, ts_of(df.iloc[len(df) - 2]))
        if last_scanned:
            cutoff = last_scanned
        else:                                          # first run: start CLEAN — a pure
            cutoff = ts_of(df.iloc[len(df) - 2])       # forward record, zero backfill
        free_at = -10**9                                # one position per market: busy
        for e in sig_fvg(df):                           # until its trade actually EXITS
            b = e["bar"]
            if b >= len(df) - 1:  continue               # skip the forming bar
            ts = ts_of(df.iloc[b])
            if ts <= cutoff or (name, ts) in known:  continue
            if b < free_at:  continue                    # still holding a position here
            known.add((name, ts))
            c = size(abs(e["entry"] - e["stop"]), pt)
            new.append(dict(entry_ts=ts, instrument=name, status="OPEN", contracts=c,
                            entry=round(e["entry"], 2), entry_fill=round(e["entry"] + slip, 2),
                            stop=round(e["stop"], 2), target=round(e["target"], 2),
                            exit_ts="", exit_price="", bars_held="", pnl=""))
            ex = b + MAX_HOLD                            # when does this trade free the market?
            for k in range(1, MAX_HOLD + 1):
                j = b + k
                if j >= len(df):  break                  # runs into the future → still open
                rr = df.iloc[j]
                if rr["Low"] <= e["stop"] or rr["High"] >= e["target"]:
                    ex = j; break
            free_at = ex
    return new, latest


# ── RESOLVE (conservative: entry bar is stop-only) ──────────────────────────────
def resolve(mkts, journal, posmaps):
    closed = []
    for idx, e in journal[journal["status"] == "OPEN"].iterrows():
        name = str(e["instrument"])
        if name not in mkts:  continue
        df, inst = mkts[name]
        pt, slip, comm = inst["point"], SLIP_TICKS * inst["tick"], inst["comm"]
        p = posmaps[name].get(str(e["entry_ts"]))
        if p is None:  continue
        ef, stop, target, c = float(e["entry_fill"]), float(e["stop"]), float(e["target"]), int(e["contracts"])
        out = None
        for k in range(0, MAX_HOLD + 1):
            j = p + k
            if j >= len(df):  break                     # not enough bars yet → stays OPEN
            r = df.iloc[j]
            if r["Low"] <= stop:                        # stop (valid even on entry bar)
                out = ("LOSS", stop - slip, k)
            elif k >= 1 and r["High"] >= target:        # target only AFTER the entry bar
                out = ("WIN", target, k)
            elif k == MAX_HOLD:
                xf = r["Close"] - slip
                out = ("EXP_WIN" if (xf - ef) * c * pt - c * comm > 0 else "EXP_LOSS", xf, k)
            if out:  break
        if out:
            status, xf, k = out
            pnl = round((xf - ef) * c * pt - c * comm, 2)
            journal.loc[idx, ["status", "exit_ts", "exit_price", "bars_held", "pnl"]] = \
                [status, ts_of(df.iloc[p + k]), round(xf, 2), k, pnl]
            closed.append(dict(instrument=name, entry_ts=e["entry_ts"], status=status, pnl=pnl))
    return closed


def notify(msg):
    tok, chat, hook = (os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT"),
                       os.environ.get("WEBHOOK_URL"))
    try:
        if tok and chat:
            url, body = f"https://api.telegram.org/bot{tok}/sendMessage", json.dumps({"chat_id": chat, "text": msg}).encode()
        elif hook:
            url, body = hook, json.dumps({"content": msg, "text": msg}).encode()
        else:
            return
        urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=10)
    except Exception as e:
        print(f"(notify failed: {e})")


# ── REPORT ───────────────────────────────────────────────────────────────────────
def report(journal):
    print("\n" + "=" * 66)
    print("ICT FVG FORWARD JOURNAL")
    print("=" * 66)
    closed = journal[journal["status"] != "OPEN"].copy()
    openn  = journal[journal["status"] == "OPEN"]
    print(f"  Logged {len(journal)} | closed {len(closed)} | open {len(openn)}")

    if len(closed):
        pnls = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        cum  = pnls.sum()
        gw, gl = pnls[pnls > 0].sum(), pnls[pnls < 0].sum()
        wr = (pnls > 0).mean() * 100
        pf = abs(gw / gl) if gl else float("inf")
        profit = max(0.0, cum)
        origin = ORIGIN + min(0.0, cum)
        print(f"\n  Closed: WR {wr:.1f}% | PF {pf:.2f} | net ${cum:.2f}")
        print(f"\n  ── ACCOUNTS ──")
        print(f"  Origin account:  ${origin:,.2f}   (base ${ORIGIN:,})")
        print(f"  Profit account:  ${profit:,.2f}")
        if len(closed) >= 10:
            rng, p = random.Random(42), pnls.tolist()
            tot = sorted(sum(p[rng.randrange(len(p))] for _ in p) for _ in range(4000))
            prob = sum(1 for t in tot if t > 0) / len(tot) * 100
            print(f"\n  Bootstrap ({len(closed)} fwd trades): P(profit) {prob:.0f}%  "
                  f"[{'✅ fund-ready' if prob >= 90 else '🟡 keep paper' if prob >= 75 else '❌ stop'}]")
        else:
            print(f"\n  Need {10 - len(closed)} more closed trades for the verdict.")
    if len(openn):
        print(f"\n  Open ({len(openn)}):")
        for _, e in openn.iterrows():
            print(f"    {e['entry_ts']} {e['instrument']:<14} {int(e['contracts'])}c "
                  f"entry {e['entry']} stop {e['stop']} tgt {e['target']}")
    print(f"\n  Journal: {JOURNAL}")


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mkts = {}
    for name, inst in INSTRUMENTS.items():
        if name not in MARKETS:  continue          # MNQ + MES only
        df = load(inst["data"])
        if df is not None:
            mkts[name] = (df, inst)
    posmaps = {name: {ts_of(df.iloc[i]): i for i in range(len(df))} for name, (df, _) in mkts.items()}

    journal = load_journal()
    closed  = resolve(mkts, journal, posmaps)
    new, latest = scan(mkts, journal, load_state())
    if new:
        journal = pd.concat([journal, pd.DataFrame(new)], ignore_index=True)
    journal.to_csv(JOURNAL, index=False)
    if latest:
        save_state(latest)

    print(f"Resolved {len(closed)} | logged {len(new)} new signal(s)")
    report(journal)

    if new or closed:
        lines = ["📈 ICT FVG update"]
        for c in closed:
            lines.append(f"Closed {c['instrument']} {c['status']} ${c['pnl']:+.0f}")
        for s in new:
            lines.append(f"Signal {s['instrument']} {s['entry_ts']} {int(s['contracts'])}c")
        notify("\n".join(lines))
