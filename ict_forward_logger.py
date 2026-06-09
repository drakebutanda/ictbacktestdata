"""
ICT FVG Forward Logger — ict_forward_logger.py
==============================================
Forward paper-trade journal for the validated ICT fair-value-gap strategy, now
across two uncorrelated books for genuine diversification:

  • ALL markets (MES, MNQ, Gold MGC, Crude MCL) → LONG *and* SHORT. Direction is
    gated by the VWAP trend filter (long above VWAP, short below), so it's
    regime-ADAPTIVE — shorts down-mornings, longs up-mornings — and never bets the
    equity up-drift continues. (Tested: equity shorts OOS PF 1.55; both-sides ~2×
    the sample + profit of long-only.)

Upgrades from the backtests:
  • MIN-GAP FILTER (0.25 ATR) — skip tiny, cost-dominated gaps. Lifted WR 57→62%,
    OOS PF 1.56→1.78, cut drawdown ~4×.
  • Gold/Crude added — OOS PF ~3.5 long, ~1.85 short, uncorrelated to equities,
    proving the edge is structural (not equity-beta) and adding real diversification.

ACCOUNTS — profits kept SEPARATE from origin capital:
  • Sizing = 3% risk/trade (quarter-Kelly) off the fixed ORIGIN ($10k), no compounding.
  • One position per market at a time (any direction); up to ~4 concurrent, but the
    two books are uncorrelated so the real risk is lower than 4× suggests.
  • Net losses draw origin down; gains above baseline accrue to a profit account.

Gap-proof scan + self-resolving, cloud-ready. Run: python3 ~/Desktop/ict_forward_logger.py
"""

import os, json, random, urllib.request
import pandas as pd

from backtest10_institutional import load, MAX_CONTRACTS, SLIP_TICKS
from backtest13_ict_orb import RR, RETRACE_WIN, MORN_START, MORN_END, MAX_HOLD

_HERE   = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(_HERE, "ict_forward_journal.csv")
STATE   = os.path.join(_HERE, ".ict_forward_state")
ORIGIN  = 10_000
RISK    = 0.03                    # 3%/trade (quarter-Kelly) off fixed ORIGIN
MIN_GAP_ATR = 0.25               # skip FVGs smaller than this × ATR (cost-noise filter)
SCAN_LOOKBACK_BARS = 576

# Liquid full-size data tickers, MICRO economics, and which sides each book trades.
INSTRUMENTS = {
    "MES (S&P 500)": dict(data="ES=F",  point=5.0,   tick=0.25, comm=1.0, sides=[1, -1]),
    "MNQ (Nasdaq)":  dict(data="NQ=F",  point=2.0,   tick=0.25, comm=1.0, sides=[1, -1]),
    "MGC (Gold)":    dict(data="GC=F",  point=10.0,  tick=0.10, comm=1.0, sides=[1, -1]),
    "MCL (Crude)":   dict(data="CL=F",  point=100.0, tick=0.01, comm=1.0, sides=[1, -1]),
}

COLUMNS = ["entry_ts", "instrument", "side", "status", "contracts", "entry", "entry_fill",
           "stop", "target", "exit_ts", "exit_price", "bars_held", "pnl"]


def ts_of(row):
    m = int(row["btime"]); return f"{row['Date']} {m // 60:02d}:{m % 60:02d}"


def load_journal():
    if os.path.exists(JOURNAL):
        return pd.read_csv(JOURNAL, dtype={"entry_ts": str, "exit_ts": str,
                                           "instrument": str, "side": str})
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


# ── SIGNAL: bullish (long) or bearish (short) FVG, retrace entry, min-gap, kill-zone ──
def sig_fvg_dir(df, side):
    ents, active = [], []
    for i in range(2, len(df)):
        r = df.iloc[i]; bt = int(r["btime"]); atr = r["ATR"]
        if side == 1:
            a, b = df.iloc[i-2]["High"], r["Low"]            # gap_lo, gap_hi (gapped up)
        else:
            b, a = df.iloc[i-2]["Low"], r["High"]            # gap_hi, gap_lo (gapped down)
        if b > a and (b - a) >= MIN_GAP_ATR * atr:
            active.append((a, b, i))                         # (gap_lo, gap_hi, born)
        trend_ok = (r["Close"] > r["VWAP"]) if side == 1 else (r["Close"] < r["VWAP"])
        if not (MORN_START <= bt <= MORN_END) or not trend_ok:
            active = [g for g in active if i - g[2] <= RETRACE_WIN]; continue
        for g in list(active):
            gap_lo, gap_hi, born = g
            if i - born > RETRACE_WIN:
                active.remove(g); continue
            if side == 1 and r["Low"] <= gap_hi:             # retrace down into bullish gap
                ents.append(dict(bar=i, dir=1, entry=gap_hi, stop=gap_lo,
                                 target=gap_hi + RR*(gap_hi-gap_lo))); active.remove(g); break
            if side == -1 and r["High"] >= gap_lo:           # retrace up into bearish gap
                ents.append(dict(bar=i, dir=-1, entry=gap_lo, stop=gap_hi,
                                 target=gap_lo - RR*(gap_hi-gap_lo))); active.remove(g); break
    return ents


# ── SCAN (gap-proof, one position per market across both directions) ────────────
def scan(mkts, journal, last_scanned):
    known = set(zip(journal["instrument"].astype(str), journal["entry_ts"].astype(str)))
    new, latest = [], last_scanned or ""
    for name, (df, inst) in mkts.items():
        pt, slip = inst["point"], SLIP_TICKS * inst["tick"]
        if len(df) >= 2:
            latest = max(latest, ts_of(df.iloc[len(df) - 2]))
        cutoff = last_scanned if last_scanned else ts_of(df.iloc[len(df) - 2])

        ents = []
        for sd in inst["sides"]:
            ents += sig_fvg_dir(df, sd)
        ents.sort(key=lambda e: e["bar"])

        free_at = -10**9
        for e in ents:
            b, d = e["bar"], e["dir"]
            if b >= len(df) - 1:  continue
            ts = ts_of(df.iloc[b])
            if ts <= cutoff or (name, ts) in known or b < free_at:  continue
            known.add((name, ts))
            c = size(abs(e["entry"] - e["stop"]), pt)
            efill = round(e["entry"] + (slip if d == 1 else -slip), 2)
            new.append(dict(entry_ts=ts, instrument=name, side=("LONG" if d == 1 else "SHORT"),
                            status="OPEN", contracts=c, entry=round(e["entry"], 2),
                            entry_fill=efill, stop=round(e["stop"], 2), target=round(e["target"], 2),
                            exit_ts="", exit_price="", bars_held="", pnl=""))
            ex = b + MAX_HOLD                                # when does this trade free the market?
            for k in range(1, MAX_HOLD + 1):
                j = b + k
                if j >= len(df):  break
                rr = df.iloc[j]
                hit = (rr["Low"] <= e["stop"] or rr["High"] >= e["target"]) if d == 1 \
                    else (rr["High"] >= e["stop"] or rr["Low"] <= e["target"])
                if hit:  ex = j; break
            free_at = ex
    return new, latest


# ── RESOLVE (direction-aware; entry bar is stop-only) ───────────────────────────
def resolve(mkts, journal, posmaps):
    closed = []
    for idx, e in journal[journal["status"] == "OPEN"].iterrows():
        name = str(e["instrument"])
        if name not in mkts:  continue
        df, inst = mkts[name]
        pt, slip, comm = inst["point"], SLIP_TICKS * inst["tick"], inst["comm"]
        p = posmaps[name].get(str(e["entry_ts"]))
        if p is None:  continue
        d = 1 if str(e["side"]) == "LONG" else -1
        ef, stop, target, c = float(e["entry_fill"]), float(e["stop"]), float(e["target"]), int(e["contracts"])
        out = None
        for k in range(0, MAX_HOLD + 1):
            j = p + k
            if j >= len(df):  break
            r = df.iloc[j]
            if d == 1:
                if r["Low"] <= stop:               out = ("LOSS", stop - slip, k)
                elif k >= 1 and r["High"] >= target: out = ("WIN", target, k)
            else:
                if r["High"] >= stop:              out = ("LOSS", stop + slip, k)
                elif k >= 1 and r["Low"] <= target: out = ("WIN", target, k)
            if out is None and k == MAX_HOLD:
                xf = r["Close"] - (slip if d == 1 else -slip)
                out = ("EXP_WIN" if (xf - ef) * d * c * pt - c * comm > 0 else "EXP_LOSS", xf, k)
            if out:  break
        if out:
            status, xf, k = out
            pnl = round((xf - ef) * d * c * pt - c * comm, 2)
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
    except Exception as ex:
        print(f"(notify failed: {ex})")


def report(journal):
    print("\n" + "=" * 66)
    print("ICT FVG FORWARD JOURNAL — equities (long) + gold/crude (long & short)")
    print("=" * 66)
    closed = journal[journal["status"] != "OPEN"].copy()
    openn  = journal[journal["status"] == "OPEN"]
    print(f"  Logged {len(journal)} | closed {len(closed)} | open {len(openn)}")

    if len(closed):
        pnls = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        cum  = pnls.sum(); gw, gl = pnls[pnls > 0].sum(), pnls[pnls < 0].sum()
        wr = (pnls > 0).mean() * 100; pf = abs(gw / gl) if gl else float("inf")
        print(f"\n  Closed: WR {wr:.1f}% | PF {pf:.2f} | net ${cum:.2f}")
        print(f"\n  ── ACCOUNTS ──")
        print(f"  Origin account:  ${ORIGIN + min(0.0, cum):,.2f}   (base ${ORIGIN:,})")
        print(f"  Profit account:  ${max(0.0, cum):,.2f}")
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
            print(f"    {e['entry_ts']} {e['instrument']:<14} {str(e['side']):<5} "
                  f"{int(e['contracts'])}c entry {e['entry']} stop {e['stop']} tgt {e['target']}")
    print(f"\n  Journal: {JOURNAL}")


if __name__ == "__main__":
    mkts = {}
    for name, inst in INSTRUMENTS.items():
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
            lines.append(f"Signal {s['instrument']} {s['side']} {s['entry_ts']} {int(s['contracts'])}c")
        notify("\n".join(lines))
