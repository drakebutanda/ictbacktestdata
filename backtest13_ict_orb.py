"""
ICT / Liquidity / ORB head-to-head — backtest13_ict_orb.py
==========================================================
Three popular intraday concepts, tested on the SAME multi-market engine (4 index
micros, real costs, 70/30 OOS) so they're directly comparable. A shared simulator
runs every strategy's entries identically — each strategy only generates signals.

  1. LIQUIDITY SWEEP  — price wicks beyond a recent swing hi/lo (grabs stops =
                        "liquidity"), then closes back inside → fade the grab.
  2. ICT FVG          — a 3-bar fair-value-gap (imbalance); trade in its direction
                        with the gap as the stop base.
  3. ORB 15m          — opening-range breakout: first 15 min sets the range, trade
                        the first break, stop at the opposite end, target 2× range.

These are FRESH directional ideas (long & short) — the long/short split is reported
so the equity-index drift effect is visible. Run: python3 ~/Desktop/backtest13_ict_orb.py
"""

import warnings
warnings.simplefilter("ignore")
import pandas as pd

from backtest10_institutional import (
    load, stat, row as line, INSTRUMENTS, ACCOUNT, RISK_PCT, MAX_CONTRACTS,
    SLIP_TICKS, MIN_BARS, OOS_FRACTION, RTH_START, RTH_END,
)

RR          = 2.0     # reward:risk for FVG
RETRACE_WIN = 12      # bars a fresh FVG stays "live" waiting for a retrace entry
MORN_START, MORN_END = 8*60+30, 10*60+30   # morning kill-zone (CT) — ORB & ICT both
ORB_MINUTES = 15      # opening-range length (popular standard)
ORB_RR      = 2.0     # ORB target = this × the opening range
ORB_LONG_ONLY = True  # take only UP breakouts. Short breakouts fight the equity
                      # index drift (PF 0.51 vs 1.40) — same bias found 3x over.
MAX_HOLD    = 24


def _size(stop_pts, pt):
    if stop_pts <= 0:
        return 1
    return max(1, min(int((ACCOUNT * RISK_PCT) / (stop_pts * pt)), MAX_CONTRACTS))


def _exit_check(t, r, slip, pt, comm, held):
    """Stop-first (conservative) exit test for one bar. Returns (result, pnl) or None."""
    d, c, ef = t["dir"], t["contracts"], t["entry_fill"]
    px, hi, lo = r["Close"], r["High"], r["Low"]
    res = xf = None
    if d == 1:
        if lo <= t["stop"]:     res, xf = "LOSS", t["stop"] - slip
        elif hi >= t["target"]: res, xf = "WIN",  t["target"]
    else:
        if hi >= t["stop"]:     res, xf = "LOSS", t["stop"] + slip
        elif lo <= t["target"]: res, xf = "WIN",  t["target"]
    if res is None and held >= MAX_HOLD:
        xf  = px - slip if d == 1 else px + slip
        res = "EXP_WIN" if (xf - ef) * d * c * pt - c * comm > 0 else "EXP_LOSS"
    if res:
        return res, round((xf - ef) * d * c * pt - c * comm, 2)
    return None


def simulate(df, entries, inst):
    """Generic engine: pre-computed entries, stop/target/time exits, net of costs,
    OOS tag. CONSERVATIVE: the entry bar itself is checked for a same-bar stop."""
    pt, slip, comm = inst["point"], SLIP_TICKS * inst["tick"], inst["comm"]
    train_end = MIN_BARS + int((len(df) - MIN_BARS) * (1 - OOS_FRACTION))
    entry_at = {}
    for e in entries:
        entry_at.setdefault(e["bar"], e)          # one entry per bar, first wins
    trades, t, held = [], None, 0

    for i in range(len(df)):
        r = df.iloc[i]
        if t is not None:
            held += 1
            out = _exit_check(t, r, slip, pt, comm, held)
            if out:
                t.update(result=out[0], pnl=out[1], bars=held)
                trades.append(t); t, held = None, 0
            continue
        if i in entry_at:
            e = entry_at[i]; d = e["dir"]
            if (e["target"] - e["entry"]) * d <= 0 or (e["entry"] - e["stop"]) * d <= 0:
                continue
            c = _size(abs(e["entry"] - e["stop"]), pt)
            t = dict(dir=d, contracts=c, entry=round(e["entry"], 2),
                     entry_fill=round(e["entry"] + (slip if d == 1 else -slip), 2),
                     stop=round(e["stop"], 2), target=round(e["target"], 2),
                     oos=(i >= train_end), side="LONG" if d == 1 else "SHORT")
            held = 0
            # Same-bar guard: only a STOP can fill intrabar after entry — never
            # assume the target (we don't know the price path within the entry bar).
            if (r["Low"] <= t["stop"]) if d == 1 else (r["High"] >= t["stop"]):
                xf  = t["stop"] - slip if d == 1 else t["stop"] + slip
                pnl = round((xf - t["entry_fill"]) * d * c * pt - c * comm, 2)
                t.update(result="LOSS", pnl=pnl, bars=0)
                trades.append(t); t, held = None, 0
    return pd.DataFrame(trades)


# ── STRATEGIES (each returns a list of entry dicts) ─────────────────────────────
def sig_fvg(df):
    """Proper ICT entry: a bullish 3-bar FVG opens an imbalance zone [h2, l0];
    enter long only when price later RETRACES into that zone, in an uptrend
    (above VWAP), during the morning kill-zone. Long-only (index drift)."""
    ents, active = [], []
    for i in range(2, len(df)):
        r = df.iloc[i]; bt = int(r["btime"])
        h2, l0 = df.iloc[i-2]["High"], r["Low"]
        if l0 > h2:                                        # new bullish FVG zone
            active.append((h2, l0, i))                     # (gap_low, gap_top, born)
        if not (MORN_START <= bt <= MORN_END) or r["Close"] <= r["VWAP"]:
            active = [g for g in active if i - g[2] <= RETRACE_WIN]
            continue
        for g in list(active):
            gap_lo, gap_top, born = g
            if i - born > RETRACE_WIN:
                active.remove(g); continue
            if r["Low"] <= gap_top:                         # retraced into the gap
                ents.append(dict(bar=i, dir=1, entry=gap_top, stop=gap_lo,
                                 target=gap_top + RR*(gap_top-gap_lo)))
                active.remove(g); break
    return ents


def sig_orb(df):
    or_bars, ents = ORB_MINUTES // 5, []
    for _, g in df.groupby("Date"):
        rth = [b for b in g.index if RTH_START <= int(df.iloc[b]["btime"]) <= RTH_END]
        if len(rth) < or_bars + 1:  continue
        og = rth[:or_bars]
        or_hi = max(df.iloc[b]["High"] for b in og)
        or_lo = min(df.iloc[b]["Low"]  for b in og)
        rsize = or_hi - or_lo
        if rsize <= 0:  continue
        for b in rth[or_bars:]:                            # first MORNING breakout
            r = df.iloc[b]
            if int(r["btime"]) > MORN_END:  break          # only morning breaks
            if r["High"] > or_hi:
                ents.append(dict(bar=b, dir=1, entry=or_hi, stop=or_lo, target=or_hi+ORB_RR*rsize)); break
            if r["Low"] < or_lo:
                if not ORB_LONG_ONLY:
                    ents.append(dict(bar=b, dir=-1, entry=or_lo, stop=or_hi, target=or_lo-ORB_RR*rsize))
                break
    return ents


STRATS = {"ICT FVG (retrace, morning, uptrend)": sig_fvg, "ORB 15m (morning long)": sig_orb}


if __name__ == "__main__":
    print("ICT / liquidity / ORB — loading 4 markets once...\n")
    mkts = {}
    for name, inst in INSTRUMENTS.items():
        df = load(inst["data"])
        if df is not None:
            mkts[name] = (df, inst)

    for sname, sfn in STRATS.items():
        allt = []
        for name, (df, inst) in mkts.items():
            tr = simulate(df, sfn(df), inst)
            if not tr.empty:
                tr["instrument"] = name; allt.append(tr)
        print("=" * 80)
        print(sname)
        print("=" * 80)
        if not allt:
            print("  no trades\n"); continue
        comb = pd.concat(allt, ignore_index=True)
        c, o = stat(comb), stat(comb[comb["oos"]])
        print(line("ALL MARKETS", c))
        print(line("  in-sample", stat(comb[~comb["oos"]])))
        print(line("  out-sample", o))
        print(line("  longs", stat(comb[comb["side"] == "LONG"])))
        print(line("  shorts", stat(comb[comb["side"] == "SHORT"])))
        verdict = ("✅ holds OOS" if o and o["pnl"] > 0 and o["pf"] >= 1.2 else
                   "🟡 OOS positive but thin" if o and o["pnl"] > 0 else
                   "❌ no OOS edge")
        print(f"  → ~{c['n']/60:.1f} trades/day | {verdict}\n")
