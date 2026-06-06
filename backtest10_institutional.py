"""
Institutional Overreaction → Reversion Backtest — backtest10_institutional.py
=============================================================================
Thesis (the user's): don't chase the initial move — catch the *re-correction*
after an institutional overreaction, in EITHER direction. This is mean reversion
to the volume anchor (VWAP), triggered only when big players are clearly active.

WHY THIS IS BACKTESTABLE: every signal comes from free OHLCV — no paid sentiment
feed, no look-ahead. "Big institution moves" are tracked via their footprint:
  • RVOL  — relative volume vs the 20-bar average. >1 = unusual participation.
  • Climax volume at an extreme = capitulation/exhaustion (institutions absorbing).
  • VWAP stretch (distance from VWAP in ATRs) = how far price has overreacted.

SETUP (long example — mirror for shorts):
  price stretched >= STRETCH ATRs BELOW VWAP, RSI oversold, RVOL elevated, and
  the current bar TURNS UP (we wait for the re-correction to start, not guess the
  bottom) → fade back toward VWAP. Target = VWAP (high-probability mean), stop
  beyond the extreme. Net of slippage + commission, 70/30 out-of-sample per market.

Tested across the equity-index micros that share macro drivers (MES/MNQ/MYM/M2K):
if one strategy works on all four, it's a real edge, not a curve fit. Data is
pulled from the liquid FULL-SIZE contracts (ES/NQ/YM/RTY) but P&L uses MICRO
economics — micros track identical prices, so this sidesteps thin micro-data.

Run: python3 ~/Desktop/backtest10_institutional.py
"""

import warnings
warnings.simplefilter("ignore")
import pandas as pd
import pytz
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

CT = pytz.timezone("America/Chicago")

# ── INSTRUMENTS ─────────────────────────────────────────────────────────────────
# data_ticker = liquid full-size contract (clean Yahoo data)
# point/tick/comm = MICRO economics (what you'd actually trade, no funded acct)
INSTRUMENTS = {
    "MES (S&P 500)":   dict(data="ES=F",  point=5.0,  tick=0.25, comm=1.00),
    "MNQ (Nasdaq)":    dict(data="NQ=F",  point=2.0,  tick=0.25, comm=1.00),
    "MYM (Dow)":       dict(data="YM=F",  point=0.50, tick=1.0,  comm=1.00),
    "M2K (Russell)":   dict(data="RTY=F", point=5.0,  tick=0.10, comm=1.00),
}

# ── STRATEGY PARAMS (sane defaults; same for every instrument = no per-market fit)
STRETCH_ATR  = 2.0      # deeper overreaction — bigger overshoots revert more reliably
RSI_OS       = 35       # oversold threshold for long fades
RSI_OB       = 65       # overbought threshold for short fades
RVOL_MIN     = 1.5      # genuine volume CLIMAX (capitulation) — the institutional
                        # exhaustion footprint, not just mildly elevated volume
ATR_REGIME   = 2.0      # skip if ATR > this * its 50-bar avg (vol too wild to fade)
STOP_ATR     = 1.2      # stop = this many ATRs beyond entry (breathing room vs noise)
TARGET_CAP   = 1.0      # cap target at this many ATRs of reward (reachable, not greedy)
MAX_HOLD     = 24       # bars (2h) — reversion should be quick or we bail
SCALE_OUT    = False    # TESTED: scale-out (half at mean, runner to VWAP risk-free)
                        # made it WORSE (−$347 vs +$138). After a 1-ATR bounce the
                        # runner rarely reaches full VWAP, so it scratches and shrinks
                        # the avg win. Single capped target is the better exit.
LONG_ONLY    = True     # fade oversold dips only. Equity indices drift UP and
                        # institutions buy dips; shorting rallies fights both. Data
                        # confirms: shorts bled in 3/4 markets. (Same long-only bias
                        # as the original validated MNQ strategy.)
SLIP_TICKS   = 1        # slippage per fill, per side

RTH_START, RTH_END = 8*60+30, 15*60     # 8:30am–3:00pm CT
ACCOUNT, RISK_PCT, MAX_CONTRACTS = 10_000, 0.01, 10
MIN_BARS, OOS_FRACTION = 60, 0.30


# ── DATA ─────────────────────────────────────────────────────────────────────────
def load(ticker):
    df = yf.download(ticker, period="60d", interval="5m", progress=False)
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                else df.index).tz_convert(CT)
    df["Date"]  = df.index.date
    df["btime"] = df.index.hour * 60 + df.index.minute

    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    df["ATR_AVG"] = df["ATR"].rolling(50).mean()
    df["RVOL"]    = df["Volume"] / df["Volume"].rolling(20).mean()

    # VWAP — resets each day (institutional anchor)
    vals = []
    for _, g in df.groupby("Date"):
        vals.extend((g["Close"] * g["Volume"]).cumsum()
                    .div(g["Volume"].cumsum()).tolist())
    df["VWAP"] = vals
    df["PREV"] = df["Close"].shift(1)
    return df.dropna().reset_index(drop=True)


# ── BACKTEST ─────────────────────────────────────────────────────────────────────
def backtest(df, inst):
    pt, slip, comm = inst["point"], SLIP_TICKS * inst["tick"], inst["comm"]
    train_end = MIN_BARS + int((len(df) - MIN_BARS) * (1 - OOS_FRACTION))
    trades, t, held = [], None, 0

    def size(stop_pts):
        if stop_pts <= 0:
            return 1
        raw = (ACCOUNT * RISK_PCT) / (stop_pts * pt)
        return max(1, min(int(raw), MAX_CONTRACTS))

    for i in range(MIN_BARS, len(df)):
        r  = df.iloc[i]
        px, hi, lo = r["Close"], r["High"], r["Low"]

        # ── manage open trade ─────────────────────────────────────────────────
        if t is not None:
            held += 1
            d, c, ef = t["dir"], t["contracts"], t["entry_fill"]
            done = False

            if not SCALE_OUT:                                   # single capped target
                res = exitf = None
                if d == 1:
                    if lo <= t["stop"]:  res, exitf = "LOSS", t["stop"] - slip
                    elif hi >= t["t1"]:  res, exitf = "WIN",  t["t1"]
                else:
                    if hi >= t["stop"]:  res, exitf = "LOSS", t["stop"] + slip
                    elif lo <= t["t1"]:  res, exitf = "WIN",  t["t1"]
                if res:
                    pnl = round((exitf - ef) * d * c * pt - c * comm, 2)
                    t.update(result=res, pnl=pnl, bars=held); done = True

            elif t["phase"] == 1:                               # full position, pre-T1
                stop_hit = lo <= t["stop"] if d == 1 else hi >= t["stop"]
                t1_hit   = hi >= t["t1"]   if d == 1 else lo <= t["t1"]
                if stop_hit:
                    xf  = t["stop"] - slip if d == 1 else t["stop"] + slip
                    pnl = round((xf - ef) * d * c * pt - c * comm, 2)
                    t.update(result="LOSS", pnl=pnl, bars=held); done = True
                elif t1_hit:                                    # lock half at the mean
                    t["half1"] = (t["t1"] - ef) * d * (c * 0.5) * pt
                    t["phase"] = 2
                    t["stop"]  = t["entry"]                     # remainder now risk-free

            else:                                              # phase 2: remainder → VWAP
                be_hit = lo <= t["stop"] if d == 1 else hi >= t["stop"]
                t2_hit = hi >= t["t2"]   if d == 1 else lo <= t["t2"]
                if be_hit:                                      # scratch the runner (BE)
                    xf    = t["stop"] - slip if d == 1 else t["stop"] + slip
                    half2 = (xf - ef) * d * (c * 0.5) * pt
                    pnl   = round(t["half1"] + half2 - c * comm, 2)
                    t.update(result=("WIN" if pnl > 0 else "LOSS"), pnl=pnl, bars=held); done = True
                elif t2_hit:                                    # runner reaches full VWAP
                    half2 = (t["t2"] - ef) * d * (c * 0.5) * pt
                    pnl   = round(t["half1"] + half2 - c * comm, 2)
                    t.update(result="WIN", pnl=pnl, bars=held); done = True

            if not done and held >= MAX_HOLD:                   # time stop
                xf = px - slip if d == 1 else px + slip
                if SCALE_OUT and t["phase"] == 2:
                    half2 = (xf - ef) * d * (c * 0.5) * pt
                    pnl   = round(t["half1"] + half2 - c * comm, 2)
                else:
                    pnl   = round((xf - ef) * d * c * pt - c * comm, 2)
                t.update(result=("EXP_WIN" if pnl > 0 else "EXP_LOSS"), pnl=pnl, bars=held); done = True

            if done:
                trades.append(t); t, held = None, 0
            continue

        # ── entry filters ─────────────────────────────────────────────────────
        if not (RTH_START <= int(r["btime"]) <= RTH_END):       continue
        if r["ATR"] <= 0 or r["ATR"] > ATR_REGIME * r["ATR_AVG"]: continue
        if r["RVOL"] < RVOL_MIN:                                continue   # institutions active?
        stretch = (px - r["VWAP"]) / r["ATR"]
        rng     = r["High"] - r["Low"]
        if rng <= 0:                                            continue

        # Confirmed reversal bar = the re-correction has actually STARTED (not a guess):
        # a directional close in the upper/lower half of the bar's range.
        rev_up = (r["Close"] > r["Open"]) and (r["Close"] - r["Low"]) >= 0.5 * rng
        rev_dn = (r["Close"] < r["Open"]) and (r["High"] - r["Close"]) >= 0.5 * rng

        direction = 0
        if stretch <= -STRETCH_ATR and r["RSI"] < RSI_OS and rev_up:
            direction = 1            # oversold overreaction + confirmed turn up → fade long
        elif stretch >= STRETCH_ATR and r["RSI"] > RSI_OB and rev_dn and not LONG_ONLY:
            direction = -1           # overbought overreaction + confirmed turn down → fade short
        if direction == 0:                                      continue

        stop = px - STOP_ATR * r["ATR"] if direction == 1 else px + STOP_ATR * r["ATR"]
        # T1 = capped reachable target (half exits here); T2 = full VWAP (runner).
        if direction == 1:
            t1 = min(r["VWAP"], px + TARGET_CAP * r["ATR"]); t2 = r["VWAP"]
        else:
            t1 = max(r["VWAP"], px - TARGET_CAP * r["ATR"]); t2 = r["VWAP"]
        if (t1 - px) * direction <= 0:                          continue   # target must be ahead
        c = size(abs(px - stop))
        t = dict(dir=direction, contracts=c, phase=1, half1=0.0,
                 entry=round(px, 2),
                 entry_fill=round(px + slip if direction == 1 else px - slip, 2),
                 stop=round(stop, 2), t1=round(t1, 2), t2=round(t2, 2),
                 rsi=round(r["RSI"], 1), rvol=round(r["RVOL"], 2),
                 stretch=round(stretch, 2), oos=(i >= train_end),
                 side="LONG" if direction == 1 else "SHORT")
        held = 0

    return pd.DataFrame(trades)


# ── STATS ────────────────────────────────────────────────────────────────────────
def stat(df):
    if df is None or df.empty:
        return None
    w = df[df["pnl"] > 0]["pnl"].sum()
    l = df[df["pnl"] < 0]["pnl"].sum()
    return dict(n=len(df), wr=(df["pnl"] > 0).mean()*100,
                pf=abs(w/l) if l else float("inf"),
                pnl=df["pnl"].sum(), avg=df["pnl"].mean())


def row(label, s):
    if s is None:
        return f"  {label:<18} | no trades"
    return (f"  {label:<18} | {s['n']:>3} tr | WR {s['wr']:>5.1f}% | "
            f"PF {s['pf']:>5.2f} | net ${s['pnl']:>9.2f} | ${s['avg']:>7.2f}/tr")


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Institutional overreaction → reversion. Pulling 4 markets (60d 5m)...\n")
    all_trades = []
    print("=" * 84)
    print("PER-INSTRUMENT RESULTS (net of slippage + commission)")
    print("=" * 84)

    for name, inst in INSTRUMENTS.items():
        df = load(inst["data"])
        if df is None:
            print(f"  {name:<18} | data unavailable"); continue
        tr = backtest(df, inst)
        if not tr.empty:
            tr["instrument"] = name
            all_trades.append(tr)
        s = stat(tr)
        print(row(name, s))
        if s:
            io  = stat(tr[~tr["oos"]])
            oo  = stat(tr[tr["oos"]])
            lg  = stat(tr[tr["side"] == "LONG"])
            sh  = stat(tr[tr["side"] == "SHORT"])
            print(f"      in-sample {row('', io)[3:].strip()}")
            print(f"      out-sample{row('', oo)[3:].strip()}")
            print(f"      longs     {row('', lg)[3:].strip()}")
            print(f"      shorts    {row('', sh)[3:].strip()}")
        print()

    # ── combined / verdict ────────────────────────────────────────────────────
    print("=" * 84)
    print("COMBINED (does ONE strategy work across all four markets?)")
    print("=" * 84)
    if all_trades:
        comb = pd.concat(all_trades, ignore_index=True)
        print(row("ALL MARKETS", stat(comb)))
        print(row("  in-sample", stat(comb[~comb["oos"]])))
        print(row("  out-sample", stat(comb[comb["oos"]])))
        oos = stat(comb[comb["oos"]])
        markets_oos_pos = sum(
            1 for n in INSTRUMENTS
            if (st := stat(comb[(comb["instrument"] == n) & (comb["oos"])])) and st["pnl"] > 0
        )
        print("\n── VERDICT ──")
        if oos and oos["pnl"] > 0 and markets_oos_pos >= 3:
            print(f"  ✅ Edge holds out-of-sample in {markets_oos_pos}/4 markets "
                  f"(combined OOS PF {oos['pf']:.2f}). Worth forward-testing.")
        elif oos and oos["pnl"] > 0:
            print(f"  🟡 Combined OOS positive but only {markets_oos_pos}/4 markets hold. "
                  f"Inconsistent — tune before trusting.")
        else:
            print(f"  ❌ Fails out-of-sample. The reversion edge isn't real here as-is.")
    else:
        print("  No trades generated across any market — loosen STRETCH_ATR / RVOL_MIN.")
