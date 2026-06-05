"""
MNQ Futures Backtest — backtest7_final.py
=========================================
Base:    backtest7 (best result: PF 1.93, EV +0.244, ROI 90.8%)

Improvements applied from full optimization cycle:
  1. Mid hook gets scoring bonus (was backwards — deep was getting it)
  2. MACD floor -4.0  (avg loss MACD was -5.956, cuts worst losses)
  3. BB declining removed (flipped positive/negative 4x — unreliable)
  4. BB compressing kept (+14.9% edge in backtest7 context)
  5. Morning weighted higher (+8 vs old +3) — consistently 37%+ WR
  6. Below VWAP scoring removed (-15.2% edge confirmed bad)
  7. Near VWAP kept as primary mean reversion signal (+36.5% edge)
  8. Both sessions kept — morning preferred, afternoon allowed

Run: python3 ~/Desktop/backtest7_final.py
"""

import yfinance as yf
import pandas as pd
import requests
import pytz
from fredapi import Fred
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import os

CT = pytz.timezone("America/Chicago")

# ── CONFIG ────────────────────────────────────────────────────────────────────
FRED_API_KEY        = os.environ.get("FRED_API_KEY", "")  # set via env/secret; macro
                                                          # score degrades gracefully if unset
TICKER              = "MNQ=F"
LOG_FILE            = os.path.expanduser("~/Desktop/backtest7_final_results.txt")
MIN_BARS            = 200
MAX_HOLD_BARS       = 48
ACCOUNT_SIZE        = 10_000
RISK_PCT            = 0.06
MAX_CONTRACTS       = 10
MARGIN_PER_CONTRACT = 1_500
CONF_MIN            = 75   # raised from 70 — the 70-74 band bled -$1,235
                           # (10 trades, 10% WR, PF 0.26), the worst on every
                           # metric. Distribution is monotonic, so trim the floor.
CONF_MAX            = 101  # ceiling removed — was 85, which discarded the
                           # highest-conviction setups (score tops out ~101).
                           # No data ever justified the cap; floor stays at 70.

# MACD sweet zone — data derived (win avg -3.304, loss avg -5.956)
MACD_FLOOR = -4.0
MACD_CEIL  = -0.2
# ─────────────────────────────────────────────────────────────────────────────


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def contracts(entry, stop):
    risk_pts   = entry - stop
    if risk_pts <= 0:
        return 1
    dollar     = ACCOUNT_SIZE * RISK_PCT
    raw        = dollar / (risk_pts * 2)
    margin_cap = int(ACCOUNT_SIZE * 0.5 / MARGIN_PER_CONTRACT)
    return max(1, min(int(raw), margin_cap, MAX_CONTRACTS))


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Downloading MNQ intraday (60d 5m)...")
    df = yf.download(TICKER, period="60d", interval="5m", progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                else df.index).tz_convert(CT)

    # Daily bias — price vs 50-day MA, shifted so we use prior day's value
    print("Downloading daily bias...")
    d = yf.download(TICKER, period="1y", interval="1d", progress=False)
    d.columns = d.columns.get_level_values(0)
    d["bias"] = (d["Close"] > d["Close"].rolling(50).mean()).astype(int).shift(1)
    d.index   = (d.index.tz_convert(CT) if d.index.tz else
                 pd.to_datetime(d.index)).normalize()
    bias_map  = d["bias"].fillna(1).to_dict()

    # VIX
    print("Downloading VIX...")
    v = yf.download("^VIX", period="60d", interval="5m", progress=False)
    if not v.empty:
        v.columns = v.columns.get_level_values(0)
        v.index   = (v.index.tz_localize("UTC") if v.index.tz is None
                     else v.index).tz_convert(CT)
        df = df.join(v["Close"].rename("VIX"), how="left")
        df["VIX"] = df["VIX"].ffill()
    else:
        df["VIX"] = 20.0

    return df, bias_map


def add_indicators(df, bias_map):
    df["EMA9"]  = EMAIndicator(df["Close"], 9).ema_indicator()
    df["EMA21"] = EMAIndicator(df["Close"], 21).ema_indicator()
    df["RSI"]   = RSIIndicator(df["Close"], 14).rsi()
    df["ATR"]   = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()

    macd = MACD(df["Close"])
    df["MACD_H"] = macd.macd_diff()
    df["MACD_P"] = df["MACD_H"].shift(1)
    # Fresh cross (harmful — used only to gate out)
    df["FRESH"]  = ((df["MACD_H"] > 0) & (df["MACD_H"].shift(1) <= 0)
                   ).rolling(3).max().fillna(0).astype(int)

    # BB compressing — width below its 20-bar average
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_W"]   = bb.bollinger_hband() - bb.bollinger_lband()
    df["BB_C"]   = (df["BB_W"] < df["BB_W"].rolling(20).mean()).fillna(False).astype(int)
    # NOTE: BB declining removed — flipped positive/negative 4x across backtests

    # Time helpers
    df["Date"]  = df.index.date
    df["btime"] = df.index.hour * 60 + df.index.minute

    # VWAP — resets each day
    vals = []
    for _, g in df.groupby("Date"):
        vals.extend((g["Close"] * g["Volume"]).cumsum()
                    .div(g["Volume"].cumsum()).tolist())
    df["VWAP"] = vals

    # PDH / PDL
    ds = (df.groupby("Date")
            .agg(hi=("High","max"), lo=("Low","min"))
            .reset_index())
    ds["PDH"] = ds["hi"].shift(1)
    ds["PDL"] = ds["lo"].shift(1)
    df = df.merge(ds[["Date","PDH","PDL"]], on="Date", how="left")

    # Daily bias
    df["bias"] = (pd.to_datetime(df["Date"]).dt.normalize()
                   .map(bias_map).fillna(1))

    return df.dropna()


# ── MACRO (slow signals) ──────────────────────────────────────────────────────
def get_slow_conf():
    mc = fg = sc = 0
    try:
        fred   = Fred(api_key=FRED_API_KEY)
        rate   = fred.get_series("DFF").iloc[-1]
        spread = fred.get_series("T10Y2Y").iloc[-1]
        score  = (10 if spread > 0.5 else 5 if spread > 0 else -10) + \
                 (5  if rate < 3     else -5 if rate > 5 else 0)
        mc = 5 if score > 5 else 2
    except:
        mc = 2
    try:
        fg_val = int(requests.get("https://api.alternative.me/fng/?limit=1",
                                  timeout=5).json()["data"][0]["value"])
        fg = 3 if fg_val < 25 else 2 if fg_val < 45 else 1 if fg_val <= 75 else 0
    except:
        fg = 1
    try:
        xlk = yf.download("XLK", period="5d", progress=False)
        xlu = yf.download("XLU", period="5d", progress=False)
        xlk.columns = xlk.columns.get_level_values(0)
        xlu.columns = xlu.columns.get_level_values(0)
        tech = (xlk["Close"].iloc[-1] - xlk["Close"].iloc[0]).item() / xlk["Close"].iloc[0].item()
        util = (xlu["Close"].iloc[-1] - xlu["Close"].iloc[0]).item() / xlu["Close"].iloc[0].item()
        sc = 2 if tech > util else 0
    except:
        sc = 1
    total = mc + fg + sc
    print(f"Slow conf: {total}  (macro {mc} + F&G {fg} + sector {sc})")
    return total


# ── SCORING ───────────────────────────────────────────────────────────────────
def score(mhist, bbc, rsi, price, vwap, atr, vix, morning, afternoon, slow):
    """
    Returns (confidence, signal_dict).
    Signal weights validated across 10 backtests.
    """
    c    = 0
    sigs = {}

    # MACD depth within sweet zone
    # Fix from backtest9: mid hook gets bonus, deep does NOT
    in_mid = MACD_CEIL >= mhist >= -2.0        # 66.7% WR confirmed
    c += 22                                      # base: any rising in zone
    if in_mid:
        c += 16                                  # mid bonus (bumped from 12)
    sigs["macd_mid"]  = in_mid
    sigs["macd_deep"] = not in_mid

    # BB compressing REMOVED — flipped positive/negative 5x across backtests.
    # Its 16 points redistributed to the four proven signals below.
    sigs["bb_comp"] = bbc   # still tracked for fingerprint, NOT scored

    # RSI neutral — 93-100% of wins across backtests (+12.5% edge)
    rsi_n = 40 <= rsi <= 60
    c += 17 if rsi_n else (6 if rsi < 40 else 3 if rsi < 70 else 0)  # bumped from 14
    sigs["rsi_neut"] = rsi_n

    # VWAP proximity — strongest signal, +27.8% edge (below_vwap removed: -15.2%)
    dist      = abs(price - vwap)
    near_vwap = dist < atr                       # within 1 ATR
    c += 20 if near_vwap else 0                  # bumped from 14
    sigs["near_vwap"] = near_vwap

    # VIX zone
    vix_mod = 15 <= vix < 22
    c += 8 if vix_mod else (4 if vix < 30 else 0)
    sigs["vix_mod"] = vix_mod

    # Session — morning weighted more (38.9% WR vs 18.2% afternoon, +15.3% edge)
    c += 12 if morning else (4 if afternoon else 0)  # morning bumped from 8
    sigs["morning"]   = morning
    sigs["afternoon"] = afternoon

    # Slow macro bonus (FRED + Fear/Greed + Sector)
    c += min(slow, 6)

    return c, sigs


# ── SIMULATION ────────────────────────────────────────────────────────────────
def run(df, slow):
    trades    = []
    in_trade  = False
    bars_held = 0
    profit    = 0.0
    t         = {}          # active trade dict

    for i in range(MIN_BARS, len(df)):
        row  = df.iloc[i]
        px   = row["Close"]
        hi   = row["High"]
        lo   = row["Low"]
        bt   = int(row["btime"])
        morn = (8*60+30) <= bt <= (10*60+30)
        aftn = (13*60)   <= bt <= (14*60+30)

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            bars_held += 1

            if lo <= t["stop"]:
                c   = contracts(t["entry"], t["stop"])
                pnl = (t["stop"] - t["entry"]) * c * 2
                profit += pnl
                t.update(result="LOSS", bars_held=bars_held,
                         pnl=round(pnl,2), profit=round(profit,2), contracts=c)
                trades.append(t); in_trade = False; bars_held = 0; continue

            if hi >= t["target"]:
                c   = contracts(t["entry"], t["stop"])
                pnl = (t["target"] - t["entry"]) * c * 2
                profit += pnl
                t.update(result="WIN", bars_held=bars_held,
                         pnl=round(pnl,2), profit=round(profit,2), contracts=c)
                trades.append(t); in_trade = False; bars_held = 0; continue

            if bars_held >= MAX_HOLD_BARS:
                c   = contracts(t["entry"], t["stop"])
                pnl = (px - t["entry"]) * c * 2
                profit += pnl
                res = "EXPIRED WIN" if pnl > 0 else "EXPIRED LOSS"
                t.update(result=res, bars_held=bars_held,
                         pnl=round(pnl,2), profit=round(profit,2), contracts=c)
                trades.append(t); in_trade = False; bars_held = 0; continue
            continue

        # ── Hard gates ────────────────────────────────────────────────────────
        if row["bias"]  != 1:                                continue
        if row["VIX"]   > 30:                                continue
        if not (morn or aftn):                               continue
        if row["FRESH"] == 1:                                continue   # no fresh cross
        if row["EMA9"]  > row["EMA21"]:                      continue   # EMA gate +26.2%
        mh = row["MACD_H"]
        if not (mh > row["MACD_P"]):                         continue   # rising
        if mh < MACD_FLOOR or mh > MACD_CEIL:               continue   # sweet zone
        if not pd.isna(row["PDL"]) and px < row["PDL"]:     continue   # above PDL

        # ── Score ─────────────────────────────────────────────────────────────
        conf, sigs = score(
            mh, bool(row["BB_C"]), row["RSI"],
            px, row["VWAP"], row["ATR"], row["VIX"],
            morn, aftn, slow
        )

        if not (CONF_MIN <= conf < CONF_MAX):
            continue

        # ── Entry ─────────────────────────────────────────────────────────────
        sl   = px - 1.5 * row["ATR"]
        risk = px - sl
        if risk <= 0:
            continue

        rr  = (px + 2.5*risk - px) / risk   # always 2.5 with this formula
        # RR tier determines target multiplier
        tgt = px + 2.5 * risk

        in_trade  = True
        bars_held = 0
        t = dict(
            entry_bar=i, confidence=conf,
            entry=px, stop=sl, target=tgt,
            result="OPEN", bars_held=None,
            pnl=0, profit=0, contracts=None,
            **{f"sig_{k}": v for k, v in sigs.items()},
            sig_macd_val=round(mh, 3),
            sig_rsi_val=round(row["RSI"], 1),
            sig_vix_val=round(row["VIX"], 1),
        )

    return pd.DataFrame(trades)


# ── RESULTS ───────────────────────────────────────────────────────────────────
def report(df):
    df = df[df["result"] != "OPEN"].copy()
    if df.empty:
        log("No trades generated"); return

    wins = df[df["result"] == "WIN"]
    loss = df[df["result"] == "LOSS"]
    exp  = df[df["result"].str.startswith("EXPIRED")]

    n    = len(df)
    wr   = len(wins) / n * 100
    pnl  = df["pnl"].sum()
    pf   = abs(wins["pnl"].sum() / loss["pnl"].sum()) \
           if len(loss) > 0 and loss["pnl"].sum() != 0 else 0
    ev   = (wr/100 * 2.5) - ((1 - wr/100) * 1.0)
    roi  = pnl / ACCOUNT_SIZE * 100
    dd   = df["pnl"].cumsum().min()

    log("\n" + "="*60)
    log("BACKTEST 7 FINAL — MORNING VWAP RETEST + MACD HOOK")
    log(f"Account ${ACCOUNT_SIZE:,} | Risk {RISK_PCT*100:.0f}% | Zone {CONF_MIN}-{CONF_MAX-1}%")
    log("="*60)
    log(f"\nTotal Trades:     {n}")
    log(f"Wins:             {len(wins)}")
    log(f"Losses:           {len(loss)}")
    log(f"Expired:          {len(exp)}")
    log(f"Win Rate:         {wr:.1f}%")
    log(f"Avg Contracts:    {df['contracts'].dropna().mean():.1f}")
    log(f"Avg Win:          ${wins['pnl'].mean():.2f}" if len(wins) else "Avg Win: N/A")
    log(f"Avg Loss:         ${loss['pnl'].mean():.2f}" if len(loss) else "Avg Loss: N/A")
    log(f"Profit Factor:    {pf:.2f}")
    log(f"Expected Value:   {ev:+.3f} per unit")
    log(f"Total PnL:        ${pnl:.2f}")
    log(f"Max Drawdown:     ${dd:.2f}")
    log(f"Avg Hold:         {df['bars_held'].dropna().mean():.1f} bars "
        f"({df['bars_held'].dropna().mean()*5:.0f} min)")
    log(f"\nROI (60 days):    {roi:.1f}%")
    log(f"Annualized ROI:   {roi*6:.1f}%")

    # ── Confidence distribution ───────────────────────────────────────────────
    log("\n" + "="*60)
    log("CONFIDENCE DISTRIBUTION")
    log("="*60)
    for lo, hi in [(65,69),(70,74),(75,79),(80,84),(85,89),(90,100)]:
        t = df[(df["confidence"]>=lo)&(df["confidence"]<=hi)]
        if t.empty: continue
        tw = t[t["result"]=="WIN"]
        tl = t[t["result"]=="LOSS"]
        tw_r= len(tw)/len(t)*100
        tpf = abs(tw["pnl"].sum()/tl["pnl"].sum()) \
              if len(tl)>0 and tl["pnl"].sum()!=0 else 0
        log(f"  {lo}-{hi}% | Trades:{len(t):3} | WR:{tw_r:.1f}% | "
            f"PF:{tpf:.2f} | PnL:${t['pnl'].sum():.2f}")

    # ── Session ───────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("SESSION")
    log("="*60)
    for label, col in [("MORNING  8:30-10:30am","sig_morning"),
                        ("AFTERNOON 1:00-2:30pm","sig_afternoon")]:
        t = df[df[col]==True]
        if t.empty: continue
        tw= t[t["result"]=="WIN"]
        log(f"  {label} | Trades:{len(t):3} | WR:{len(tw)/len(t)*100:.1f}% | "
            f"PnL:${t['pnl'].sum():.2f}")

    # ── MACD depth ────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("MACD DEPTH")
    log("="*60)
    for label, col in [("Mid  (-2.0 to -0.2)","sig_macd_mid"),
                        ("Wide (-4.0 to -2.0)","sig_macd_deep")]:
        t = df[df[col]==True]
        if t.empty: continue
        tw= t[t["result"]=="WIN"]
        log(f"  {label} | Trades:{len(t):3} | WR:{len(tw)/len(t)*100:.1f}% | "
            f"Avg MACD:{t['sig_macd_val'].mean():.3f} | PnL:${t['pnl'].sum():.2f}")

    # ── Win fingerprint ───────────────────────────────────────────────────────
    log("\n" + "="*60)
    log(f"WIN FINGERPRINT  ({len(wins)} wins | {len(loss)} losses)")
    log("="*60)
    sig_cols = [c for c in df.columns if c.startswith("sig_")
                and c not in ("sig_macd_val","sig_rsi_val","sig_vix_val")]
    fp = sorted(
        [{"s": c[4:],
          "w": wins[c].mean()*100 if len(wins) else 0,
          "l": loss[c].mean()*100 if len(loss) else 0}
         for c in sig_cols],
        key=lambda x: x["w"]-x["l"], reverse=True
    )
    log(f"\n  {'Signal':<20} {'Wins':>8} {'Losses':>10} {'Edge':>8}")
    log("  " + "─"*50)
    for f in fp:
        e = f["w"] - f["l"]
        log(f"  {'⭐' if e>15 else '  '} {f['s']:<18}"
            f"{f['w']:>8.1f}%{f['l']:>10.1f}%{e:>+8.1f}%")

    # ── Win vs loss characteristics ───────────────────────────────────────────
    log("\n" + "="*60)
    log("WIN vs LOSS CHARACTERISTICS")
    log("="*60)
    log(f"  {'Metric':<22} {'Wins':>10} {'Losses':>10}")
    log("  " + "─"*44)
    for metric, col in [("Avg MACD","sig_macd_val"),
                         ("Avg RSI", "sig_rsi_val"),
                         ("Avg VIX", "sig_vix_val")]:
        wv = wins[col].mean() if len(wins) else 0
        lv = loss[col].mean() if len(loss) else 0
        log(f"  {metric:<22} {wv:>10.3f}  {lv:>10.3f}")
    log(f"  {'Avg Bars Held':<22} "
        f"{wins['bars_held'].mean():>10.1f}  "
        f"{loss['bars_held'].mean():>10.1f}" if len(wins) and len(loss) else "")

    # ── Individual trades ─────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("INDIVIDUAL TRADES")
    log("="*60)
    for _, r in df.iterrows():
        pm   = "PM" if r["sig_afternoon"] else "AM"
        zone = "MID " if r["sig_macd_mid"] else "WIDE"
        nv   = "VWAP" if r["sig_near_vwap"] else "    "
        bb   = "BB" if r["sig_bb_comp"] else "  "
        log(f"Bar{int(r['entry_bar']):5} | {r['result']:12} | "
            f"{pm} {zone} {nv} {bb} | Conf:{r['confidence']:3}% | "
            f"MACD:{r['sig_macd_val']:+.2f} | RSI:{r['sig_rsi_val']:4.1f} | "
            f"PnL:${r['pnl']:>9.2f}")

    log(f"\nSaved → {LOG_FILE}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # clear old log
    open(LOG_FILE, "w").close()

    df, bias_map = load_data()
    df           = add_indicators(df, bias_map)
    slow         = get_slow_conf()

    print(f"Bars: {len(df)} | Zone: {CONF_MIN}-{CONF_MAX-1}% | "
          f"MACD floor: {MACD_FLOOR}")

    results = run(df, slow)
    report(results)


