"""
MNQ Futures Backtest — backtest8_robust.py
==========================================
Built on backtest7_final's validated edge (morning VWAP retest + deep MACD
hook, high-conviction band). The point of THIS file is not a bigger backtest
number — it is a TRUSTWORTHY one. Three things separate a system that thrives
from one that just backtests well:

  1. REAL COSTS      — commission + slippage on every fill. The old sim filled
                       at the exact stop/target for free. Real money doesn't.
  2. OUT-OF-SAMPLE   — the rules are frozen, then scored separately on the first
                       70% of history (in-sample) and the last 30% it never
                       "saw". If the edge only exists in-sample, it isn't real.
  3. CONVICTION SIZING — risk scales with the signal. The 85+ band (PF ~2.2)
                       gets full risk; the marginal 75-84 band gets half. This
                       is where compounding money is actually made.

Reuses the data / indicator / scoring layer from backtest7_final unchanged, so
the signal is identical — only the execution realism and sizing differ.

Run: python3 ~/Desktop/backtest8_robust.py
"""

import os
import pandas as pd

from backtest7_final import (
    load_data, add_indicators, get_slow_conf, score,
    MIN_BARS, MAX_HOLD_BARS, ACCOUNT_SIZE, MAX_CONTRACTS,
    MARGIN_PER_CONTRACT, CONF_MIN, CONF_MAX, MACD_FLOOR, MACD_CEIL,
)

# ── EXECUTION REALISM ──────────────────────────────────────────────────────────
POINT_VALUE   = 2.0    # MNQ = $2 per index point per contract
TICK          = 0.25   # MNQ tick size in points ($0.50 per tick)
SLIP_TICKS    = 1      # slippage assumed on every market fill, per side
COMMISSION_RT = 1.50   # broker commission, round-trip, per contract (retail MNQ)

# ── CONVICTION SIZING ──────────────────────────────────────────────────────────
# Half-Kelly-ish. The monotonic confidence distribution earns the right to size
# by conviction: full risk on the proven 85+ band, half on the marginal band.
RISK_HI   = 0.06   # conf >= 85
RISK_LO   = 0.03   # CONF_MIN <= conf < 85

# ── OUT-OF-SAMPLE SPLIT ────────────────────────────────────────────────────────
OOS_FRACTION = 0.30    # last 30% of bars is the holdout the rules never tune to

LOG_FILE = os.path.expanduser("~/Desktop/backtest8_robust_results.txt")


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def contracts(entry, stop, conf):
    """Position size: risk a % of the account, scaled by conviction."""
    risk_pts = entry - stop
    if risk_pts <= 0:
        return 1
    risk_pct   = RISK_HI if conf >= 85 else RISK_LO
    dollar     = ACCOUNT_SIZE * risk_pct
    raw        = dollar / (risk_pts * POINT_VALUE)
    margin_cap = int(ACCOUNT_SIZE * 0.5 / MARGIN_PER_CONTRACT)
    return max(1, min(int(raw), margin_cap, MAX_CONTRACTS))


# ── SIMULATION (with costs) ────────────────────────────────────────────────────
def run(df, slow, train_end_bar):
    """Identical gates/scoring to backtest7_final, but every fill pays
    slippage + commission, sizing scales with conviction, and each trade is
    tagged in-sample (IS) or out-of-sample (OOS) by entry bar."""
    slip = SLIP_TICKS * TICK
    trades, in_trade, bars_held, t = [], False, 0, {}

    for i in range(MIN_BARS, len(df)):
        row = df.iloc[i]
        px, hi, lo = row["Close"], row["High"], row["Low"]
        bt   = int(row["btime"])
        morn = (8*60+30) <= bt <= (10*60+30)
        aftn = (13*60)    <= bt <= (14*60+30)

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            bars_held += 1
            c        = t["contracts"]
            comm     = c * COMMISSION_RT
            ent_fill = t["entry_fill"]

            if lo <= t["stop"]:                                   # stop: pay slip
                exit_fill = t["stop"] - slip
                pnl = (exit_fill - ent_fill) * c * POINT_VALUE - comm
                t.update(result="LOSS", bars_held=bars_held, pnl=round(pnl, 2))
                trades.append(t); in_trade = False; bars_held = 0; continue

            if hi >= t["target"]:                                 # target: limit, no slip
                exit_fill = t["target"]
                pnl = (exit_fill - ent_fill) * c * POINT_VALUE - comm
                t.update(result="WIN", bars_held=bars_held, pnl=round(pnl, 2))
                trades.append(t); in_trade = False; bars_held = 0; continue

            if bars_held >= MAX_HOLD_BARS:                        # time exit: market, pay slip
                exit_fill = px - slip
                pnl = (exit_fill - ent_fill) * c * POINT_VALUE - comm
                res = "EXPIRED WIN" if pnl > 0 else "EXPIRED LOSS"
                t.update(result=res, bars_held=bars_held, pnl=round(pnl, 2))
                trades.append(t); in_trade = False; bars_held = 0; continue
            continue

        # ── Hard gates (unchanged from backtest7_final) ───────────────────────
        if row["bias"] != 1:                              continue
        if row["VIX"]  > 30:                              continue
        if not (morn or aftn):                            continue
        if row["FRESH"] == 1:                             continue
        if row["EMA9"]  > row["EMA21"]:                   continue
        mh = row["MACD_H"]
        if not (mh > row["MACD_P"]):                      continue
        if mh < MACD_FLOOR or mh > MACD_CEIL:             continue
        if not pd.isna(row["PDL"]) and px < row["PDL"]:   continue

        conf, sigs = score(mh, bool(row["BB_C"]), row["RSI"], px, row["VWAP"],
                           row["ATR"], row["VIX"], morn, aftn, slow)
        if not (CONF_MIN <= conf < CONF_MAX):             continue

        # ── Entry (slippage on the fill) ──────────────────────────────────────
        sl   = px - 1.5 * row["ATR"]
        risk = px - sl
        if risk <= 0:                                     continue
        tgt  = px + 2.5 * risk
        c    = contracts(px, sl, conf)

        in_trade, bars_held = True, 0
        t = dict(
            entry_bar=i, confidence=conf, contracts=c,
            entry=px, entry_fill=px + slip, stop=sl, target=tgt,
            risk_pct=(RISK_HI if conf >= 85 else RISK_LO),
            oos=(i >= train_end_bar), result="OPEN", bars_held=None, pnl=0,
            **{f"sig_{k}": v for k, v in sigs.items()},
            sig_macd_val=round(mh, 3), sig_rsi_val=round(row["RSI"], 1),
            sig_vix_val=round(row["VIX"], 1),
        )

    return pd.DataFrame(trades)


# ── STATS HELPER ───────────────────────────────────────────────────────────────
def stats(df):
    if df.empty:
        return None
    wins = df[df["result"] == "WIN"]
    loss = df[df["result"] == "LOSS"]
    gross_w = df[df["pnl"] > 0]["pnl"].sum()
    gross_l = df[df["pnl"] < 0]["pnl"].sum()
    n   = len(df)
    return dict(
        n=n,
        wr=(df["pnl"] > 0).sum() / n * 100,
        pnl=df["pnl"].sum(),
        pf=abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        avg=df["pnl"].mean(),
        dd=df["pnl"].cumsum().min(),
    )


def line(label, s):
    if s is None:
        log(f"  {label:<14} | no trades"); return
    log(f"  {label:<14} | Trades:{s['n']:3} | WR:{s['wr']:5.1f}% | "
        f"PF:{s['pf']:5.2f} | PnL:${s['pnl']:>9.2f} | "
        f"Exp:${s['avg']:>7.2f}/trade | DD:${s['dd']:>8.2f}")


# ── RESULTS ────────────────────────────────────────────────────────────────────
def report(df):
    df = df[df["result"] != "OPEN"].copy()
    if df.empty:
        log("No trades generated"); return

    full = stats(df)
    is_  = stats(df[~df["oos"]])
    oos  = stats(df[df["oos"]])

    log("\n" + "=" * 78)
    log("BACKTEST 8 ROBUST — REAL COSTS · CONVICTION SIZING · OUT-OF-SAMPLE")
    log(f"Account ${ACCOUNT_SIZE:,} | Slip {SLIP_TICKS} tick/side | "
        f"Comm ${COMMISSION_RT}/RT | Risk {RISK_LO*100:.0f}%/{RISK_HI*100:.0f}%")
    log("=" * 78)

    log("\n── PERFORMANCE (net of all costs) ──")
    line("FULL 60d", full)
    line("In-sample 70%", is_)
    line("Out-sample 30%", oos)

    # Annualized on the full window
    roi = full["pnl"] / ACCOUNT_SIZE * 100
    log(f"\n  ROI (60d): {roi:+.1f}%   |   Annualized: {roi*6:+.1f}%")

    # ── The honesty verdict ───────────────────────────────────────────────────
    log("\n── VERDICT ──")
    if oos is None:
        log("  ⚠️  No out-of-sample trades — window too short to validate.")
    elif oos["pnl"] > 0 and oos["pf"] >= 1.3:
        log(f"  ✅ Edge HELD out-of-sample (PF {oos['pf']:.2f}, "
            f"+${oos['pnl']:.2f} on {oos['n']} unseen trades). Tradeable — "
            f"forward-test next.")
    elif oos["pnl"] > 0:
        log(f"  🟡 Marginal out-of-sample (PF {oos['pf']:.2f}, "
            f"+${oos['pnl']:.2f}). Positive but thin — paper-trade, don't fund.")
    else:
        log(f"  ❌ Edge FAILED out-of-sample (${oos['pnl']:.2f} on {oos['n']} "
            f"trades). The in-sample profit is likely overfit. Do NOT fund.")

    # ── Sizing breakdown ──────────────────────────────────────────────────────
    log("\n── CONVICTION SIZING ──")
    for label, mask in [("85+ (full risk)", df["confidence"] >= 85),
                        ("75-84 (half)",    df["confidence"] < 85)]:
        s = stats(df[mask])
        if s:
            ac = df[mask]["contracts"].mean()
            log(f"  {label:<16} | Trades:{s['n']:3} | WR:{s['wr']:5.1f}% | "
                f"PF:{s['pf']:5.2f} | PnL:${s['pnl']:>9.2f} | "
                f"AvgContracts:{ac:.1f}")

    # ── Cost drag (what realism actually took) ────────────────────────────────
    total_comm = (df["contracts"] * COMMISSION_RT).sum()
    log(f"\n── COST DRAG ──")
    log(f"  Commission paid: ${total_comm:.2f} over {len(df)} trades "
        f"({df['contracts'].sum():.0f} contract-sides). Slippage is baked into "
        f"every fill above.")

    # ── Individual trades ─────────────────────────────────────────────────────
    log("\n── TRADES (IS = in-sample, OOS = holdout) ──")
    for _, r in df.iterrows():
        tag  = "OOS" if r["oos"] else "IS "
        pm   = "PM" if r["sig_afternoon"] else "AM"
        zone = "MID " if r["sig_macd_mid"] else "DEEP"
        nv   = "VWAP" if r["sig_near_vwap"] else "    "
        log(f"  {tag} Bar{int(r['entry_bar']):5} | {r['result']:12} | "
            f"{pm} {zone} {nv} | Conf:{r['confidence']:3} | "
            f"{int(r['contracts'])}c | MACD:{r['sig_macd_val']:+.2f} | "
            f"PnL:${r['pnl']:>9.2f}")

    log(f"\nSaved → {LOG_FILE}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    open(LOG_FILE, "w").close()

    df, bias_map = load_data()
    df           = add_indicators(df, bias_map)
    slow         = get_slow_conf()

    train_end_bar = MIN_BARS + int((len(df) - MIN_BARS) * (1 - OOS_FRACTION))
    log(f"Bars: {len(df)} | Zone: {CONF_MIN}-{CONF_MAX-1}% | "
        f"OOS holdout: last {OOS_FRACTION*100:.0f}% (bar >= {train_end_bar})")

    results = run(df, slow, train_end_bar)
    report(results)
