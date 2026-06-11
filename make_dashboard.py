"""
make_dashboard.py — generate DASHBOARD.md from ict_forward_journal.csv.
Standalone (reads only the CSV) so it needs no import of the logger. Runs in its
own workflow and commits a clean, GitHub-rendered view of the live paper account.
"""
import os, random
import pandas as pd
from datetime import datetime, timezone

_HERE   = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(_HERE, "ict_forward_journal.csv")
DASH    = os.path.join(_HERE, "DASHBOARD.md")
ORIGIN  = 10_000


def load():
    cols = ["entry_ts", "instrument", "side", "status", "contracts", "entry",
            "entry_fill", "stop", "target", "exit_ts", "exit_price", "bars_held", "pnl"]
    if os.path.exists(JOURNAL):
        try:
            return pd.read_csv(JOURNAL)
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def main():
    j = load()
    closed = j[j["status"] != "OPEN"].copy() if "status" in j else j.iloc[0:0]
    openn  = j[j["status"] == "OPEN"] if "status" in j else j.iloc[0:0]
    pnls   = pd.to_numeric(closed["pnl"], errors="coerce").dropna() if len(closed) else pd.Series([], dtype=float)
    cum    = float(pnls.sum())
    wr     = (pnls > 0).mean() * 100 if len(pnls) else 0.0
    gw, gl = pnls[pnls > 0].sum(), pnls[pnls < 0].sum()
    pf     = abs(gw / gl) if gl else (float("inf") if gw else 0.0)
    origin, profit = ORIGIN + min(0.0, cum), max(0.0, cum)

    L = ["# 📊 ICT FVG — Live Paper Trading Dashboard", ""]
    L += [f"*Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · auto-generated each run · "
          f"3% risk · $10k account · MES/MNQ/Gold/Crude · long & short*", ""]
    L += ["## 💰 Accounts", "", "| Account | Balance |", "|---|---:|",
          f"| 🏦 Origin (your capital) | **${origin:,.2f}**  (base ${ORIGIN:,}) |",
          f"| 💵 Profit (house money) | **${profit:,.2f}** |",
          f"| 📈 Total net P&L | **${cum:,.2f}** |", ""]
    L += ["## 📈 Performance", "", "| Metric | Value |", "|---|---:|",
          f"| Closed trades | {len(closed)} |", f"| Open now | {len(openn)} |",
          f"| Win rate | {wr:.1f}% |", f"| Profit factor | {pf:.2f} |"]
    if len(closed) >= 10:
        rng, p = random.Random(42), pnls.tolist()
        tot = [sum(p[rng.randrange(len(p))] for _ in p) for _ in range(4000)]
        prob = sum(1 for t in tot if t > 0) / len(tot) * 100
        verdict = "✅ fund-ready" if prob >= 90 else "🟡 keep paper-trading" if prob >= 75 else "❌ stop"
        L += [f"| Bootstrap P(profit) | {prob:.0f}% — {verdict} |"]
    else:
        L += [f"| Status | building sample — {max(0, 10 - len(closed))} more closed trades for a verdict |"]
    L += [""]

    if len(closed):
        L += ["## 🎯 By market", "", "| Market | Trades | Win rate | Net P&L |", "|---|---:|---:|---:|"]
        for name, g in closed.groupby("instrument"):
            gp = pd.to_numeric(g["pnl"], errors="coerce")
            L += [f"| {name} | {len(g)} | {(gp > 0).mean()*100:.0f}% | ${gp.sum():,.2f} |"]
        L += [""]
    if len(openn):
        L += ["## 🔓 Open positions", "", "| Opened | Market | Side | Qty | Entry | Stop | Target |",
              "|---|---|---|---:|---:|---:|---:|"]
        for _, e in openn.iterrows():
            L += [f"| {e['entry_ts']} | {e['instrument']} | {e.get('side','')} | {int(e['contracts'])} | "
                  f"{e['entry']} | {e['stop']} | {e['target']} |"]
        L += [""]
    if len(closed):
        L += ["## 🧾 Recent closed trades", "", "| Closed | Market | Side | Result | P&L |", "|---|---|---|---|---:|"]
        for _, e in closed.tail(15).iloc[::-1].iterrows():
            try: pv = f"${float(e['pnl']):,.2f}"
            except Exception: pv = str(e['pnl'])
            L += [f"| {e['exit_ts']} | {e['instrument']} | {e.get('side','')} | {e['status']} | {pv} |"]
        L += [""]
    L += ["---", "*Paper trading on 15-min-delayed data — validation only, not live execution.*"]
    with open(DASH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {DASH}: {len(closed)} closed, {len(openn)} open, net ${cum:,.2f}")


if __name__ == "__main__":
    main()
