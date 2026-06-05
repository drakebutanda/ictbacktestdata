# MNQ Forward Logger — self-running paper track record

Builds an **un-overfittable forward record** of the morning deep-VWAP / MACD-hook
edge, on GitHub's servers, **with your Mac off**. Once a day it:

1. **Resolves** any open paper trades against real subsequent price (win/loss/expired, net of slippage + commission),
2. **Scans** for new signals since the last run (gap-proof — never misses a day),
3. **Commits** the updated `mnq_forward_journal.csv` back here, and
4. optionally **pings your phone**.

You just open `mnq_forward_journal.csv` whenever you want. Once **10 trades have
closed**, the logger prints a bootstrap verdict (fund / keep-paper / don't-fund).

---

## One-time setup (~10 min)

### 1. Create a **private** repo and push these files
> Private matters: it keeps your strategy (and any API keys) off the public internet.
> Private repos still get free Actions minutes.

From this folder (`~/Desktop/mnq-forward-bot`):

```bash
git init
git add -A
git commit -m "initial forward logger"
git branch -M main
# create an EMPTY private repo named e.g. mnq-forward-bot at https://github.com/new
git remote add origin https://github.com/<YOUR_USERNAME>/mnq-forward-bot.git
git push -u origin main
```

### 2. Confirm Actions is on
Repo → **Actions** tab → enable workflows if prompted. The job runs daily at
**01:00 UTC**. To run it now, open **Actions → MNQ forward logger → Run workflow**.

### 3. (Optional) Phone alerts
Repo → **Settings → Secrets and variables → Actions → New repository secret**.
Pick **one** channel:

- **Telegram** (simplest phone push): message [@BotFather](https://t.me/BotFather) → `/newbot` →
  copy the token. Then message your new bot once, and visit
  `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your numeric chat id.
  Add secrets `TELEGRAM_TOKEN` and `TELEGRAM_CHAT`.
- **Discord/Slack**: create an incoming webhook, add its URL as secret `WEBHOOK_URL`.

No secret set → it just commits the journal silently (still fully works).

### 4. (Optional) FRED macro bonus
Add secret `FRED_API_KEY` (free key from https://fred.stlouisfed.org/docs/api/api_key.html).
Without it the macro score degrades gracefully — no error.

---

## Reading the journal

`mnq_forward_journal.csv`, one row per paper trade:

| column | meaning |
|---|---|
| `status` | OPEN until resolved → WIN / LOSS / EXPIRED_WIN / EXPIRED_LOSS |
| `pnl` | **net of slippage + commission** |
| `zone` | DEEP (−4 to −2 MACD, the edge) or MID |
| `confidence` | 85+ = full-risk band |

---

## What this is NOT (yet)

A **validation** tool, not a trading bot. Data is ~15-min delayed (free Yahoo),
so it's paper-only. **Fund nothing until ≥10 closed trades show the edge holds
forward.** That's the whole point — prove it on data nobody tuned to.
