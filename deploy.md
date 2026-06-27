# Deploying the chop bot on a VPS

The hourly executor is `hyperbot/binance_bot.py` (the plain‑1:3 OI‑chop strategy on
Binance USDT‑M perpetuals). This guide stands it up on a Linux VPS under cron. It runs
on the **demo / testnet** endpoint by default — no real funds until you pass `--mainnet`.

## 0. Pick a VPS that can reach Binance

Binance's API — **both** `fapi.binance.com` and `demo-fapi.binance.com` — geo‑blocks US
IPs with HTTP **451**. GCP's free `e2-micro` is US‑region‑only, so it **will not work**.
Before anything else, run this *on the VPS* and confirm you get `200`:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://demo-fapi.binance.com/fapi/v1/time
# 200 = reachable.   451 = geo-blocked -> this VPS is useless for Binance.
```

If you get `451`, use **Oracle Cloud Always Free** with a non‑US home region
(e.g. Mumbai / Singapore — ARM Ampere, and more generous than GCP free anyway).

## 1. System dependencies

```bash
sudo apt update && sudo apt install -y git python3-venv python3-pip
```

## 2. Clone and switch to the execution branch

```bash
cd ~
git clone https://github.com/<your-user>/Trading-bot.git
cd Trading-bot
git checkout binance-execution
```

## 3. Virtualenv + dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 4. Secrets (`hyperbot/.env`)

Create this **by hand** — it is gitignored and must never be committed or pasted into
chat. Copy the template and fill in your demo keys from
<https://testnet.binancefuture.com>:

```bash
cp .env.example hyperbot/.env
nano hyperbot/.env          # set BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET
chmod 600 hyperbot/.env     # owner-only read/write
```

## 5. Smoke test (no orders placed)

```bash
.venv/bin/python -m hyperbot.binance_bot --dry
```

`--dry` evaluates the signal and prints what it *would* do without sending any order.
Confirm it fetches candles + OI and reports a regime/recommendation cleanly.

## 6. Cron — fire every hour

```bash
crontab -e
```

Add this single line (replace `<you>` with your username):

```cron
2 * * * * cd /home/<you>/Trading-bot && /home/<you>/Trading-bot/.venv/bin/python -m hyperbot.binance_bot >> /home/<you>/Trading-bot/bot.log 2>&1
```

Why it's written this way:

- **`2 * * * *`** — 2 minutes past the hour, so the 1h candle has *closed* and Binance
  has published it before the bot reads "the last closed bar." Firing at `0` risks
  reading a bar that doesn't exist yet.
- **`cd … &&`** — cron starts in `$HOME` with a bare environment. The bot resolves
  `hyperbot/.env` and `hyperbot/config.yaml` relative to the working directory, so it
  must run from the repo root.
- **absolute venv path** — cron's `PATH` will not find your virtualenv otherwise.
- **`>> bot.log 2>&1`** — captures stdout and errors into your audit trail
  (gitignored).
- **no `--mainnet`** — stays on the demo endpoint. Add `--mainnet` (and set
  `BINANCE_KEY` / `BINANCE_SECRET`) only when going live with real funds.

Timezone is a non‑issue: the bot keys off UTC candle timestamps and `time.time()`, so
hourly firing is correct regardless of the VPS's local timezone.

## 7. Operating it

```bash
tail -f bot.log                         # watch live
git -C ~/Trading-bot pull               # update: state + log are gitignored, no conflicts
crontab -l                              # confirm the job is installed
```

The bot is idempotent per run: if a position is open it does nothing; if flat it cancels
any leftover bracket orders, evaluates the signal on the last closed bar, and (only on a
fresh bar) enters with stop + take‑profit brackets. Running it twice in one hour is safe.