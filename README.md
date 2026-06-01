# Hyperbot

Read-only Hyperliquid **testnet** trading bot: fetch candles, run technical strategies, backtest. No live trading.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example hyperbot\.env
```

## Usage
```bash
python -m hyperbot.backtest --symbol BTC --interval 15m --days 30 --rr 2 --confidence 50 --minagree 3
python -m hyperbot.pnl_calc backtest_results.json
python -m hyperbot.analyze
```

Long backtests (e.g. `--days 180`) recompute strategies per bar and can take several minutes.