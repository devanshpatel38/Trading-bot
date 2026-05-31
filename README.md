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
python -m hyperbot.backtest --out backtest_results.json
python -m hyperbot.pnl_calc backtest_results.json
python -m hyperbot.show_signals backtest_results.json
python -m hyperbot.analyze
```