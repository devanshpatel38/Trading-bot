from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# New Binance "Demo Trading" futures endpoint (the classic testnet.binancefuture.com is
# a separate, older system). Mainnet is fapi.binance.com — same API, real money.
TESTNET_BASE = "https://demo-fapi.binance.com"
MAINNET_BASE = "https://fapi.binance.com"


class BinanceFuturesError(RuntimeError):
    pass


class BinanceFuturesClient:
    """Minimal signed REST client for Binance USDT-M perpetual futures.

    Read + trade (order placement) — this is the execution layer, intentionally separate
    from the read-only data/strategy code. Defaults to the demo (testnet) endpoint.
    """

    def __init__(self, symbol: str = "BTCUSDT", testnet: bool = True, env_path: str = "hyperbot/.env"):
        load_dotenv(env_path)
        if testnet:
            self.key = os.getenv("BINANCE_TESTNET_KEY")
            self.secret = os.getenv("BINANCE_TESTNET_SECRET")
            self.base = TESTNET_BASE
        else:
            self.key = os.getenv("BINANCE_KEY")
            self.secret = os.getenv("BINANCE_SECRET")
            self.base = MAINNET_BASE
        if not self.key or not self.secret:
            raise BinanceFuturesError(f"API keys not found in {env_path} (testnet={testnet})")
        self.symbol = symbol
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.key})
        self._tick, self._step, self._min_notional = self._load_filters()

    # --- low-level -------------------------------------------------------- #
    def _server_time(self) -> int:
        return int(self.session.get(self.base + "/fapi/v1/time", timeout=15).json()["serverTime"])

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = self._server_time()
            params["recvWindow"] = 5000
            qs = urlencode(params)
            params_sig = qs + "&signature=" + hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
            url = f"{self.base}{path}?{params_sig}"
            resp = self.session.request(method, url, timeout=15)
        else:
            resp = self.session.request(method, self.base + path, params=params, timeout=15)
        if resp.status_code != 200:
            raise BinanceFuturesError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _load_filters(self):
        ex = self.session.get(self.base + "/fapi/v1/exchangeInfo", timeout=15).json()
        sym = next(s for s in ex["symbols"] if s["symbol"] == self.symbol)
        tick = step = min_notional = None
        for f in sym["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
            elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("notional", f.get("minNotional", 0)))
        return tick, step, (min_notional or 0.0)

    # --- rounding helpers ------------------------------------------------- #
    def round_price(self, p: float) -> float:
        return round(math.floor(p / self._tick) * self._tick, 8)

    def round_qty(self, q: float) -> float:
        return round(math.floor(q / self._step) * self._step, 8)

    # --- account ---------------------------------------------------------- #
    def available_usdt(self) -> float:
        for b in self._request("GET", "/fapi/v2/balance", signed=True):
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def position(self) -> dict | None:
        """Current position for the symbol, or None if flat. Includes Binance's
        breakEvenPrice (entry adjusted for round-trip fees) so the breakeven-stop move
        after the 2R partial can rest at true net-zero rather than the raw entry."""
        for p in self._request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol}, signed=True):
            amt = float(p["positionAmt"])
            if amt != 0.0:
                return {"qty": amt, "side": "long" if amt > 0 else "short",
                        "entry": float(p["entryPrice"]),
                        "break_even": float(p.get("breakEvenPrice") or p["entryPrice"]),
                        "mark": float(p["markPrice"]),
                        "liq": float(p["liquidationPrice"]),
                        "unreal": float(p["unRealizedProfit"])}
        return None

    def set_leverage(self, leverage: int):
        return self._request("POST", "/fapi/v1/leverage",
                             {"symbol": self.symbol, "leverage": leverage}, signed=True)

    # --- orders ----------------------------------------------------------- #
    def market(self, side: str, qty: float, reduce_only: bool = False):
        return self._request("POST", "/fapi/v1/order", {
            "symbol": self.symbol, "side": side.upper(), "type": "MARKET",
            "quantity": self.round_qty(qty), "reduceOnly": str(reduce_only).lower(),
        }, signed=True)

    def _conditional(self, side: str, otype: str, trigger: float,
                     qty: float | None = None, close_position: bool = False):
        """Conditional (algo) order. Since 2025-12-09 STOP/TP orders go through the Algo
        Service (/fapi/v1/algoOrder), not /fapi/v1/order. `side` is the CLOSING side."""
        p = {"algoType": "CONDITIONAL", "symbol": self.symbol, "side": side.upper(),
             "type": otype, "triggerPrice": self.round_price(trigger), "workingType": "MARK_PRICE"}
        if close_position:
            p["closePosition"] = "true"
        else:
            p["quantity"] = self.round_qty(qty)
            p["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/algoOrder", p, signed=True)

    def stop_market(self, side: str, trigger_price: float, qty: float | None = None,
                    close_position: bool = False):
        return self._conditional(side, "STOP_MARKET", trigger_price, qty, close_position)

    def take_profit(self, side: str, trigger_price: float, qty: float | None = None,
                    close_position: bool = False):
        return self._conditional(side, "TAKE_PROFIT_MARKET", trigger_price, qty, close_position)

    # --- order bookkeeping ----------------------------------------------- #
    def open_orders(self):
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol}, signed=True)

    def open_algo_orders(self):
        return self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": self.symbol}, signed=True)

    def cancel_algo(self, algo_id):
        return self._request("DELETE", "/fapi/v1/cancelAlgoOrder",
                             {"symbol": self.symbol, "algoId": algo_id}, signed=True)

    def cancel_all(self):
        """Cancel both regular and algo (conditional) open orders for the symbol."""
        for o in self.open_algo_orders():
            self.cancel_algo(o["algoId"])
        try:
            self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": self.symbol}, signed=True)
        except BinanceFuturesError:
            pass