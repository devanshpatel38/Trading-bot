"""Retry behaviour of the Binance exec client's low-level _request.

Safety-critical: idempotent GET reads retry on transient 408/5xx; order-placing POSTs
must NEVER retry (a timed-out order may have executed -> a retry could double-fill)."""
import pytest

from hyperbot.binance_exec import BinanceFuturesClient, BinanceFuturesError


class _Resp:
    def __init__(self, code, body=None):
        self.status_code = code
        self.text = "error body"
        self._body = body if body is not None else {"ok": True}

    def json(self):
        return self._body


def _client(responder):
    """Bare client (bypass __init__/network) with a fake session driven by `responder`."""
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.base = "https://demo-fapi.test"
    c.secret = "secret"
    calls = {"n": 0}

    class _Sess:
        def request(self, method, url, **kw):
            calls["n"] += 1
            return responder(calls["n"])

    c.session = _Sess()
    return c, calls


def test_get_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr("hyperbot.binance_exec.time.sleep", lambda *_: None)
    c, calls = _client(lambda n: _Resp(200) if n >= 3 else _Resp(408))  # fail twice, then 200
    assert c._request("GET", "/fapi/v2/positionRisk") == {"ok": True}
    assert calls["n"] == 3                                              # retried, not aborted


def test_get_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("hyperbot.binance_exec.time.sleep", lambda *_: None)
    c, calls = _client(lambda n: _Resp(503))                           # always transient
    with pytest.raises(BinanceFuturesError):
        c._request("GET", "/fapi/v2/balance")
    assert calls["n"] == BinanceFuturesClient._GET_RETRIES             # bounded, then raises


def test_post_never_retries_on_transient(monkeypatch):
    monkeypatch.setattr("hyperbot.binance_exec.time.sleep", lambda *_: None)
    c, calls = _client(lambda n: _Resp(408))                           # order timed out
    with pytest.raises(BinanceFuturesError):
        c._request("POST", "/fapi/v1/order")
    assert calls["n"] == 1                                             # exactly one attempt — no double-fill


def test_get_does_not_retry_on_non_transient(monkeypatch):
    monkeypatch.setattr("hyperbot.binance_exec.time.sleep", lambda *_: None)
    c, calls = _client(lambda n: _Resp(400))                           # client error, not transient
    with pytest.raises(BinanceFuturesError):
        c._request("GET", "/fapi/v2/positionRisk")
    assert calls["n"] == 1
