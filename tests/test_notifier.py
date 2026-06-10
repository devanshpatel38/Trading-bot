import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperbot.notifier import load_state, save_state, send_ntfy, get_current_state, notify


# --- load_state ---

def test_load_state_returns_default_when_no_file(tmp_path):
    result = load_state(tmp_path / "missing.json")
    assert result == {"state": "flat", "trade": None}


def test_load_state_reads_existing_file(tmp_path):
    p = tmp_path / "state.json"
    data = {
        "state": "open",
        "trade": {"side": "short", "entry": 63440.0,
                  "entry_time": "2026-06-08 18:00:00",
                  "stop": 65084.21, "tp": 58507.36},
    }
    p.write_text(json.dumps(data))
    result = load_state(p)
    assert result["state"] == "open"
    assert result["trade"]["entry"] == 63440.0


# --- save_state ---

def test_save_state_writes_file(tmp_path):
    p = tmp_path / "state.json"
    state = {"state": "flat", "trade": None}
    save_state(state, p)
    assert json.loads(p.read_text()) == state


def test_save_state_overwrites_existing(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"state": "open", "trade": {}}')
    save_state({"state": "flat", "trade": None}, p)
    assert json.loads(p.read_text())["state"] == "flat"


# --- send_ntfy ---

def test_send_ntfy_posts_to_correct_url():
    with patch("hyperbot.notifier.requests.post") as mock_post:
        send_ntfy("Test Title", "Test body", "my-topic", "high")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://ntfy.sh/my-topic"
    assert kwargs["headers"]["Title"] == "Test Title"
    assert kwargs["headers"]["Priority"] == "high"
    assert kwargs["data"] == b"Test body"


def test_send_ntfy_non_fatal_on_network_error():
    with patch("hyperbot.notifier.requests.post", side_effect=Exception("timeout")):
        send_ntfy("title", "body", "topic")  # must not raise


# --- get_current_state ---

def test_get_current_state_returns_open_when_open_trade_exists():
    trades = [
        {"outcome": "win", "entry_time": "t1", "side": "long", "entry": 100.0,
         "exit_price": 106.0, "r_multiple": 2.0, "stop": 94.0, "tp": 106.0},
        {"outcome": "open", "entry_time": "t2", "side": "short", "entry": 90.0,
         "stop": 92.0, "tp": 84.0, "bars_held": 3},
    ]
    state, trade, last_resolved = get_current_state(trades)
    assert state == "open"
    assert trade["entry"] == 90.0
    assert last_resolved["entry"] == 100.0


def test_get_current_state_returns_flat_when_all_resolved():
    trades = [
        {"outcome": "win", "entry_time": "t1", "side": "long", "entry": 100.0,
         "exit_price": 106.0, "r_multiple": 2.0, "stop": 94.0, "tp": 106.0},
    ]
    state, trade, last_resolved = get_current_state(trades)
    assert state == "flat"
    assert trade is None
    assert last_resolved["outcome"] == "win"


def test_get_current_state_returns_flat_on_empty_trades():
    state, trade, last_resolved = get_current_state([])
    assert state == "flat"
    assert trade is None
    assert last_resolved is None


# --- notify ---

def _open_trade(entry_time="2026-06-08 18:00:00"):
    return {"outcome": "open", "entry_time": entry_time, "side": "short",
            "entry": 63440.0, "stop": 65084.21, "tp": 58507.36, "bars_held": 5}


def _resolved_trade(entry_time="2026-06-08 18:00:00", outcome="win"):
    return {
        "outcome": outcome, "entry_time": entry_time, "side": "short",
        "entry": 63440.0, "stop": 65084.21, "tp": 58507.36,
        "exit_price": 58507.36 if outcome == "win" else 65084.21,
        "r_multiple": 2.96 if outcome == "win" else -1.0,
        "bars_held": 20,
    }


def _new_signal_trade():
    return {"side": "long", "entry": 65000.0, "entry_time": "2026-06-10 10:00:00",
            "stop": 62000.0, "tp": 74000.0}


def test_notify_flat_to_new_signal_sends_high_priority_notification():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], _new_signal_trade(), topic="test-topic")
    mock_ntfy.assert_called_once()
    title, body, *rest = mock_ntfy.call_args[0]
    assert "LONG" in title
    assert "65000" in body
    assert mock_ntfy.call_args[1].get("priority") == "high" or (
        len(mock_ntfy.call_args[0]) > 3 and mock_ntfy.call_args[0][3] == "high"
    )
    assert new_state["state"] == "open"
    assert new_state["trade"]["entry"] == 65000.0


def test_notify_open_to_flat_sends_win_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    trades = [_resolved_trade(entry_time=entry_time, outcome="win")]
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, trades, None, topic="test-topic")
    mock_ntfy.assert_called_once()
    title = mock_ntfy.call_args[0][0]
    assert "WIN" in title
    assert new_state["state"] == "flat"


def test_notify_open_to_flat_sends_loss_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    trades = [_resolved_trade(entry_time=entry_time, outcome="loss")]
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, trades, None, topic="test-topic")
    mock_ntfy.assert_called_once()
    title = mock_ntfy.call_args[0][0]
    assert "LOSS" in title
    assert new_state["state"] == "flat"


def test_notify_open_stays_open_no_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [_open_trade(entry_time)], None, topic="test-topic")
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "open"


def test_notify_flat_stays_flat_no_notification():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], None, topic="test-topic")
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "flat"


def test_notify_no_topic_updates_state_without_sending():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], _new_signal_trade(), topic=None)
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "open"  # state still advances even in dry-run