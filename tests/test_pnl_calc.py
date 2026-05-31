from hyperbot.pnl_calc import compute_stats


def test_compute_stats_basic():
    result = {
        "initial_equity": 10000.0,
        "final_equity": 10400.0,
        "trades": [
            {"pnl": 500.0, "return_pct": 5.0},
            {"pnl": -100.0, "return_pct": -1.0},
        ],
        "equity_curve": [
            {"time": "t0", "equity": 10000.0},
            {"time": "t1", "equity": 10500.0},
            {"time": "t2", "equity": 10400.0},
        ],
    }
    stats = compute_stats(result)
    assert stats["trades"] == 2
    assert stats["total_pnl"] == 400.0
    assert stats["win_rate"] == 50.0
    assert stats["return_pct"] == 4.0
    assert stats["max_drawdown_pct"] > 0.0