from hyperbot.pnl_calc import compute_stats


def _result(*trade_tuples):
    """Build a minimal result dict from (outcome, r_multiple, bars_held) tuples."""
    trades = [
        {"outcome": outcome, "r_multiple": r_multiple, "bars_held": bars_held}
        for outcome, r_multiple, bars_held in trade_tuples
    ]
    return {"trades": trades}


def test_compute_stats_basic_win_loss():
    result = _result(("win", 2.0, 4), ("loss", -1.0, 2))
    stats = compute_stats(result)
    assert stats["trades"] == 2
    assert stats["resolved"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["open"] == 0
    assert stats["win_rate"] == 50.0
    assert stats["total_r"] == 1.0
    assert stats["expectancy_r"] == 0.5
    assert stats["avg_bars_held"] == 3.0
    assert stats["best_r"] == 2.0
    assert stats["worst_r"] == -1.0


def test_compute_stats_with_open_trade():
    result = _result(("win", 2.0, 5), ("loss", -1.0, 3), ("open", 0.0, 7))
    stats = compute_stats(result)
    assert stats["trades"] == 3
    assert stats["resolved"] == 2
    assert stats["open"] == 1
    assert stats["win_rate"] == 50.0
    # open trade excluded from R stats
    assert stats["total_r"] == 1.0
    assert stats["avg_bars_held"] == 4.0


def test_compute_stats_all_wins():
    result = _result(("win", 2.0, 10), ("win", 2.0, 6))
    stats = compute_stats(result)
    assert stats["win_rate"] == 100.0
    assert stats["total_r"] == 4.0
    assert stats["expectancy_r"] == 2.0
    assert stats["best_r"] == 2.0
    assert stats["worst_r"] == 2.0


def test_compute_stats_no_trades():
    stats = compute_stats({"trades": []})
    assert stats["trades"] == 0
    assert stats["resolved"] == 0
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["open"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["total_r"] == 0.0
    assert stats["expectancy_r"] == 0.0
    assert stats["avg_bars_held"] == 0.0
    assert stats["best_r"] == 0.0
    assert stats["worst_r"] == 0.0


def test_compute_stats_only_open():
    result = _result(("open", 0.0, 3))
    stats = compute_stats(result)
    assert stats["trades"] == 1
    assert stats["resolved"] == 0
    assert stats["open"] == 1
    assert stats["win_rate"] == 0.0
    assert stats["total_r"] == 0.0
    assert stats["expectancy_r"] == 0.0
    assert stats["avg_bars_held"] == 0.0