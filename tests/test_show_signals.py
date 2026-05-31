from rich.table import Table

from hyperbot.show_signals import render


def test_render_returns_table_with_rows():
    result = {
        "trades": [
            {
                "side": "LONG", "entry_time": "t0", "entry_price": 100.0, "exit_price": 110.0,
                "pnl": 10.0, "buy_confidence": 80.0, "sell_confidence": 0.0,
                "regime": "trending", "entry_reason": "x",
            }
        ]
    }
    table = render(result)
    assert isinstance(table, Table)
    assert table.row_count == 1