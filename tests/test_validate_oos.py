from hyperbot.validate_oos import equity_curve


def test_equity_curve_compounds_and_tracks_drawdown():
    # one +2R win then one -1R loss at 10% risk, $100 start
    trades = [{"r_multiple": 2.0}, {"r_multiple": -1.0}]
    final, maxdd = equity_curve(trades, risk=0.10)
    # 100 -> 100*(1+0.2)=120 -> 120*(1-0.1)=108 ; peak 120 -> dd (120-108)/120=10%
    assert round(final, 2) == 108.00
    assert round(maxdd, 2) == 10.00
