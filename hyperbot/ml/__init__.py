"""Offline meta-labeling research layer.

The ungated 5-strategy vote is the PRIMARY signal generator (buy/sell on every
bar behind the EMA800 filter). These modules build a labeled dataset from that
vote's own win/loss history, train a meta-model to predict P(win) per signal,
and evaluate — walk-forward, out-of-sample — whether gating by that probability
beats the raw ungated baseline. Research only: nothing here places orders.
"""
