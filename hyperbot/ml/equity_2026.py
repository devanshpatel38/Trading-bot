"""BTC 2026 equity curve under the logistic tercile-SIZING policy.

Out-of-sample: the logistic is trained on BTC signals BEFORE 2026 and scores each
2026 signal. Tercile cutoffs are the 33rd/67th percentiles of the model's
predictions on that TRAINING set (known before 2026 — no lookahead). Sizing:

    low tercile   -> SKIP
    mid tercile   -> risk max($100, 2% of capital)
    high tercile  -> risk max($250, 5% of capital)

Capital compounds from $5000; r_multiple is already net of fees + slippage. A naive
"take every signal at flat 5%" curve is computed alongside as a reference.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset
from .research import _fit_logistic

def run(start_capital: float = 5000.0, year: int = 2026):
    if os.path.exists(CACHE_PATH):
        data = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])
    else:
        data = build_dataset()
    data = data.sort_values("entry_time").reset_index(drop=True)
    data["year"] = data["entry_time"].dt.year

    train = data[data["year"] < year]
    test = data[data["year"] == year].copy()

    model = _fit_logistic(train[FEATURE_COLS].values, train["y"].values)
    p_train = model.predict_proba(train[FEATURE_COLS].values)[:, 1]
    q33, q67 = np.quantile(p_train, [1 / 3, 2 / 3])  # tercile cutoffs from TRAIN only
    test["p"] = model.predict_proba(test[FEATURE_COLS].values)[:, 1]

    def tercile(p):
        return "low" if p < q33 else ("mid" if p < q67 else "high")

    cap = start_capital
    naive = start_capital
    curve = [{"date": str(test["entry_time"].min())[:10], "equity": round(cap, 2),
              "naive": round(naive, 2), "tercile": "start", "size": 0.0, "r": 0.0, "pnl": 0.0}]
    peak, mdd = cap, 0.0
    taken = {"mid": 0, "high": 0}
    skipped = 0
    wins = 0

    for _, t in test.iterrows():
        ter = tercile(t["p"])
        r = float(t["r_multiple"])
        # naive reference: take EVERY signal at flat 5% floored at $250
        naive += max(250.0, 0.05 * naive) * r

        if ter == "low":
            skipped += 1
            risk, pnl = 0.0, 0.0          # skip: equity carries flat this signal
        else:
            if ter == "mid":
                risk = max(100.0, 0.02 * cap)
            else:
                risk = max(250.0, 0.05 * cap)
            pnl = risk * r
            cap += pnl
            taken[ter] += 1
            wins += r > 0
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        curve.append({"date": str(t["entry_time"])[:16], "equity": round(cap, 2),
                      "naive": round(naive, 2), "tercile": ter, "p": round(float(t["p"]), 3),
                      "size": round(risk, 2), "r": round(r, 2), "pnl": round(pnl, 2)})

    n_taken = taken["mid"] + taken["high"]
    summary = {
        "year": year,
        "start_capital": start_capital,
        "final_equity": round(cap, 2),
        "return_pct": round((cap / start_capital - 1) * 100, 1),
        "naive_final": round(naive, 2),
        "naive_return_pct": round((naive / start_capital - 1) * 100, 1),
        "max_drawdown_pct": round(mdd * 100, 1),
        "n_signals": len(test),
        "n_taken": n_taken, "n_mid": taken["mid"], "n_high": taken["high"],
        "n_skipped_low": skipped,
        "win_rate_taken": round(wins / n_taken * 100, 1) if n_taken else 0.0,
        "q33": round(float(q33), 3), "q67": round(float(q67), 3),
    }
    return curve, summary


def main():
    import argparse
    p = argparse.ArgumentParser(description="BTC equity curve under the logistic tercile-sizing policy.")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--start-capital", type=float, default=5000.0)
    args = p.parse_args()

    curve, s = run(start_capital=args.start_capital, year=args.year)
    out_json = f"data/equity_{args.year}.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump({"curve": curve, "summary": s}, fh, indent=2)

    print(f"BTC {args.year} — logistic tercile-sizing policy (OOS)")
    print(f"  tercile cutoffs (from pre-{args.year} train): low<{s['q33']}  mid<{s['q67']}  high>=")
    print(f"  {args.year} signals {s['n_signals']} | skipped(low) {s['n_skipped_low']} | "
          f"taken {s['n_taken']} (mid {s['n_mid']}, high {s['n_high']}) | WR {s['win_rate_taken']}%")
    print(f"  ${s['start_capital']:,.0f} -> ${s['final_equity']:,.0f}  "
          f"({s['return_pct']:+.1f}%)  | max DD {s['max_drawdown_pct']:.1f}%")
    print(f"  naive take-all @flat5%: ${s['naive_final']:,.0f} ({s['naive_return_pct']:+.1f}%)")
    print(f"  saved -> {out_json}")
    # compact trade log
    print("\n  #  date              ter   P    size$   R      equity")
    for i, p in enumerate(curve[1:], 1):
        print(f"  {i:>2} {p['date']:<16} {p['tercile']:<4} {p.get('p',0):.2f} "
              f"{p['size']:>7.0f} {p['r']:>+5.2f} {p['equity']:>10,.0f}")


if __name__ == "__main__":
    main()
