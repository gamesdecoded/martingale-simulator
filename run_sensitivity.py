"""Sensitivity grid for Martingale, following the v2 verify list item 8.
Martingale only, full basic strategy, same validated blackjack engine.
Vary table max ($500 / $2,000 / $10,000) x starting bankroll ($5,000 / $50,000).
Base bet $25, 200 hands per evening, 1,000,000 sessions per cell.
Ruin = bankroll cannot cover the next required bet (same as run_martingale.py).
Cap hit = a bet at the table max was actually placed.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_martingale import simulate_sessions, MARTINGALE

N_SESSIONS = 1_000_000
HANDS_PER = 200
BASE = 25.0


def summarize_grid(final, hands_played, ruined, max_bet, total_action, bankroll0, tmax):
    n = final.shape[0]
    return {
        "sessions": n,
        "starting_bankroll": bankroll0,
        "table_max": tmax,
        "base_bet": BASE,
        "hands_per_session": HANDS_PER,
        "p_ruin": float(np.mean(ruined)),
        "p_ahead": float(np.mean(final > bankroll0)),
        "mean_profit_usd": float(np.mean(final - bankroll0)),
        "median_profit_usd": float(np.median(final - bankroll0)),
        "mean_total_action_usd": float(np.mean(total_action)),
        "mean_hands_played": float(np.mean(hands_played)),
        "median_hands_to_ruin": (
            float(np.median(hands_played[ruined == 1])) if np.mean(ruined) > 0 else None
        ),
        "p_hit_table_max": float(np.mean(max_bet >= tmax)),
        "mean_max_bet": float(np.mean(max_bet)),
    }


def main():
    results = {}
    for tmax in (500.0, 2000.0, 10000.0):
        for bankroll0 in (5000.0, 50000.0):
            key = f"cap{int(tmax)}_bank{int(bankroll0)}"
            out = np.zeros((N_SESSIONS, 5), dtype=np.float64)
            t0 = time.time()
            simulate_sessions(MARTINGALE, N_SESSIONS, HANDS_PER, bankroll0, BASE, tmax, out)
            s = summarize_grid(out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], bankroll0, tmax)
            s["run_seconds"] = time.time() - t0
            results[key] = s
            print(f"{key}: p_ruin={s['p_ruin']:.4f} p_ahead={s['p_ahead']:.4f} "
                  f"mean_profit=${s['mean_profit_usd']:.2f} "
                  f"p_cap={s['p_hit_table_max']:.4f} mean_action=${s['mean_total_action_usd']:.0f} "
                  f"({s['run_seconds']:.1f}s)", flush=True)
    path = os.path.join(os.path.dirname(__file__), "sensitivity_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
