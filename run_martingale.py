"""
Martingale betting-system Monte Carlo for the simulation YouTube channel.

Question: does the Martingale progression (double after every loss, reset after
every win) beat the house edge, or just reshape the losses?

Method: reuse the validated blackjack engine (infinite-deck approximation,
S17, 3:2 blackjack, DAS, resplit non-aces, no surrender) with full basic
strategy. Card outcomes are bet-size independent, so per-hand results from
play_hand() are scaled by the current progression bet inside each session.

Betting systems compared, same card streams per session index (different RNG
draws, same parameters):
  - flat:      $25 every hand
  - martingale: $25 base; double after a loss; reset to $25 after a win;
                hold the same bet after a push; bet capped at the table
                maximum ($500) -- the progression stays parked at the max
                until a win; session ends (ruin) when the bankroll cannot
                cover the next required bet.

Parameters (disclosed, chosen to mirror a realistic $25-min table evening):
  - starting bankroll $5,000
  - 200 hands per session ("an evening")
  - base bet $25, table max $500
  - 1,000,000 sessions per system

Caveats (for METHODS.md / the script's honesty standard):
  - The model assumes the bankroll covers split/double side action; at a
    $5,000 bankroll vs $25 base this binds only in near-ruin states, where
    the session is already recorded as ruined.
  - Infinite-deck approximation matches 6-deck math within ~0.05%.
  - No surrender, no insurance, no side bets -- same rules as the blackjack run.
"""
import json
import os
import sys
import time

import numpy as np
import numba

sys.path.insert(0, os.path.dirname(__file__))  # engine.py ships in this repo
from engine import play_hand, BASIC  # noqa: E402

FLAT = 0
MARTINGALE = 1

N_SESSIONS = 1_000_000
HANDS_PER = 200
BANKROLL0 = 5_000.0
BASE = 25.0
TABLE_MAX = 500.0


@numba.njit(parallel=True)
def simulate_sessions(mode, n_sessions, hands_per, bankroll0, base, tmax, out):
    """out columns: final_bankroll, hands_played, ruined, max_bet, total_action."""
    for s in numba.prange(n_sessions):
        bankroll = bankroll0
        bet = base
        ruined = 0.0
        h_played = 0.0
        max_bet = 0.0
        total_action = 0.0
        for _ in range(hands_per):
            if bet > bankroll:
                ruined = 1.0
                break
            profit_u, is_win, is_loss, is_push, _bj = play_hand(BASIC)
            total_action += bet
            bankroll += bet * profit_u
            if bankroll < 0.0:
                bankroll = 0.0
            h_played += 1.0
            if bet > max_bet:
                max_bet = bet
            if mode == MARTINGALE:
                if is_win:
                    bet = base
                elif is_loss:
                    bet = bet * 2.0
                    if bet > tmax:
                        bet = tmax
                # push: keep the same bet
            # flat mode: bet stays at base every hand
        out[s, 0] = bankroll
        out[s, 1] = h_played
        out[s, 2] = ruined
        out[s, 3] = max_bet
        out[s, 4] = total_action


def summarize(final, hands_played, ruined, max_bet, total_action, name):
    n = final.shape[0]
    p_ahead = float(np.mean(final > BANKROLL0))
    p_ruin = float(np.mean(ruined))
    p_broke_even_or_better = float(np.mean(final >= BANKROLL0))
    return {
        "name": name,
        "sessions": n,
        "hands_per_session": HANDS_PER,
        "starting_bankroll": BANKROLL0,
        "base_bet": BASE,
        "table_max": TABLE_MAX,
        "p_ruin": p_ruin,
        "p_ahead": p_ahead,
        "p_broke_even_or_better": p_broke_even_or_better,
        "mean_final_bankroll": float(np.mean(final)),
        "median_final_bankroll": float(np.median(final)),
        "p10_final_bankroll": float(np.percentile(final, 10)),
        "p90_final_bankroll": float(np.percentile(final, 90)),
        "mean_profit_usd": float(np.mean(final - BANKROLL0)),
        "median_profit_usd": float(np.median(final - BANKROLL0)),
        "mean_total_action_usd": float(np.mean(total_action)),
        "mean_hands_played": float(np.mean(hands_played)),
        "median_hands_to_ruin": (
            float(np.median(hands_played[ruined == 1])) if p_ruin > 0 else None
        ),
        "p_hit_table_max": float(np.mean(max_bet >= TABLE_MAX)),
        "mean_max_bet": float(np.mean(max_bet)),
    }


def main():
    t0 = time.time()
    results = {}
    for mode, name in ((FLAT, "flat_25"), (MARTINGALE, "martingale")):
        out = np.zeros((N_SESSIONS, 5), dtype=np.float64)
        ts = time.time()
        simulate_sessions(mode, N_SESSIONS, HANDS_PER, BANKROLL0, BASE, TABLE_MAX, out)
        dt = time.time() - ts
        s = summarize(out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], name)
        s["run_seconds"] = dt
        results[name] = s
        print(f"{name}: p_ruin={s['p_ruin']:.4f} p_ahead={s['p_ahead']:.4f} "
              f"mean_profit=${s['mean_profit_usd']:.2f} "
              f"mean_action=${s['mean_total_action_usd']:.0f} ({dt:.1f}s)")

    # Validation anchor: flat mode must match the 100M-hand run.
    # Expected: mean_profit ~= HANDS_PER * BASE * (-0.004976935) = -$24.88
    flat_ev_check = HANDS_PER * BASE * -0.004976935
    results["validation"] = {
        "flat_mean_profit_expected_usd": flat_ev_check,
        "flat_mean_profit_actual_usd": results["flat_25"]["mean_profit_usd"],
        "match": abs(results["flat_25"]["mean_profit_usd"] - flat_ev_check) < 3.0,
    }
    results["run_seconds_total"] = time.time() - t0

    path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", path, "| validation match:", results["validation"]["match"])


if __name__ == "__main__":
    main()
