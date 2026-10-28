# Martingale on Blackjack — Monte Carlo Simulation

**Question:** Does doubling your bet after every loss beat blackjack, or just
reshape the losses?

**Answer in one line:** Martingale wins most evenings and still loses more
money on average — the table maximum breaks the progression.

## Headline results (main run, 1,000,000 sessions per system)

| | Flat $25 | Martingale |
|---|---|---|
| Ahead at end of evening | 46.9% | 54.7% |
| Median profit | -$25 | +$275 |
| Mean profit | -$25.29 | -$80.07 |
| Ruin (cannot cover next bet) | 0% | 9.5% (about 1 in 10.5) |
| Table max ($500) hit | 0% | 97.6% |
| Total wagered per evening | $5,000 | $16,055 |

Setup: $5,000 bankroll, $25 base bet, $500 table maximum, 200 hands per
evening ("an evening"), validated blackjack engine with full basic strategy
(infinite-deck approximation, dealer stands on soft 17, blackjack pays 3:2,
double-after-split allowed, resplit non-aces, no surrender, no insurance).
Flat mean loss validates against the earlier 100M-hand run: expected
-$24.88 vs actual -$25.29 (within a dollar).

## Sensitivity grid

`run_sensitivity.py` varies table max ($500 / $2,000 / $10,000) against
starting bankroll ($5,000 / $50,000), 1,000,000 sessions per cell.
Small-bankroll ruin rises to 41.5% then 51.8% as the cap rises; with a
$50,000 bankroll and a $10,000 cap, mean loss grows to about $152.
A higher limit moves the wall — it does not remove it. Full grid in
`sensitivity_results.json`. Care note: ruined sessions stop early with
bankroll left (ruin = cannot cover the next required bet, not $0), so a
lower mean loss alongside higher ruin is not "better" on its own.

## How to run

```bash
pip install -r requirements.txt
python run_martingale.py        # main run: 1M sessions per system
python run_sensitivity.py       # grid: 6 cells x 1M sessions
```

Requires numba (the engine is JIT-compiled); the main run took ~100s on
the channel's machine. `engine.py` is the validated blackjack engine,
shipped here so this repo runs standalone.

## Files

- `engine.py` — blackjack engine (play_hand, full basic strategy)
- `run_martingale.py` — main run, flat vs Martingale
- `run_sensitivity.py` — table-max x bankroll grid
- `METHODS.md` — setup, validation, disclosed simplifications
- `results.json` — main-run results
- `sensitivity_results.json` — grid results

## Disclaimer

Math experiment, not betting advice. Betting systems cannot change the
per-hand expected value; they redistribute it. Do not use this as a way
to win money.
