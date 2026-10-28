# Martingale simulation — methods

**Question:** Does doubling your bet after every loss beat blackjack, or just
reshape the losses?

## Setup

- **Game:** the validated blackjack Monte Carlo engine
  (`../blackjack/engine.py`): infinite-deck approximation, dealer stands on
  soft 17, blackjack pays 3:2, double on any first two cards, double-after-split
  allowed, pairs resplit except aces (one card each), no surrender, no insurance.
- **Player strategy:** full basic strategy, same as the 100M-hand run.
- **Systems compared:** flat $25 every hand vs. Martingale ($25 base, double
  after a loss, reset to $25 after a win, hold the bet after a push, bet capped
  at the $500 table maximum — the progression parks at the max until a win).
- **Session:** $5,000 starting bankroll, 200 hands ("an evening"). A session
  ends early (ruin) when the bankroll cannot cover the next required bet.
- **Scale:** 1,000,000 sessions per system (400M hands total).

## Validation

- Flat-betting mode is an independent re-run of the earlier 100M-hand results:
  expected mean profit ≈ 200 × $25 × (−0.0049769) ≈ −$24.88. The run prints a
  match check against this number; it must agree within $3.
- Card outcomes are bet-size independent, so the same engine draws feed both
  systems; only the bet sizing differs.

## Disclosed simplifications

- The model assumes the bankroll covers split/double side action; with a
  $5,000 bankroll vs. $25 base this only binds in near-ruin states that are
  already counted as ruined.
- Push keeps the same bet (some tables treat a push as progression reset;
  the difference is negligible at this scale and is disclosed).
- Infinite-deck approximation matches 6-deck math within ~0.05% house edge.

## How to read the results

- Betting systems cannot change per-hand expected value; they redistribute
  it. Compare **mean profit** (Martingale's is worse in dollars, because more
  total money goes into action) against **p_ahead** (Martingale wins small
  most evenings) and **p_ruin** (the concentrated tail).
- `p_hit_table_max` shows how often the table limit breaks the progression —
  the mechanism by which the "can't lose" folk claim fails in real casinos.
