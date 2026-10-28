"""
Blackjack Monte Carlo engine for the simulation-format YouTube channel.

Rules (disclosed methodology):
- Infinite-deck approximation: every card drawn i.i.d. with
  P(10-value)=4/13, P(ace)=1/13, P(2..9)=1/13 each.
  (Standard approximation in blackjack mathematics; results match
  6-deck games within a few hundredths of a percent.)
- Dealer stands on soft 17 (S17).
- Blackjack pays 3:2. No even-money side bets.
- Double on any first two cards; double after split allowed (DAS).
- Split pairs; resplit allowed except aces (split aces: one card each, done).
- No surrender.

Card encoding: rank 1..13 (1=ace, 11-13 = J/Q/K).
Value: ace=1 (adjusted to 11 when usable), face cards=10.
"""

import numba
import numpy as np

HIT, STAND, DOUBLE, SPLIT = 0, 1, 2, 3

# Strategy ids
BASIC = 0
BASIC_NO_DS = 1      # basic strategy but never double or split (hit/stand only)
MIMIC_DEALER = 2     # hit until 17, stand on soft 17 (dealer's own rules)
NEVER_BUST = 3       # stand on 12+ (never risk busting)
DEALER_HAS_10 = 4    # folk strategy: assume hole card is a 10, hit until total >= up+10

STRATEGY_NAMES = {
    BASIC: "Basic strategy (full)",
    BASIC_NO_DS: "Basic strategy, no doubles/splits",
    MIMIC_DEALER: "Mimic the dealer",
    NEVER_BUST: "Never bust (stand on 12+)",
    DEALER_HAS_10: "Assume dealer has a 10",
}


@numba.njit
def draw_rank():
    return np.random.randint(1, 14)


@numba.njit
def rank_value(rank):
    if rank == 1:
        return 1
    if rank > 10:
        return 10
    return rank


@numba.njit
def hand_value(cards, n):
    """Returns (total, soft) where soft=True means a usable ace (counted as 11)."""
    total = 0
    aces = 0
    for i in range(n):
        r = cards[i]
        if r == 1:
            aces += 1
            total += 1
        elif r > 10:
            total += 10
        else:
            total += r
    soft = False
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
        soft = True
    return total, soft


@numba.njit
def basic_action(total, soft, pair_rank, up, can_double, can_split):
    """Full basic strategy, 6-deck S17 DAS. up: 2..11 (11 = ace)."""
    if can_split and pair_rank > 0:
        r = pair_rank
        if r == 1 or r == 8:
            return SPLIT
        if r == 10:
            pass  # never split tens: fall through to hard 20
        elif r == 9:
            if up == 2 or up == 3 or up == 4 or up == 5 or up == 6 or up == 8 or up == 9:
                return SPLIT
        elif r == 7:
            if 2 <= up <= 7:
                return SPLIT
        elif r == 6:
            if 2 <= up <= 6:
                return SPLIT
        elif r == 4:
            if up == 5 or up == 6:
                return SPLIT  # with DAS
        elif r == 3 or r == 2:
            if 2 <= up <= 7:
                return SPLIT
        # 5,5: never split -> falls through to hard 10
    if soft:
        if total <= 14:  # A,2 / A,3
            if up == 5 or up == 6:
                return DOUBLE if can_double else HIT
            return HIT
        elif total <= 16:  # A,4 / A,5
            if 4 <= up <= 6:
                return DOUBLE if can_double else HIT
            return HIT
        elif total == 17:  # A,6
            if 3 <= up <= 6:
                return DOUBLE if can_double else HIT
            return HIT
        elif total == 18:  # A,7
            if 3 <= up <= 6:
                return DOUBLE if can_double else STAND
            if up == 2 or up == 7 or up == 8:
                return STAND
            return HIT
        else:  # A,8+
            return STAND
    else:
        if total <= 8:
            return HIT
        if total == 9:
            if 3 <= up <= 6:
                return DOUBLE if can_double else HIT
            return HIT
        if total == 10:
            if 2 <= up <= 9:
                return DOUBLE if can_double else HIT
            return HIT
        if total == 11:
            if up != 11:
                return DOUBLE if can_double else HIT
            return HIT
        if total == 12:
            if 4 <= up <= 6:
                return STAND
            return HIT
        if 13 <= total <= 16:
            if 2 <= up <= 6:
                return STAND
            return HIT
        return STAND  # 17+


@numba.njit
def strategy_action(sid, total, soft, pair_rank, up, can_double, can_split):
    if sid == BASIC:
        return basic_action(total, soft, pair_rank, up, can_double, can_split)
    if sid == BASIC_NO_DS:
        # basic strategy, but doubles become hits and pairs are never split
        a = basic_action(total, soft, 0, up, False, False)
        if a == DOUBLE:
            return HIT
        return a
    if sid == MIMIC_DEALER:
        return HIT if total < 17 else STAND
    if sid == NEVER_BUST:
        return HIT if total <= 11 else STAND
    if sid == DEALER_HAS_10:
        assumed = up + 10  # up==11 (ace) -> 21
        if total <= 11:
            return HIT
        return HIT if total < assumed else STAND
    return STAND


@numba.njit
def play_hand(sid):
    """
    Play one hand. Returns (profit, is_win, is_loss, is_push, is_blackjack).
    Profit is in units of the initial bet.
    """
    # hands storage: up to 8 hands (splits), up to 12 cards each
    cards = np.zeros((8, 12), dtype=np.int64)
    ncards = np.zeros(8, dtype=np.int64)
    bets = np.ones(8, dtype=np.float64)
    nhands = 1

    # deal
    cards[0, 0] = draw_rank()
    cards[0, 1] = draw_rank()
    ncards[0] = 2
    dealer_up = draw_rank()
    dealer_hole = draw_rank()

    up_val = 11 if dealer_up == 1 else (10 if dealer_up > 10 else dealer_up)

    # blackjack checks
    dc = np.zeros(2, dtype=np.int64)
    dc[0] = dealer_up
    dc[1] = dealer_hole
    d_total, _ = hand_value(dc, 2)
    p_total, _ = hand_value(cards[0], 2)
    player_bj = (p_total == 21)
    dealer_bj = (d_total == 21)
    if dealer_bj or player_bj:
        if player_bj and dealer_bj:
            return 0.0, False, False, True, False
        if player_bj:
            return 1.5, True, False, False, True
        return -1.0, False, True, False, False

    # player plays each hand
    h = 0
    while h < nhands:
        n = ncards[h]
        can_double = (n == 2)
        can_split = (n == 2)
        done = False
        split_aces = False
        while not done:
            total, soft = hand_value(cards[h], ncards[h])
            if total > 21:
                done = True
                break
            pair_rank = 0
            if ncards[h] == 2 and cards[h, 0] == cards[h, 1]:
                pair_rank = cards[h, 0]
            act = strategy_action(sid, total, soft, pair_rank, up_val,
                                  can_double, can_split)
            if act == STAND:
                done = True
            elif act == HIT:
                cards[h, ncards[h]] = draw_rank()
                ncards[h] += 1
                can_double = False
                can_split = False
            elif act == DOUBLE:
                cards[h, ncards[h]] = draw_rank()
                ncards[h] += 1
                bets[h] = 2.0
                done = True
            elif act == SPLIT:
                if nhands >= 8:
                    # absurdly rare triple-resplit: just hit instead
                    cards[h, ncards[h]] = draw_rank()
                    ncards[h] += 1
                    can_double = False
                    can_split = False
                    continue
                r = cards[h, 0]
                # hand h keeps first card, new hand gets second
                cards[h, 1] = draw_rank()
                ncards[h] = 2
                nh = nhands
                nhands += 1
                cards[nh, 0] = r
                cards[nh, 1] = draw_rank()
                ncards[nh] = 2
                bets[nh] = 1.0
                if r == 1:
                    # split aces: one card each, no further play, no resplit
                    done = True
                    split_aces = True
                    # mark the new ace hand as done by pre-setting a flag:
                    # we handle it by skipping play when we reach it
                    cards[nh, 2] = -1  # sentinel: ace-split hand, do not play
                can_double = True   # DAS
                can_split = True    # resplit allowed (except aces, handled)
                if split_aces:
                    can_split = False
        # advance to next hand; skip ace-split sentinel hands
        h += 1
        while h < nhands and ncards[h] == 2 and cards[h, 2] == -1:
            h += 1

    # dealer plays (S17)
    dc = np.zeros(12, dtype=np.int64)
    dc[0] = dealer_up
    dc[1] = dealer_hole
    dn = 2
    d_total, d_soft = hand_value(dc, dn)
    while d_total < 17:
        dc[dn] = draw_rank()
        dn += 1
        d_total, d_soft = hand_value(dc, dn)

    # settle
    profit = 0.0
    is_win = False
    is_loss = False
    is_push = False
    for h in range(nhands):
        if ncards[h] == 2 and cards[h, 2] == -1:
            n_ace = 2  # sentinel hand: 2 cards
        total, soft = hand_value(cards[h], ncards[h])
        b = bets[h]
        if total > 21:
            profit -= b
            is_loss = True
        elif d_total > 21:
            profit += b
            is_win = True
        elif total > d_total:
            profit += b
            is_win = True
        elif total < d_total:
            profit -= b
            is_loss = True
        else:
            is_push = True
    return profit, is_win, is_loss, is_push, False


@numba.njit(parallel=True)
def simulate(sid, n_hands, block, out):
    """
    Run n_hands. out: [total_profit, wins, losses, pushes, blackjacks].
    block: block size for session stats; block_profits filled with per-block profit.
    """
    n_blocks = n_hands // block
    for bi in numba.prange(n_blocks):
        bp = 0.0
        w = 0
        l = 0
        p = 0
        bj = 0
        for _ in range(block):
            profit, is_win, is_loss, is_push, is_bj = play_hand(sid)
            bp += profit
            if is_win:
                w += 1
            if is_loss:
                l += 1
            if is_push:
                p += 1
            if is_bj:
                bj += 1
        out[bi, 0] = bp
        out[bi, 1] = w
        out[bi, 2] = l
        out[bi, 3] = p
        out[bi, 4] = bj
