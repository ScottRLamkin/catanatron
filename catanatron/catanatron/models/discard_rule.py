"""The "keep rule": a deterministic heuristic that picks which cards to
discard on a 7 by keeping the cards needed for a prioritized list of
purchases ("pick the top priority thing I want to buy, keep the relevant
cards, then repeat").

This is *not* an engine rule: the engine accepts any legal discard set (see
``catanatron.models.actions.discard_possibilities``). It is a shared policy
used by the engine bots (to prune the discard branching factor), the gym
(action space) and by CatanRL (``catanrl.policies.planning.choose_discard_set``).
Both implementations must agree exactly, so keep this file in sync with the
contract in DISCARD_SPEC.md.

Freqdecks are ``(WOOD, BRICK, SHEEP, WHEAT, ORE)`` tuples.
"""

import itertools
from typing import Sequence, Tuple

CITY = "CITY"
SETTLEMENT = "SETTLEMENT"
DEV_CARD = "DEV_CARD"
ROAD = "ROAD"

TARGETS = (CITY, SETTLEMENT, DEV_CARD, ROAD)

# Canonical enumeration of the 24 priority orders (CITY-first orders come
# first). Action tables (gym, bots) index into this tuple.
DISCARD_ORDERS: Tuple[Tuple[str, str, str, str], ...] = tuple(
    itertools.permutations(TARGETS)
)
DISCARD_ORDER_INDEX = {order: i for i, order in enumerate(DISCARD_ORDERS)}

# Costs in (WOOD, BRICK, SHEEP, WHEAT, ORE) order.
TARGET_COSTS = {
    CITY: (0, 0, 0, 2, 3),
    SETTLEMENT: (1, 1, 1, 1, 0),
    DEV_CARD: (0, 0, 1, 1, 1),
    ROAD: (1, 1, 0, 0, 0),
}


def keep_rule(
    hand_freqdeck: Sequence[int], owed: int, order: Sequence[str]
) -> Tuple[int, int, int, int, int]:
    """Discard freqdeck for a player holding `hand_freqdeck` that owes `owed`
    cards, keeping cards for the purchases in `order` (a permutation of
    TARGETS), first purchase first, repeated until the keep budget is used.

    Deterministic, integers only. The result sums to `owed` and never
    exceeds the hand in any resource.

    Raises:
        ValueError: if `owed` is negative or larger than the hand, or `order`
            is not a permutation of TARGETS.
    """
    hand = tuple(int(n) for n in hand_freqdeck)
    if len(hand) != 5 or any(n < 0 for n in hand):
        raise ValueError(f"hand must be a non-negative 5-freqdeck, got {hand}")
    if tuple(order) not in DISCARD_ORDER_INDEX:
        raise ValueError(f"order must be a permutation of {TARGETS}, got {order}")
    total = sum(hand)
    if owed < 0 or owed > total:
        raise ValueError(f"owed must be in [0, {total}], got {owed}")

    keep_n = total - owed
    kept = [0] * 5
    avail = list(hand)
    num_kept = 0

    done = num_kept == keep_n
    while not done:  # "repeat"
        progress = False
        for target in order:  # priority order
            cost = TARGET_COSTS[target]
            for r in range(5):  # resource order tie-break
                take = min(cost[r], avail[r], keep_n - num_kept)
                if take > 0:
                    kept[r] += take
                    avail[r] -= take
                    num_kept += take
                    progress = True
                if num_kept == keep_n:
                    done = True
                    break
            if done:
                break
        if not progress:
            break

    # Fill any remaining keep slots with leftover cards, most-held first
    # (ties go to the lower resource index).
    while num_kept < keep_n:
        r = max(range(5), key=lambda i: (avail[i], -i))
        kept[r] += 1
        avail[r] -= 1
        num_kept += 1

    return tuple(hand[r] - kept[r] for r in range(5))


def keep_rule_discard_sets(hand_freqdeck: Sequence[int], owed: int):
    """The distinct discard sets produced by the 24 orders, in DISCARD_ORDERS
    order of first appearance (at most 24, typically far fewer)."""
    seen = set()
    result = []
    for order in DISCARD_ORDERS:
        discard = keep_rule(hand_freqdeck, owed, order)
        if discard not in seen:
            seen.add(discard)
            result.append(discard)
    return result
