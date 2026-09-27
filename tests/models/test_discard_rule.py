"""Pins the shared keep-rule contract (DISCARD_SPEC.md section B)."""

import itertools
import random

import pytest

from catanatron.models.discard_rule import (
    CITY,
    DEV_CARD,
    DISCARD_ORDER_INDEX,
    DISCARD_ORDERS,
    ROAD,
    SETTLEMENT,
    TARGETS,
    keep_rule,
    keep_rule_discard_sets,
)


def test_orders_are_canonical_permutations():
    assert len(DISCARD_ORDERS) == 24
    assert len(set(DISCARD_ORDERS)) == 24
    assert DISCARD_ORDERS == tuple(
        itertools.permutations(("CITY", "SETTLEMENT", "DEV_CARD", "ROAD"))
    )
    assert DISCARD_ORDERS[0] == (CITY, SETTLEMENT, DEV_CARD, ROAD)
    assert all(DISCARD_ORDERS[i][0] == CITY for i in range(6))
    assert DISCARD_ORDER_INDEX[(ROAD, DEV_CARD, SETTLEMENT, CITY)] == 23
    assert TARGETS == (CITY, SETTLEMENT, DEV_CARD, ROAD)


# Spec test vectors
def test_vector_1_city_first():
    order = (CITY, SETTLEMENT, DEV_CARD, ROAD)
    assert keep_rule((2, 2, 2, 2, 2), 5, order) == (1, 2, 2, 0, 0)


def test_vector_2_single_resource_any_order():
    for order in DISCARD_ORDERS:
        assert keep_rule((8, 0, 0, 0, 0), 4, order) == (4, 0, 0, 0, 0)


def test_vector_3_road_first_wheat_ore():
    # keep_n 4: ROAD nothing; DEV_CARD wheat 1, ore 1; SETTLEMENT wheat 1;
    # CITY wheat 1 => kept (0,0,0,3,1)
    order = (ROAD, DEV_CARD, SETTLEMENT, CITY)
    assert keep_rule((0, 0, 0, 4, 4), 4, order) == (0, 0, 0, 1, 3)


def test_repeat_pass_and_filler():
    order = (ROAD, SETTLEMENT, DEV_CARD, CITY)
    # 9 ore, owed 4 -> keep 5: DEV_CARD ore 1, CITY ore 3, second pass
    # DEV_CARD ore 1 (kept 5). Discard the other 4.
    assert keep_rule((0, 0, 0, 0, 9), 4, order) == (0, 0, 0, 0, 4)
    # keep 3 of (2, 2, 0, 0, 0): ROAD keeps 1 wood 1 brick, SETTLEMENT the
    # 2nd wood (resource order tie-break) => discard 1 brick.
    assert keep_rule((2, 2, 0, 0, 0), 1, order) == (0, 1, 0, 0, 0)
    # (0, 0, 5, 0, 0) owed 1 keep 4: SETTLEMENT sheep 1, DEV_CARD sheep 1,
    # repeat: sheep 1, sheep 1 (kept 4) => discard 1 sheep.
    assert keep_rule((0, 0, 5, 0, 0), 1, order) == (0, 0, 1, 0, 0)


def test_owed_zero_and_full_hand():
    for order in DISCARD_ORDERS:
        assert keep_rule((1, 2, 3, 4, 5), 0, order) == (0, 0, 0, 0, 0)
        assert keep_rule((1, 2, 3, 4, 5), 15, order) == (1, 2, 3, 4, 5)


def test_validation():
    order = DISCARD_ORDERS[0]
    with pytest.raises(ValueError):
        keep_rule((1, 1, 1, 1, 1), 6, order)
    with pytest.raises(ValueError):
        keep_rule((1, 1, 1, 1, 1), -1, order)
    with pytest.raises(ValueError):
        keep_rule((1, 1, 1, 1), 1, order)
    with pytest.raises(ValueError):
        keep_rule((1, 1, 1, 1, 1), 1, (CITY, CITY, ROAD, DEV_CARD))


def test_properties_on_random_hands():
    rng = random.Random(0)
    for _ in range(2000):
        hand = tuple(rng.randint(0, 6) for _ in range(5))
        total = sum(hand)
        if total == 0:
            continue
        owed = rng.randint(0, total)
        order = rng.choice(DISCARD_ORDERS)
        discard = keep_rule(hand, owed, order)
        assert len(discard) == 5
        assert sum(discard) == owed
        assert all(0 <= d <= h for d, h in zip(discard, hand))
        assert all(isinstance(d, int) for d in discard)
        # deterministic
        assert keep_rule(list(hand), owed, list(order)) == discard


def test_keep_rule_discard_sets_are_distinct_and_ordered():
    sets = keep_rule_discard_sets((2, 2, 2, 2, 2), 5)
    assert len(sets) == len(set(sets)) <= 24
    assert sets[0] == (1, 2, 2, 0, 0)  # CITY-first order comes first
    first_seen = {}
    for i, order in enumerate(DISCARD_ORDERS):
        first_seen.setdefault(keep_rule((2, 2, 2, 2, 2), 5, order), i)
    assert sets == sorted(first_seen, key=first_seen.get)
    assert keep_rule_discard_sets((8, 0, 0, 0, 0), 4) == [(4, 0, 0, 0, 0)]
