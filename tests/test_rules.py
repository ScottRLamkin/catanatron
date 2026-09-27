"""Tests for exact Settlers of Catan rules: player-chosen discards, domestic
trade fixes, bank shortage, longest road recount, win-only-on-own-turn and
seedable maps."""

import random
from unittest.mock import patch

import pytest

from catanatron.apply_action import yield_resources
from catanatron.game import Game, is_valid_action, is_valid_trade
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import Board
from catanatron.models.decks import freqdeck_count, freqdeck_draw, starting_resource_bank
from catanatron.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    SHEEP,
    WHEAT,
    WOOD,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.state import State
from catanatron.state_functions import (
    build_road,
    build_settlement,
    get_actual_victory_points,
    get_longest_road_color,
    get_player_freqdeck,
    maintain_longest_road,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_key,
    player_num_resource_cards,
)


def advance_to_play_turn(game):
    while not any(a.action_type == ActionType.ROLL for a in game.playable_actions):
        game.play_tick()


def set_hand(state, color, freqdeck):
    player_freqdeck_subtract(state, color, get_player_freqdeck(state, color))
    for resource, amount in zip(RESOURCES, freqdeck):
        player_deck_replenish(state, color, resource, amount)


# ===== Discard
def discard_values(game):
    assert all(a.action_type == ActionType.DISCARD for a in game.playable_actions)
    return [a.value for a in game.playable_actions]


@patch("catanatron.apply_action.roll_dice")
def test_discard_owed_counts(fake_roll_dice):
    fake_roll_dice.return_value = (3, 4)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    advance_to_play_turn(game)
    p0, p1, p2 = game.state.colors

    set_hand(game.state, p0, [8, 0, 0, 0, 0])  # 8 => owes 4
    set_hand(game.state, p1, [7, 0, 0, 0, 0])  # 7 => nothing (limit is >7)
    set_hand(game.state, p2, [3, 3, 3, 0, 0])  # 9 => owes 4
    game.play_tick()  # roll 7

    owed = [game.state.player_state[f"P{i}_DISCARD_OWED"] for i in range(3)]
    assert owed == [4, 0, 4]
    assert game.state.is_discarding
    assert game.state.current_color() == p0  # roller first
    assert game.state.pending_discards == {}


@patch("catanatron.apply_action.roll_dice")
def test_discard_limit_is_respected(fake_roll_dice):
    fake_roll_dice.return_value = (3, 4)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, discard_limit=9)
    advance_to_play_turn(game)
    p0, p1 = game.state.colors
    set_hand(game.state, p1, [9, 0, 0, 0, 0])  # 9 => nothing (limit 9)
    set_hand(game.state, p0, [10, 0, 0, 0, 0])  # 10 => owes 5
    game.play_tick()  # roll 7

    assert game.state.player_state["P1_DISCARD_OWED"] == 0
    assert game.state.player_state["P0_DISCARD_OWED"] == 5
    assert game.state.current_color() == p0
    assert game.state.current_prompt == ActionPrompt.DISCARD
    assert discard_values(game) == [(5, 0, 0, 0, 0)]
    game.play_tick()
    assert get_player_freqdeck(game.state, p0) == [5, 0, 0, 0, 0]
    assert get_player_freqdeck(game.state, p1) == [9, 0, 0, 0, 0]
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER


def test_discard_possibilities_enumerate_all_sets():
    """One action per multiset of size owed drawn from the hand."""
    from catanatron.models.actions import discard_possibilities, iter_discard_freqdecks

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    p0 = state.colors[0]
    set_hand(state, p0, [1, 1, 0, 3, 3])
    state.player_state["P0_DISCARD_OWED"] = 4
    actions = discard_possibilities(state, p0)
    values = [a.value for a in actions]
    assert all(a.action_type == ActionType.DISCARD and a.color == p0 for a in actions)
    assert len(values) == len(set(values)) == 14
    assert all(sum(v) == 4 and all(0 <= n <= h for n, h in zip(v, [1, 1, 0, 3, 3])) for v in values)
    assert (0, 0, 0, 1, 3) in values and (1, 1, 0, 2, 0) in values
    assert values == sorted(values)  # lexicographic, deterministic

    # 20-card hand owing 10 has C(14,4) = 1001 sets when every resource is deep
    assert len(list(iter_discard_freqdecks((10, 10, 10, 10, 10), 10))) == 1001
    assert len(list(iter_discard_freqdecks((4, 4, 4, 4, 4), 10))) == 381
    assert list(iter_discard_freqdecks((1, 0, 0, 0, 0), 2)) == []
    assert list(iter_discard_freqdecks((1, 2, 0, 0, 0), 0)) == [(0, 0, 0, 0, 0)]


@patch("catanatron.apply_action.roll_dice")
def test_simultaneous_hidden_discard_flow(fake_roll_dice):
    fake_roll_dice.return_value = (3, 4)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    advance_to_play_turn(game)
    # make p1 the roller so we test wrap-around order (p1, p2, p0)
    game.state.current_player_index = 1
    game.state.current_turn_index = 1
    game.playable_actions = generate_playable_actions(game.state)
    p0, p1, p2 = game.state.colors

    set_hand(game.state, p0, [2, 2, 2, 2, 0])  # 8 => 4
    set_hand(game.state, p1, [0, 0, 0, 0, 0])  # nothing
    set_hand(game.state, p2, [1, 1, 0, 3, 3])  # 8 => 4
    game.play_tick()  # p1 rolls 7
    bank_before = game.state.resource_freqdeck.copy()

    # p2 first (seat after roller, skipping p1 who owes nothing)
    assert game.state.current_color() == p2
    assert game.state.current_prompt == ActionPrompt.DISCARD
    values = discard_values(game)
    assert len(values) == 14  # multisets of size 4 within (1,1,0,3,3)
    assert all(a.color == p2 for a in game.playable_actions)

    # validation: wrong size / not held / not a freqdeck
    for bad in [(0, 0, 0, 1, 2), (0, 0, 0, 3, 3), (0, 0, 1, 1, 2), WOOD, (1, 1, 1, 1), (2, 0, 0, 1, 1), (-1, 0, 0, 2, 3)]:
        with pytest.raises(ValueError):
            game.execute(Action(p2, ActionType.DISCARD, bad))
        with pytest.raises(ValueError):
            game.execute(Action(p2, ActionType.DISCARD, bad), validate_action=False)
    # a non-owing player can't discard
    with pytest.raises(ValueError):
        game.execute(Action(p1, ActionType.DISCARD, (0, 0, 0, 0, 0)), validate_action=False)

    game.execute(Action(p2, ActionType.DISCARD, (0, 0, 0, 1, 3)))
    # hidden & pending: nothing moved yet
    assert game.state.pending_discards == {p2: (0, 0, 0, 1, 3)}
    assert get_player_freqdeck(game.state, p2) == [1, 1, 0, 3, 3]
    assert game.state.resource_freqdeck == bank_before
    assert game.state.player_state["P2_DISCARD_OWED"] == 4  # public, unchanged
    assert game.state.player_state["P2_DISCARD_SUBMITTED"]
    # can't submit twice
    with pytest.raises(ValueError):
        game.execute(Action(p2, ActionType.DISCARD, (0, 0, 0, 1, 3)), validate_action=False)

    # now p0 (wrap-around): its view of p2's hand count is unchanged and the
    # earlier choice is not in the JSON it might be shown
    assert game.state.current_color() == p0
    assert game.state.current_prompt == ActionPrompt.DISCARD
    assert player_num_resource_cards(game.state, p2) == 8
    import json as json_module
    from catanatron.json import GameEncoder

    serialized = json_module.loads(json_module.dumps(game, cls=GameEncoder))
    assert "pending_discards" not in serialized
    assert serialized["player_state"]["P2_WHEAT_IN_HAND"] == 3
    assert serialized["player_state"]["P2_DISCARD_OWED"] == 4
    # The submitted set lives only in pending_discards and the (deterministic)
    # action record; a UI/JSON layer must redact DISCARD values in the log
    # while state.is_discarding (follow-up work, see json.py).
    assert game.state.action_records[-1].action == Action(p2, ActionType.DISCARD, (0, 0, 0, 1, 3))
    assert game.state.action_records[-1].result is None

    game.execute(Action(p0, ActionType.DISCARD, (2, 2, 0, 0, 0)))

    # all applied at once, in seat order irrelevant
    assert game.state.pending_discards == {}
    assert get_player_freqdeck(game.state, p2) == [1, 1, 0, 2, 0]
    assert get_player_freqdeck(game.state, p0) == [0, 0, 2, 2, 0]
    assert game.state.resource_freqdeck == [
        b + d for b, d in zip(bank_before, [2, 2, 0, 1, 3])
    ]
    assert all(game.state.player_state[f"P{i}_DISCARD_OWED"] == 0 for i in range(3))
    assert not any(game.state.player_state[f"P{i}_DISCARD_SUBMITTED"] for i in range(3))

    # back to roller moving robber
    assert not game.state.is_discarding
    assert game.state.is_moving_knight
    assert game.state.current_color() == p1
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    discards = [ar for ar in game.state.action_records if ar.action.action_type == ActionType.DISCARD]
    assert len(discards) == 2
    assert all(ar.result is None and len(ar.action.value) == 5 for ar in discards)


@patch("catanatron.apply_action.roll_dice")
def test_discard_prompt_order_from_roller_skips_non_owing(fake_roll_dice):
    fake_roll_dice.return_value = (3, 4)
    players = [SimplePlayer(c) for c in [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]]
    game = Game(players)
    advance_to_play_turn(game)
    game.state.current_player_index = 2
    game.state.current_turn_index = 2
    game.playable_actions = generate_playable_actions(game.state)
    p0, p1, p2, p3 = game.state.colors
    set_hand(game.state, p0, [8, 0, 0, 0, 0])
    set_hand(game.state, p1, [0, 0, 0, 0, 0])
    set_hand(game.state, p2, [0, 0, 0, 0, 0])  # roller, owes nothing
    set_hand(game.state, p3, [0, 0, 0, 0, 9])
    game.play_tick()  # p2 rolls 7

    order = []
    while game.state.is_discarding:
        order.append(game.state.current_color())
        game.play_tick()
    assert order == [p3, p0]
    assert game.state.current_color() == p2
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert player_num_resource_cards(game.state, p0) == 4
    assert player_num_resource_cards(game.state, p3) == 5


def test_pending_discards_are_copied():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    p0 = state.colors[0]
    state.player_state["P0_DISCARD_OWED"] = 3
    state.pending_discards[p0] = (1, 1, 1, 0, 0)
    copy = state.copy()
    assert copy.player_state["P0_DISCARD_OWED"] == 3
    assert copy.pending_discards == {p0: (1, 1, 1, 0, 0)}
    copy.player_state["P0_DISCARD_OWED"] = 0
    copy.pending_discards.clear()
    assert state.player_state["P0_DISCARD_OWED"] == 3
    assert state.pending_discards == {p0: (1, 1, 1, 0, 0)}


def test_bots_prune_discards_to_keep_rule_sets():
    from catanatron.models.discard_rule import keep_rule_discard_sets
    from catanatron.players.tree_search_utils import prune_bot_actions, prune_discard_actions

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    p0 = state.colors[0]
    set_hand(state, p0, [4, 4, 4, 4, 4])
    state.player_state["P0_DISCARD_OWED"] = 10
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    actions = generate_playable_actions(state)
    assert len(actions) == 381
    pruned = prune_discard_actions(actions, state)
    expected = keep_rule_discard_sets([4, 4, 4, 4, 4], 10)
    assert [a.value for a in pruned] == expected
    assert 1 <= len(pruned) <= 24
    assert all(a in actions for a in pruned)
    assert prune_bot_actions(actions, state) == pruned
    # non-discard lists pass through
    other = [Action(p0, ActionType.END_TURN, None)]
    assert prune_discard_actions(other, state) == other


# ===== Robber
def test_robber_victims_in_seat_order():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    advance_to_play_turn(game)
    for color in game.state.colors:
        player_deck_replenish(game.state, color, WOOD, 1)
    game.state.current_prompt = ActionPrompt.MOVE_ROBBER
    actions = generate_playable_actions(game.state)
    by_tile = {}
    for a in actions:
        by_tile.setdefault(a.value[0], []).append(a.value[1])
    for victims in by_tile.values():
        indexes = [game.state.color_to_index[v] for v in victims if v is not None]
        assert indexes == sorted(indexes)


# ===== Domestic Trade
@patch("catanatron.apply_action.roll_dice")
def test_trade_responders_exclude_offerer_at_index_1(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    advance_to_play_turn(game)
    p0, p1, p2 = game.state.colors
    game.play_tick()  # p0 rolls
    game.execute(Action(p0, ActionType.END_TURN, None))
    assert game.state.current_color() == p1
    game.play_tick()  # p1 rolls

    set_hand(game.state, p1, [1, 0, 0, 0, 0])
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    game.execute(Action(p1, ActionType.OFFER_TRADE, offer))
    assert game.state.current_trade == (*offer, 1)
    assert game.state.current_color() == p0
    game.execute(Action(p0, ActionType.REJECT_TRADE, game.state.current_trade))
    assert game.state.current_color() == p2  # offerer is skipped
    game.execute(Action(p2, ActionType.REJECT_TRADE, game.state.current_trade))
    assert not game.state.is_resolving_trade
    assert game.state.current_color() == p1
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN

    # now with an acceptee: p0 rejects, p2 accepts => p1 decides acceptees
    game.state.turn_trade_offers = ()
    set_hand(game.state, p2, [0, 1, 0, 0, 0])
    game.execute(Action(p1, ActionType.OFFER_TRADE, offer))
    game.execute(Action(p0, ActionType.REJECT_TRADE, game.state.current_trade))
    assert game.state.current_color() == p2
    game.execute(Action(p2, ActionType.ACCEPT_TRADE, game.state.current_trade))
    assert game.state.current_color() == p1
    assert game.state.current_prompt == ActionPrompt.DECIDE_ACCEPTEES
    assert Action(p1, ActionType.CONFIRM_TRADE, (*offer, p2)) in game.playable_actions
    game.execute(Action(p1, ActionType.CONFIRM_TRADE, (*offer, p2)))
    assert get_player_freqdeck(game.state, p1)[1] == 1
    assert get_player_freqdeck(game.state, p2)[0] == 1


@patch("catanatron.apply_action.roll_dice")
def test_offer_validation(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    advance_to_play_turn(game)
    p0, p1 = game.state.colors
    game.play_tick()  # roll
    set_hand(game.state, p0, [1, 0, 0, 0, 0])

    # cannot offer cards you don't have
    bad = Action(p0, ActionType.OFFER_TRADE, (0, 1, 0, 0, 0, 1, 0, 0, 0, 0))
    assert not is_valid_action(game.playable_actions, game.state, bad)
    with pytest.raises(ValueError):
        game.execute(bad)
    # (also refused at apply level)
    with pytest.raises(ValueError):
        game.execute(bad, validate_action=False)

    # non-template but legal offer is accepted (1 wood for 3 ore)
    weird = Action(p0, ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 0, 0, 0, 3))
    assert weird not in game.playable_actions
    assert is_valid_action(game.playable_actions, game.state, weird)

    # giving away / same-resource still invalid
    assert not is_valid_trade((1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    assert not is_valid_trade((1, 0, 0, 0, 0, 1, 0, 0, 0, 0))

    # responder can only accept if they hold the asked cards
    set_hand(game.state, p1, [0, 0, 0, 0, 0])
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    game.execute(Action(p0, ActionType.OFFER_TRADE, offer))
    assert [a.action_type for a in game.playable_actions] == [ActionType.REJECT_TRADE]
    game.play_tick()

    # confirm refuses if acceptee lost the cards in the meantime
    game.state.turn_trade_offers = ()
    set_hand(game.state, p1, [0, 1, 0, 0, 0])
    game.execute(Action(p0, ActionType.OFFER_TRADE, offer))
    game.execute(Action(p1, ActionType.ACCEPT_TRADE, game.state.current_trade))
    set_hand(game.state, p1, [0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        game.execute(Action(p0, ActionType.CONFIRM_TRADE, (*offer, p1)))


@patch("catanatron.apply_action.roll_dice")
def test_offer_cap_and_no_repeats(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, max_trade_offers_per_turn=2)
    assert game.state.max_trade_offers_per_turn == 2
    assert game.copy().state.max_trade_offers_per_turn == 2
    advance_to_play_turn(game)
    p0, p1 = game.state.colors
    game.play_tick()  # roll
    set_hand(game.state, p0, [2, 2, 0, 0, 0])
    set_hand(game.state, p1, [0, 0, 0, 0, 0])
    game.playable_actions = generate_playable_actions(game.state)

    offers = [a for a in game.playable_actions if a.action_type == ActionType.OFFER_TRADE]
    assert len(offers) > 0
    # templates: give 1/2 of held resource for 1/2 of a different one
    for a in offers:
        offering, asking = a.value[:5], a.value[5:]
        assert len(a.value) == 10
        assert sum(1 for x in offering if x) == 1 and sum(1 for x in asking if x) == 1
        assert (sum(offering), sum(asking)) in [(1, 1), (2, 1), (1, 2)]
        assert is_valid_trade(a.value, state=game.state, color=p0)
    assert Action(p0, ActionType.OFFER_TRADE, (2, 0, 0, 0, 0, 0, 0, 0, 1, 0)) in offers
    assert Action(p0, ActionType.OFFER_TRADE, (0, 0, 1, 0, 0, 1, 0, 0, 0, 0)) not in offers

    first = (1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    game.execute(Action(p0, ActionType.OFFER_TRADE, first))
    game.play_tick()  # p1 rejects (only option)
    assert game.state.turn_trade_offers == (first,)
    # identical offer not allowed again this turn (neither generated nor accepted)
    repeat = Action(p0, ActionType.OFFER_TRADE, first)
    assert repeat not in game.playable_actions
    assert not is_valid_action(game.playable_actions, game.state, repeat)

    second = (0, 1, 0, 0, 0, 0, 0, 1, 0, 0)
    game.execute(Action(p0, ActionType.OFFER_TRADE, second))
    game.play_tick()  # rejected
    # cap reached
    assert not any(a.action_type == ActionType.OFFER_TRADE for a in game.playable_actions)
    third = Action(p0, ActionType.OFFER_TRADE, (0, 1, 0, 0, 0, 0, 0, 0, 1, 0))
    assert not is_valid_action(game.playable_actions, game.state, third)
    with pytest.raises(ValueError):
        game.execute(third)

    # END_TURN resets the per-turn log
    game.execute(Action(p0, ActionType.END_TURN, None))
    assert game.state.turn_trade_offers == ()


def test_no_offers_before_rolling_or_without_opponents():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    advance_to_play_turn(game)
    assert not any(a.action_type == ActionType.OFFER_TRADE for a in game.playable_actions)


# ===== Bank shortage
def _tile_and_nodes(board):
    tile, red_node, blue_node = board.map.land_tiles[(0, 0, 0)], 3, 0
    if tile.resource is None:  # is desert
        tile, red_node, blue_node = board.map.land_tiles[(1, -1, 0)], 8, 6
    return tile, red_node, blue_node


def test_bank_shortage_single_recipient_gets_remainder():
    board = Board()
    bank = starting_resource_bank()
    tile, red_node, _ = _tile_and_nodes(board)

    board.build_settlement(Color.RED, red_node, initial_build_phase=True)
    board.build_city(Color.RED, red_node)  # owed 2
    freqdeck_draw(bank, 18, tile.resource)  # 1 left

    payout, depleted = yield_resources(board, bank, tile.number)
    assert depleted == [tile.resource]
    assert freqdeck_count(payout[Color.RED], tile.resource) == 1


def test_bank_shortage_multiple_recipients_get_nothing():
    board = Board()
    bank = starting_resource_bank()
    tile, red_node, blue_node = _tile_and_nodes(board)

    board.build_settlement(Color.RED, red_node, initial_build_phase=True)
    board.build_settlement(Color.BLUE, blue_node, initial_build_phase=True)
    freqdeck_draw(bank, 18, tile.resource)  # 1 left, 2 owed to 2 players

    payout, depleted = yield_resources(board, bank, tile.number)
    assert depleted == [tile.resource]
    assert freqdeck_count(payout[Color.RED], tile.resource) == 0
    assert freqdeck_count(payout[Color.BLUE], tile.resource) == 0


# ===== Longest road recount
class RoadFixture:
    """State + board helper mirroring what apply_action does for free builds."""

    def __init__(self, colors):
        self.state = State([SimplePlayer(c) for c in colors])
        self.board = self.state.board

    def settlement(self, color, node_id, initial=True):
        result = self.board.build_settlement(color, node_id, initial)
        build_settlement(self.state, color, node_id, True)
        if not initial:
            maintain_longest_road(self.state, *result)

    def road(self, color, edge):
        result = self.board.build_road(color, edge)
        build_road(self.state, color, edge, True)
        maintain_longest_road(self.state, *result)

    def vps(self, color):
        return get_actual_victory_points(self.state, color)


def _red_line(f, length):
    """RED: settlement at 0, road 0-1-2-3-4-5-16-18 truncated to `length`."""
    path = [0, 1, 2, 3, 4, 5, 16, 18]
    f.settlement(Color.RED, 0)
    for a, b in zip(path, path[1:]):
        if len(f.state.buildings_by_color[Color.RED]["ROAD"]) >= length:
            break
        f.road(Color.RED, (a, b))


def _blue_five(f):
    """BLUE: 5-road line 10-11-12-13-14-15 far from RED plus a second
    settlement at 7 with roads 7-8-9 reaching node 2 (to cut RED there)."""
    f.settlement(Color.BLUE, 10)
    for a, b in [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]:
        f.road(Color.BLUE, (a, b))
    f.settlement(Color.BLUE, 7)
    for a, b in [(7, 8), (8, 9), (9, 2)]:
        f.road(Color.BLUE, (a, b))


def _white_five(f):
    f.settlement(Color.WHITE, 36)
    for a, b in [(36, 35), (35, 34), (34, 33), (33, 32), (32, 31)]:
        f.road(Color.WHITE, (a, b))


def test_cut_road_passes_card_to_sole_longest():
    f = RoadFixture([Color.RED, Color.BLUE])
    _red_line(f, 6)
    assert f.board.road_color == Color.RED and f.board.road_length == 6
    _blue_five(f)
    assert f.board.road_color == Color.RED  # 5 < 6
    assert f.vps(Color.RED) == 1 + 2 and f.vps(Color.BLUE) == 2

    f.settlement(Color.BLUE, 2, initial=False)  # cut: RED becomes 2 + 4
    assert f.board.road_lengths[Color.RED] == 4
    assert f.board.road_color == Color.BLUE and f.board.road_length == 5
    assert get_longest_road_color(f.state) == Color.BLUE
    assert f.vps(Color.RED) == 1 and f.vps(Color.BLUE) == 3 + 2


def test_cut_road_below_five_with_no_successor_sets_card_aside():
    f = RoadFixture([Color.RED, Color.BLUE])
    _red_line(f, 5)
    assert f.board.road_color == Color.RED
    f.settlement(Color.BLUE, 7)
    for a, b in [(7, 8), (8, 9), (9, 2)]:
        f.road(Color.BLUE, (a, b))
    f.settlement(Color.BLUE, 2, initial=False)  # RED: 2 + 3

    assert f.board.road_color is None and f.board.road_length == 0
    assert get_longest_road_color(f.state) is None
    assert f.vps(Color.RED) == 1 and f.vps(Color.BLUE) == 2

    # first to reach 5 again takes it
    for a, b in [(9, 10), (10, 11), (11, 12)]:
        f.road(Color.BLUE, (a, b))  # 2-9-10-11-12 (via 8-9 too): 7-8-9-10-11-12 = 5
    assert f.board.road_color == Color.BLUE
    assert f.vps(Color.BLUE) == 4


def test_cut_road_with_tie_among_others_sets_card_aside():
    f = RoadFixture([Color.RED, Color.BLUE, Color.WHITE])
    _red_line(f, 6)
    _blue_five(f)
    _white_five(f)
    assert f.board.road_color == Color.RED

    f.settlement(Color.BLUE, 2, initial=False)  # RED: 2 + 4; BLUE 5, WHITE 5 tie
    assert f.board.road_color is None
    assert f.board.road_length == 5
    assert get_longest_road_color(f.state) is None
    assert f.vps(Color.RED) == 1
    assert f.vps(Color.BLUE) == 3 and f.vps(Color.WHITE) == 1

    # tie broken by WHITE building a 6th road
    f.road(Color.WHITE, (31, 30))
    assert f.board.road_color == Color.WHITE
    assert f.vps(Color.WHITE) == 3


def test_cut_road_holder_keeps_card_when_still_tied():
    f = RoadFixture([Color.RED, Color.BLUE])
    _red_line(f, 7)  # 0-1-2-3-4-5-16-18
    _blue_five(f)
    assert f.board.road_color == Color.RED

    f.settlement(Color.BLUE, 2, initial=False)  # RED: 2 + 5 => tied with BLUE at 5
    assert f.board.road_lengths[Color.RED] == 5
    assert f.board.road_color == Color.RED and f.board.road_length == 5
    assert f.vps(Color.RED) == 3 and f.vps(Color.BLUE) == 3


# ===== Win only on own turn
@patch("catanatron.apply_action.roll_dice")
def test_win_only_on_own_turn(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    advance_to_play_turn(game)
    p0, p1 = game.state.colors

    # p1 reaches 10 during p0's turn: not a winner yet
    game.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = 10
    assert game.winning_color() is None
    game.play_tick()  # p0 rolls
    assert game.winning_color() is None
    game.execute(Action(p0, ActionType.END_TURN, None))
    # ...but wins as soon as their turn starts
    assert game.state.current_color() == p1
    assert game.winning_color() == p1

    # play() stops right there
    assert game.play() == p1
    assert game.state.current_turn_index == 1


@patch("catanatron.apply_action.roll_dice")
def test_current_player_wins_immediately(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players, vps_to_win=3)
    advance_to_play_turn(game)
    p0 = game.state.colors[0]
    game.play_tick()  # roll
    game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 3
    assert game.winning_color() == p0


# ===== Seedable maps
def _map_signature(game):
    tiles = [
        (coord, t.resource, t.number) for coord, t in game.state.board.map.land_tiles.items()
    ]
    ports = [(coord, p.resource) for coord, p in game.state.board.map.tiles.items() if hasattr(p, "resource") and coord not in game.state.board.map.land_tiles]
    return tiles, ports


def test_seeded_games_are_reproducible():
    def make(seed):
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE)]
        return Game(players, seed=seed)

    random.seed(123)  # global state must not matter
    a = make(7)
    random.seed(456)
    b = make(7)
    assert _map_signature(a) == _map_signature(b)
    assert a.state.colors == b.state.colors
    assert a.state.development_listdeck == b.state.development_listdeck

    c = make(8)
    assert _map_signature(a) != _map_signature(c)


def test_build_map_accepts_rng():
    from catanatron.models.map import build_map

    m1 = build_map("BASE", rng=random.Random(1))
    m2 = build_map("BASE", rng=random.Random(1))
    m3 = build_map("BASE", rng=random.Random(2))
    sig = lambda m: [(c, t.resource, t.number) for c, t in m.land_tiles.items()]
    assert sig(m1) == sig(m2)
    assert sig(m1) != sig(m3)
