import pytest
import json

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, WOOD, BRICK, SHEEP, ORE
from catanatron.models.player import SimplePlayer, Color
from catanatron.json import GameEncoder, action_from_json


def test_serialization():
    game = Game(
        players=[
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
    )

    string = json.dumps(game, cls=GameEncoder)
    result = json.loads(string)

    # Loosely assert looks like expected
    assert isinstance(result["robber_coordinate"], list)
    assert isinstance(result["tiles"], list)
    assert isinstance(result["edges"], list)
    assert isinstance(result["nodes"], dict)
    assert isinstance(result["action_records"], list)

    # New trade/discard related state is serialized
    assert result["current_trade"] == list(game.state.current_trade)
    assert result["acceptees"] == {
        color.value: accepted
        for color, accepted in zip(game.state.colors, game.state.acceptees)
    }
    assert result["turn_trade_offers"] == list(game.state.turn_trade_offers)
    assert result["max_trade_offers_per_turn"] == game.state.max_trade_offers_per_turn


def test_action_from_json_maritime_trade():
    data = ["RED", "MARITIME_TRADE", [SHEEP, SHEEP, SHEEP, SHEEP, ORE]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MARITIME_TRADE
    assert action.value == (SHEEP, SHEEP, SHEEP, SHEEP, ORE)


def test_action_from_json_play_year_of_plenty_two_resources():
    data = ["RED", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (WOOD, BRICK)


def test_action_from_json_play_year_of_plenty_one_resource():
    data = ["BLUE", "PLAY_YEAR_OF_PLENTY", [SHEEP]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (SHEEP,)


def test_action_from_json_play_year_of_plenty_invalid():
    data = ["WHITE", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK, SHEEP]]
    with pytest.raises(
        ValueError, match="Year of Plenty action must have 1 or 2 resources"
    ):
        action_from_json(data)


def test_action_from_json_move_robber_with_victim():
    data = ["ORANGE", "MOVE_ROBBER", [[0, 0, 0], "RED"]]
    action = action_from_json(data)
    assert action.color == Color.ORANGE
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((0, 0, 0), Color.RED)


def test_action_from_json_move_robber_without_victim():
    data = ["RED", "MOVE_ROBBER", [[1, -1, 0], None]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((1, -1, 0), None)


def test_action_from_json_build_road():
    data = ["BLUE", "BUILD_ROAD", [0, 1]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.BUILD_ROAD
    assert action.value == (0, 1)


def test_action_from_json_discard():
    data = ["RED", "DISCARD", [1, 2, 0, 0, 1]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.DISCARD
    assert action.value == (1, 2, 0, 0, 1)


def test_action_from_json_discard_redacted():
    data = ["RED", "DISCARD", None]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.DISCARD
    assert action.value is None


def test_serialization_redacts_pending_discard_values():
    """While a 7-roll's discards are still being submitted (is_discarding),
    no submitted DISCARD value should be visible in the JSON action log --
    not even the discarder's own, since consumers of the serialized game
    are treated as any other later decider/observer. Once everyone has
    submitted (round resolved), the values become visible again.

    Deterministically forces the scenario (two players over the discard
    limit) rather than relying on random play to hit it."""
    from catanatron.apply_action import apply_discard, apply_roll
    from catanatron.models.discard_rule import DISCARD_ORDERS, keep_rule
    from catanatron.models.enums import Action, ActionRecord
    from catanatron.state_functions import get_player_freqdeck, player_key

    game = Game(
        players=[
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
    )
    state = game.state
    state.is_initial_build_phase = False
    state.current_player_index = 0
    state.current_turn_index = 0
    from catanatron.models.enums import ActionPrompt

    state.current_prompt = ActionPrompt.PLAY_TURN

    # Seats 0 and 1 go over the discard limit (7); seats 2, 3 don't owe
    # anything. (Game shuffles seating, so we don't assume which Color ends
    # up at each seat -- only the seat index matters here.)
    for i, resource_count in [(0, 10), (1, 8)]:
        state.player_state[f"P{i}_WOOD_IN_HAND"] = resource_count

    roller = state.colors[0]
    apply_roll(
        state,
        Action(roller, ActionType.ROLL, None),
        action_record=ActionRecord(action=Action(roller, ActionType.ROLL, None), result=(3, 4)),
    )
    state.action_records.append(ActionRecord(Action(roller, ActionType.ROLL, (3, 4)), (3, 4)))
    assert state.is_discarding
    assert state.player_state["P0_DISCARD_OWED"] == 5
    assert state.player_state["P1_DISCARD_OWED"] == 4

    def submit(color_index):
        color = state.colors[color_index]
        hand = get_player_freqdeck(state, color)
        owed = state.player_state[f"{player_key(state, color)}_DISCARD_OWED"]
        value = keep_rule(hand, owed, DISCARD_ORDERS[0])
        record = apply_discard(state, Action(color, ActionType.DISCARD, value))
        state.action_records.append(record)
        return value

    # A record in the JSON action log is [[color, action_type, value], result].
    first_value = submit(0)  # seat 0 submits; seat 1 still owes -> round open
    assert state.is_discarding

    result = json.loads(json.dumps(game, cls=GameEncoder))
    first_record = result["action_records"][-1]
    assert first_record[0][1] == "DISCARD"
    assert first_record[0][2] is None  # hidden: seat 1 hasn't submitted yet

    second_value = submit(1)  # seat 1 submits: round resolves
    assert not state.is_discarding

    result = json.loads(json.dumps(game, cls=GameEncoder))
    second_record = result["action_records"][-1]
    first_record_after = result["action_records"][-2]
    assert second_record[0][1] == "DISCARD"
    assert second_record[0][2] == list(second_value)  # visible: round resolved
    assert first_record_after[0][2] == list(first_value)  # now visible too


def test_action_from_json_offer_trade_round_trips():
    offer = (2, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    data = ["RED", "OFFER_TRADE", list(offer)]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.OFFER_TRADE
    assert action.value == offer


def test_action_from_json_accept_and_reject_trade_round_trip():
    current_trade = (2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    for action_type in ["ACCEPT_TRADE", "REJECT_TRADE"]:
        data = ["WHITE", action_type, list(current_trade)]
        action = action_from_json(data)
        assert action.color == Color.WHITE
        assert action.action_type == ActionType[action_type]
        assert action.value == current_trade


def test_action_from_json_confirm_trade_round_trip():
    offer10 = (2, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    data = ["RED", "CONFIRM_TRADE", [*offer10, "BLUE"]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.CONFIRM_TRADE
    assert action.value == (*offer10, Color.BLUE)
