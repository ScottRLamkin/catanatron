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
    data = ["BLUE", "DISCARD", "WOOD"]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.DISCARD
    assert action.value == "WOOD"


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
