"""
Classes to encode/decode catanatron classes to JSON format.
"""

import json
from enum import Enum

from catanatron.models.map import Water, Port, LandTile
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.state_functions import get_longest_road_length, get_state_index
from catanatron.state import State


def longest_roads_by_player(state: State):
    result = dict()
    for color in state.colors:
        result[color.value] = get_longest_road_length(state, color)
    return result


def redacted_action_records(state: State):
    """`state.action_records`, with the DISCARD value of any still-pending
    submission hidden (replaced with None). A 7-roll's discards are prompted
    one owing player at a time but resolved simultaneously: while
    `state.is_discarding` is True, the trailing run of DISCARD records is
    this still-open round, and no later decider (nor any JSON/snapshot
    consumer) should see what an earlier player in the same round chose to
    discard. Once the round resolves (is_discarding is False), those values
    are revealed like any other historical action."""
    records = list(state.action_records)
    if not state.is_discarding:
        return records
    redacted = list(records)
    for i in range(len(redacted) - 1, -1, -1):
        action = redacted[i].action
        if action.action_type != ActionType.DISCARD:
            break
        redacted[i] = ActionRecord(
            action=Action(action.color, action.action_type, None),
            result=redacted[i].result,
        )
    return redacted


def action_from_json(data) -> Action:
    color = Color[data[0]]
    action_type = ActionType[data[1]]
    if action_type == ActionType.BUILD_ROAD:
        action = Action(color, action_type, tuple(data[2]))
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = tuple(data[2])
        if len(resources) not in [1, 2]:
            raise ValueError("Year of Plenty action must have 1 or 2 resources")
        action = Action(color, action_type, resources)
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = data[2]
        coordinate = tuple(coordinate)
        victim = Color[victim] if victim else None
        action = Action(color, action_type, (coordinate, victim))
    elif action_type == ActionType.MARITIME_TRADE:
        value = tuple(data[2])
        action = Action(color, action_type, value)
    elif action_type in (
        ActionType.OFFER_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    ):
        # 10-tuple (offered freqdeck + asked freqdeck), or an 11-tuple for
        # ACCEPT/REJECT_TRADE (same, plus the offerer's index).
        action = Action(color, action_type, tuple(data[2]))
    elif action_type == ActionType.CONFIRM_TRADE:
        # 11-tuple: 10-value offer plus the accepting player's Color.
        *offer, acceptee = data[2]
        action = Action(color, action_type, (*offer, Color[acceptee]))
    elif action_type == ActionType.DISCARD:
        # A 5-tuple freqdeck (WOOD, BRICK, SHEEP, WHEAT, ORE), or None for a
        # redacted (still-pending, hidden) submission -- see
        # redacted_action_records / GameEncoder.
        action = Action(
            color, action_type, None if data[2] is None else tuple(data[2])
        )
    else:
        action = Action(color, action_type, data[2])
    return action


class GameEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
            return obj
        if isinstance(obj, Game):
            nodes = {}
            edges = {}
            for coordinate, tile in obj.state.board.map.tiles.items():
                for direction, node_id in tile.nodes.items():
                    building = obj.state.board.buildings.get(node_id, None)
                    color = None if building is None else building[0]
                    building_type = None if building is None else building[1]
                    nodes[node_id] = {
                        "id": node_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building_type),
                        "color": self.default(color),
                    }
                for direction, edge in tile.edges.items():
                    color = obj.state.board.roads.get(edge, None)
                    edge_id = tuple(sorted(edge))
                    edges[edge_id] = {
                        "id": edge_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "color": self.default(color),
                    }
            return {
                "tiles": [
                    {"coordinate": coordinate, "tile": self.default(tile)}
                    for coordinate, tile in obj.state.board.map.tiles.items()
                ],
                "adjacent_tiles": obj.state.board.map.adjacent_tiles,
                "nodes": nodes,
                "edges": list(edges.values()),
                "action_records": [
                    self.default(a) for a in redacted_action_records(obj.state)
                ],
                "player_state": obj.state.player_state,
                "colors": obj.state.colors,
                "bot_colors": list(
                    map(
                        lambda p: p.color, filter(lambda p: p.is_bot, obj.state.players)
                    )
                ),
                "is_initial_build_phase": obj.state.is_initial_build_phase,
                "robber_coordinate": obj.state.board.robber_coordinate,
                "current_color": obj.state.current_color(),
                "current_prompt": obj.state.current_prompt,
                "current_playable_actions": obj.playable_actions,
                "longest_roads_by_player": longest_roads_by_player(obj.state),
                "winning_color": obj.winning_color(),
                "state_index": get_state_index(obj.state),
                "current_trade": obj.state.current_trade,
                "acceptees": {
                    color.value: accepted
                    for color, accepted in zip(obj.state.colors, obj.state.acceptees)
                },
                "turn_trade_offers": obj.state.turn_trade_offers,
                "max_trade_offers_per_turn": obj.state.max_trade_offers_per_turn,
            }
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            return {
                "id": obj.id,
                "type": "PORT",
                "direction": self.default(obj.direction),
                "resource": self.default(obj.resource),
            }
        if isinstance(obj, LandTile):
            if obj.resource is None:
                return {"id": obj.id, "type": "DESERT"}
            return {
                "id": obj.id,
                "type": "RESOURCE_TILE",
                "resource": self.default(obj.resource),
                "number": obj.number,
            }
        return json.JSONEncoder.default(self, obj)
