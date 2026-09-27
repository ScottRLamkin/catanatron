import random
from typing import TypedDict, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionType
from catanatron.models.board import get_edges
from catanatron.models.actions import DOMESTIC_TRADE_TEMPLATES
from catanatron.models.discard_rule import DISCARD_ORDERS, keep_rule
from catanatron.state_functions import get_player_freqdeck, player_key
from catanatron.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron.gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)


BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]

# Relative seat offsets (from P0's perspective, in seat order) that a
# MOVE_ROBBER victim or a CONFIRM_TRADE acceptee can be. None means
# "no victim"/self is never a valid CONFIRM_TRADE target. Up to 3 other
# players can sit relative to P0 (4-player game).
RELATIVE_SEATS = [None, 1, 2, 3]
OTHER_RELATIVE_SEATS = [1, 2, 3]


def colors_relative_seat(colors, p0_color, color):
    """Seat offset (0..len(colors)-1) of `color` relative to `p0_color`, in
    seating order. Returns None if `color` is None."""
    if color is None:
        return None
    n = len(colors)
    p0_idx = colors.index(p0_color)
    idx = colors.index(color)
    return (idx - p0_idx) % n


def relative_seat_to_color(colors, p0_color, relative_seat):
    """Inverse of colors_relative_seat."""
    if relative_seat is None:
        return None
    n = len(colors)
    p0_idx = colors.index(p0_color)
    idx = (p0_idx + relative_seat) % n
    return colors[idx]


# Fixed enumeration of the domestic trade offers exposed in the action space:
# for every ordered pair of distinct resources and every
# DOMESTIC_TRADE_TEMPLATES (give, get) template. 5 * 4 * 3 = 60 offers.
def _build_offer_trade_values():
    values = []
    for give_resource in RESOURCES:
        for get_resource in RESOURCES:
            if give_resource == get_resource:
                continue
            for give, get in DOMESTIC_TRADE_TEMPLATES:
                value = [0] * 10
                value[RESOURCES.index(give_resource)] = give
                value[5 + RESOURCES.index(get_resource)] = get
                values.append(tuple(value))
    return values


OFFER_TRADE_VALUES = _build_offer_trade_values()

ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    # TODO: One for each tile (and abuse 1v1 setting).
    *[
        (ActionType.MOVE_ROBBER, (tile, relative_seat))
        for tile in TILE_COORDINATES
        for relative_seat in RELATIVE_SEATS
    ],
    # DISCARD slot i = discard the keep-rule set for DISCARD_ORDERS[i] (see
    # catanatron.models.discard_rule); the engine action value is the
    # resulting 5-freqdeck. Mapping in either direction needs the State.
    *[(ActionType.DISCARD, i) for i in range(len(DISCARD_ORDERS))],
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # Domestic (player-to-player) trade
    *[(ActionType.OFFER_TRADE, value) for value in OFFER_TRADE_VALUES],
    (ActionType.ACCEPT_TRADE, None),
    (ActionType.REJECT_TRADE, None),
    *[
        (ActionType.CONFIRM_TRADE, relative_seat)
        for relative_seat in OTHER_RELATIVE_SEATS
    ],
    (ActionType.CANCEL_TRADE, None),
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
ACTIONS_ARRAY_INDEX = {value: i for i, value in enumerate(ACTIONS_ARRAY)}
ACTION_TYPES = [i for i in ActionType]


def to_action_type_space(action_type: ActionType) -> int:
    return ACTION_TYPES.index(action_type)


def _discard_keep_sets(state, color):
    """[(order_index, discard freqdeck)] for `color`'s current hand/owed."""
    owed = state.player_state[f"{player_key(state, color)}_DISCARD_OWED"]
    hand = get_player_freqdeck(state, color)
    return [(i, keep_rule(hand, owed, order)) for i, order in enumerate(DISCARD_ORDERS)]


def discard_to_order_index(state, action):
    """Action-space value for a DISCARD action: the lowest DISCARD_ORDERS
    index whose keep-rule set equals action.value. A set no order produces
    (e.g. a RandomPlayer's) maps to the closest keep-rule set (L1 distance,
    ties to the lowest index) so that logging never fails; this is lossy."""
    target = tuple(action.value)
    best_index, best_distance = 0, None
    for i, discard in _discard_keep_sets(state, action.color):
        distance = sum(abs(a - b) for a, b in zip(discard, target))
        if distance == 0:
            return i
        if best_distance is None or distance < best_distance:
            best_index, best_distance = i, distance
    return best_index


# NOTE: I think I don't need this if we separate action and action_record nicely...
def normalize_action(action, colors, p0_color=None, state=None):
    """Maps an Action to the color-agnostic (relative-seat) representation
    used by ACTIONS_ARRAY.

    Args:
        action: the Action to normalize.
        colors: seating order (e.g. game.state.colors) used to compute
            relative seats for actions that reference another player
            (MOVE_ROBBER victim, CONFIRM_TRADE acceptee).
        p0_color: the color from whose perspective relative seats are
            computed. Defaults to the acting player's own color, which is
            enough for callers (like logging/accumulators) that only need a
            canonical, unambiguous encoding of the action itself.
        state: the State the action is taken in. Required for DISCARD
            (its freqdeck value is encoded as a keep-rule order index, which
            depends on the player's hand).
    """
    p0_color = p0_color if p0_color is not None else action.color
    action_type = action.action_type
    if action_type == ActionType.ROLL:
        return Action(action.color, action_type, None)
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = action.value
        relative_seat = colors_relative_seat(colors, p0_color, victim)
        return Action(action.color, action_type, (coordinate, relative_seat))
    elif action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action_type, tuple(sorted(action.value)))
    elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action_type, None)
    elif action_type == ActionType.DISCARD:
        if state is None:
            raise ValueError("normalize_action needs `state` for DISCARD actions")
        return Action(action.color, action_type, discard_to_order_index(state, action))
    elif action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
        # value-less in the action space; resolved via state.current_trade.
        return Action(action.color, action_type, None)
    elif action_type == ActionType.CONFIRM_TRADE:
        acceptee_color = action.value[10]
        relative_seat = colors_relative_seat(colors, p0_color, acceptee_color)
        return Action(action.color, action_type, relative_seat)
    return action


def to_action_space(action, colors, p0_color=None, state=None):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action, colors, p0_color, state)
    return ACTIONS_ARRAY_INDEX[(normalized.action_type, normalized.value)]


def from_action_space(action_int, playable_actions, colors, p0_color=None, state=None):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    if action_type == ActionType.DISCARD:
        # Decode the order index into the exact keep-rule set, then require
        # that set to be listed (it always is when the player is discarding).
        assert state is not None, "from_action_space needs `state` for DISCARD"
        color = playable_actions[0].color
        wanted = dict(_discard_keep_sets(state, color))[value]
        candidate = Action(color, ActionType.DISCARD, wanted)
        catan_action = candidate if candidate in playable_actions else None
    else:
        for action in playable_actions:
            normalized = normalize_action(action, colors, p0_color, state)
            if normalized.action_type == action_type and normalized.value == value:
                catan_action = action
                break  # return the first one
    assert catan_action is not None
    return catan_action


FEATURES = get_feature_ordering(num_players=2)
NUM_FEATURES = len(FEATURES)

# Highest features is NUM_RESOURCES_IN_HAND which in theory is all resource cards
HIGH = 19 * 5


def simple_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1


class MixedObservation(TypedDict):
    board: np.ndarray
    numeric: np.ndarray


class CatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.discard_limit = self.config.get("discard_limit", 7)
        self.max_trade_offers_per_turn = self.config.get(
            "max_trade_offers_per_turn", 3
        )
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space tighter if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=np.float64
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            # TODO: This could be tigher (e.g. _ROADS_AVAILABLE <= 15)
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float64
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            # TODO: This could be tigher (e.g. _ROADS_AVAILABLE <= 15)
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float64
            )

        self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid (deduplicated) action-space integers.
        """
        colors = self.game.state.colors
        state = self.game.state
        playable_actions = self.game.playable_actions
        if self.game.winning_color() is not None:
            # Game over: nothing to decide (the listed actions may belong to
            # the opponent whose move ended the game, and would not encode
            # from P0's perspective).
            return []
        if state.current_prompt == ActionPrompt.DISCARD:
            # Every keep-rule set is legal, so all 24 slots are valid; skip
            # normalizing the (up to 1001) listed sets one by one.
            return sorted(
                {
                    ACTIONS_ARRAY_INDEX[(ActionType.DISCARD, i)]
                    for i in range(len(DISCARD_ORDERS))
                }
            )
        return sorted(
            {to_action_space(a, colors, self.p0.color, state) for a in playable_actions}
        )

    def step(self, action):
        colors = self.game.state.colors
        try:
            catan_action = from_action_space(
                action, self.game.playable_actions, colors, self.p0.color, self.game.state
            )
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        # Use a dedicated generator (seeded the same way Game seeds itself)
        # so that the map is reproducible given `seed`, matching the seating
        # order and development deck (see Game.__init__).
        map_rng = random.Random(seed) if seed is not None else None
        catan_map = build_map(self.map_type, rng=map_rng)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            discard_limit=self.discard_limit,
            max_trade_offers_per_turn=self.max_trade_offers_per_turn,
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self) -> Union[np.ndarray, MixedObservation]:
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        # current_color() already reflects whoever must act next, including
        # out-of-turn prompts directed at another player (e.g. an enemy
        # owing a DISCARD, or P0 being asked to DECIDE_TRADE/DECIDE_ACCEPTEES
        # on someone else's turn) -- so a plain color check is enough to
        # hand control back to P0 whenever it is P0's turn to decide anything.
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


CatanatronEnv.__doc__ = f"""
1v1 environment against a random player

Attributes:
    reward_range: -1 if player lost, 1 if player won, 0 otherwise.
    action_space: Integers from the [0, {ACTION_SPACE_SIZE - 1}] interval.
        See Action Space table below.
    observation_space: Numeric Feature Vector. See Observation Space table
        below for quantities. They appear in vector in alphabetical order,
        from the perspective of "current" player (hiding/showing information
        accordingly). P0 is "current" player. P1 is next in line.

        We use the following nomenclature for Tile ids and Node ids.
        Edge ids are self-describing (node-id, node-id) tuples. We also
        use Cube coordinates for tiles (see
        https://www.redblobgames.com/grids/hexagons/#coordinates)

        MOVE_ROBBER and CONFIRM_TRADE values reference other players by
        their *relative seat* from P0 (1, 2 or 3 seats away, in seating
        order), rather than by Color, so the action space is agnostic to
        which Color P0 happens to be. DISCARD, OFFER_TRADE, ACCEPT_TRADE,
        REJECT_TRADE and CANCEL_TRADE are also part of the action space now:
        - DISCARD picks one of the 24 keep-rule priority orders
          (catanatron.models.discard_rule.DISCARD_ORDERS); the engine
          action discards the full set that order keeps away.
        - OFFER_TRADE is one of a fixed enumeration of 60 template offers
          (every ordered pair of distinct resources x each of the 3
          give/get templates).
        - ACCEPT_TRADE/REJECT_TRADE/CANCEL_TRADE carry no value; they act on
          whatever `state.current_trade` currently is.
        - CONFIRM_TRADE picks which (relative-seat) acceptee to trade with.

.. image:: _static/tile-ids.png
  :width: 300
  :alt: Tile Ids
.. image:: _static/node-ids.png
  :width: 300
  :alt: Node Ids

.. list-table:: Action Space
   :widths: 10 100
   :header-rows: 1

   * - Integer
     - Catanatron Action
"""
for i, v in enumerate(ACTIONS_ARRAY):
    CatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"

CatanatronEnv.__doc__ += """

.. list-table:: Observation Space (Raw)
   :widths: 10 50 10 10
   :header-rows: 1

   * - Feature Name
     - Description
     - Number of Features (N=number of players)
     - Type

   * - BANK_<resource>
     - Number of cards of that `resource` in bank
     - 5
     - Integer
   * - BANK_DEV_CARDS
     - Number of development cards in bank
     - 1
     - Integer

   * - EDGE<i>_P<j>_ROAD
     - Whether edge `i` is owned by player `j`
     - 72 * N
     - Boolean
   * - NODE<i>_P<j>_SETTLEMENT
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - NODE<i>_P<j>_CITY
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - PORT<i>_IS_<resource>
     - Whether node `i` is port of `resource` (or THREE_TO_ONE).
     - 9 * 6
     - Boolean
   * - TILE<i>_HAS_ROBBER
     - Whether robber is on tile `i`.
     - 19
     - Boolean
   * - TILE<i>_IS_<resource>
     - Whether tile `i` yields `resource` (or DESERT).
     - 19 * 6
     - Boolean
   * - TILE<i>_PROBA
     - Tile `i`'s probability of being rolled.
     - 19
     - Float

   * - IS_DISCARDING
     - Whether current player must discard. Discards are resolved one card
       at a time via the DISCARD action (see P0_DISCARD_OWED for how many
       cards P0 still owes).
     - 1
     - Boolean
   * - IS_MOVING_ROBBER
     - Whether current player must move robber (because played knight
       or because rolled a 7).
     - 1
     - Boolean
   * - IS_DECIDING_TRADE
     - Whether P0 is being asked to accept/reject a domestic trade offer.
     - 1
     - Boolean
   * - IS_DECIDING_ACCEPTEES
     - Whether P0 (as the offerer) must confirm-with or cancel a domestic
       trade that one or more players accepted.
     - 1
     - Boolean
   * - P0_DISCARD_OWED
     - Number of cards P0 still owes to discard (0 if none).
     - 1
     - Integer
   * - P<i>_HAS_ROLLED
     - Whether player `i` already rolled dice.
     - N
     - Boolean
   * - P0_HAS_PLAYED _DEVELOPMENT_CARD _IN_TURN
     - Whether current player already played a development card
     - 1
     - Boolean
   * - P0_TRADE_OFFERS_MADE_THIS_TURN
     - Number of domestic trade offers P0 has made so far this turn.
     - 1
     - Integer
   * - CURRENT_TRADE_IS_OFFERED_<resource>
     - How many of `resource` are offered in state.current_trade (0 if
       there is no active trade), from P0's perspective.
     - 5
     - Integer
   * - CURRENT_TRADE_IS_ASKED_<resource>
     - How many of `resource` are asked for in state.current_trade (0 if
       there is no active trade), from P0's perspective.
     - 5
     - Integer
   * - CURRENT_TRADE_OFFERER_IS_P<i>
     - One-hot of which (relative-seat) player made state.current_trade.
     - N
     - Boolean

   * - P0_ACTUAL_VPS
     - Total Victory Points (including Victory Point Development Cards)
     - 1
     - Integer
   * - P0_<resource>_IN_HAND
     - Number of `resource` cards in hand
     - 5
     - Integer
   * - P0_<dev-card>_IN_HAND
     - Number of `dev-card` cards in hand
     - 5
     - Integer
   * - P<i>_NUM_DEVS_IN_HAND
     - Number of hidden development cards player `i` has
     - N
     - Integer
   * - P<i>_NUM_RESOURCES _IN_HAND
     - Number of hidden resource cards player `i` has
     - N
     - Integer

   * - P<i>_HAS_ARMY
     - Whether player <i> has Largest Army
     - N
     - Boolean
   * - P<i>_HAS_ROAD
     - Whether player <i> has Longest Road
     - N
     - Boolean
   * - P<i>_ROADS_LEFT
     - Number of roads pieces player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_SETTLEMENTS_LEFT
     - Number of settlements player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_CITIES_LEFT
     - Number of cities player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_LONGEST_ROAD _LENGTH
     - Length of longest road by player `i`
     - N
     - Integer
   * - P<i>_PUBLIC_VPS
     - Amount of visible victory points for player `i` (i.e.
       doesn't include hidden victory point cards; only army,
       road and settlements/cities).
     - N
     - Integer
   * - P<i>_<dev-card>_PLAYED
     - Amount of `dev-card` cards player `i` has played in game
       (VICTORY_POINT not included).
     - 4 * N
     - Integer
"""
