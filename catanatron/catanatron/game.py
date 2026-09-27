"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import Sequence, Union, Optional

from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.state import State
from catanatron.apply_action import apply_action
from catanatron.state_functions import (
    player_key,
    player_has_rolled,
    player_resource_freqdeck_contains,
)
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, Player

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def is_valid_action(playable_actions, state: State, action: Action) -> bool:
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE in the right time.

    OFFER_TRADE is special-cased (instead of just checking membership in
    playable_actions) because playable_actions only lists a bounded set of
    "template" offers, but any legal offer must be accepted.
    """
    if action.action_type == ActionType.OFFER_TRADE:
        return can_offer_trade(state, action.color) and is_valid_trade(
            action.value, state=state, color=action.color
        )

    return action in playable_actions


def can_offer_trade(state: State, color) -> bool:
    """Whether `color` is allowed to make a domestic trade offer right now:
    must be its turn, in PLAY_TURN after rolling, not in the middle of another
    trade or Road Building, with at least one opponent and under the
    per-turn offer cap."""
    return (
        state.current_color() == color
        and state.current_turn_index == state.current_player_index
        and state.current_prompt == ActionPrompt.PLAY_TURN
        and not state.is_resolving_trade
        and not state.is_road_building
        and player_has_rolled(state, color)
        and len(state.colors) > 1
        and len(state.turn_trade_offers) < state.max_trade_offers_per_turn
    )


def is_valid_trade(action_value, state: State = None, color=None):
    """Checks the value of a OFFER_TRADE does not give away resources
    (both sides non-empty) or trade matching resources (official rules).

    If `state` and `color` are given, additionally checks that the offering
    player actually holds the offered cards and that this exact offer has not
    already been made this turn.
    """
    if len(action_value) < 10:
        return False
    offering = action_value[:5]
    asking = action_value[5:10]
    if any(i < 0 for i in offering) or any(j < 0 for j in asking):
        return False
    if sum(offering) == 0 or sum(asking) == 0:
        return False  # cant give away cards

    for i, j in zip(offering, asking):
        if i > 0 and j > 0:
            return False  # cant trade same resources

    if state is not None and color is not None:
        if not player_resource_freqdeck_contains(state, color, offering):
            return False  # cant offer cards you dont have
        if tuple(action_value[:10]) in state.turn_trade_offers:
            return False  # cant repeat identical offer in same turn
    return True


class GameAccumulator:
    """Interface to hook into different game lifecycle events.

    Useful to compute aggregate statistics, log information, etc...
    """

    def __init__(*args, **kwargs):
        pass

    def before(self, game):
        """
        Called when the game is created, no actions have
        been taken by players yet, but the board is decided.
        """
        pass

    def step(self, game_before_action, action):
        """
        Called after each action taken by a player.
        Game should be right before action is taken.
        """
        pass

    def after(self, game):
        """
        Called when the game is finished.

        Check game.winning_color() to see if the game
        actually finished or exceeded turn limit (is None).
        """
        pass


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).

    Attributes:
        state (State): Current game state.
        playable_actions (List[Action]): List of playable actions by current player.
    """

    def __init__(
        self,
        players: Sequence[Player],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        initialize: bool = True,
        max_trade_offers_per_turn: int = 3,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (List[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            discard_limit (int, optional): Discard limit to use. Defaults to 7.
            vps_to_win (int, optional): Victory Points needed to win. Defaults to 10.
            catan_map (CatanMap, optional): Map to use. Defaults to None, in which
                case a random BASE map is built from the seed.
            initialize (bool, optional): Whether to initialize. Defaults to True.
            max_trade_offers_per_turn (int, optional): Cap on domestic trade
                offers a player can make per turn. Defaults to 3.
        """
        if initialize:
            self.seed = seed if seed is not None else random.randrange(sys.maxsize)
            # Global module seeded for dice rolls / robber steals (legacy);
            # a dedicated generator makes map, seating and dev deck reproducible.
            random.seed(self.seed)
            rng = random.Random(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.state = State(
                players,
                catan_map,
                discard_limit=discard_limit,
                max_trade_offers_per_turn=max_trade_offers_per_turn,
                rng=rng,
            )
            self.playable_actions = generate_playable_actions(self.state)

    def play(self, accumulators=[], decide_fn=None):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            accumulators (list[Accumulator], optional): list of Accumulator classes to use.
                Their .consume method will be called with every action, and
                their .finalize method will be called when the game ends (if it ends)
                Defaults to [].
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.
        Returns:
            Color: winning color or None if game exceeded TURNS_LIMIT
        """
        for accumulator in accumulators:
            accumulator.before(self)
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(decide_fn=decide_fn, accumulators=accumulators)
        for accumulator in accumulators:
            accumulator.after(self)
        return self.winning_color()

    def play_tick(self, decide_fn=None, accumulators=[]):
        """Advances game by one ply (player decision).

        Args:
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.

        Returns:
            ActionRecord: representing the executed action
        """
        # Ask Player for action
        player = self.state.current_player()
        action = (
            decide_fn(player, self, self.playable_actions)
            if decide_fn is not None
            else player.decide(self, self.playable_actions)
        )

        # Call accumulator.step here, because we want game_before_action, action
        if len(accumulators) > 0:
            for accumulator in accumulators:
                accumulator.step(self, action)

        # Apply Action, and do Move Generation
        return self.execute(action)

    def execute(
        self,
        action: Action,
        validate_action: bool = True,
        action_record: ActionRecord = None,
    ) -> ActionRecord:
        """Internal call that carries out decided action by player"""
        if validate_action and not is_valid_action(
            self.playable_actions, self.state, action
        ):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.playable_actions}"
            )

        action_record = apply_action(self.state, action, action_record)
        self.playable_actions = generate_playable_actions(self.state)
        return action_record

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color.

        Official rule: a player can only win during their own turn. So the
        winner is the player whose turn it is, if they have reached
        `vps_to_win`. A player that reaches the threshold during someone
        else's turn (e.g. via a Longest Road swap) wins as soon as their
        turn starts (this is checked after every action, including END_TURN).

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        color = self.state.colors[self.state.current_turn_index]
        key = player_key(self.state, color)
        if self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= self.vps_to_win:
            return color
        return None

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game(players=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.state = self.state.copy()
        game_copy.playable_actions = self.playable_actions
        return game_copy
