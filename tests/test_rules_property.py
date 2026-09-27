"""Property test: many full games with random bots, checking invariants of
the exact-rules engine (termination, non-negative hands, resource
conservation, discard amounts)."""

import random

from catanatron.game import TURNS_LIMIT, Game, GameAccumulator
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import (
    get_actual_victory_points,
    get_player_freqdeck,
    player_num_resource_cards,
)

NUM_GAMES = 200
TOTAL_PER_RESOURCE = 19


def check_invariants(state):
    hands = [get_player_freqdeck(state, color) for color in state.colors]
    for hand in hands:
        assert all(n >= 0 for n in hand), f"negative hand {hand}"
    assert all(n >= 0 for n in state.resource_freqdeck)
    for i, _ in enumerate(RESOURCES):
        total = state.resource_freqdeck[i] + sum(hand[i] for hand in hands)
        assert total == TOTAL_PER_RESOURCE, f"conservation broken: {total}"
    for i in range(len(state.colors)):
        assert state.player_state[f"P{i}_DISCARD_OWED"] >= 0


class InvariantAccumulator(GameAccumulator):
    def __init__(self):
        self.prev_hands = None
        self.expected_discards = []  # per 7 rolled that triggered discards
        self.actual_discards = []
        self.num_steps = 0

    def step(self, game, action):
        state = game.state
        self.num_steps += 1
        check_invariants(state)

        last = state.action_records[-1] if state.action_records else None
        if (
            last is not None
            and last.action.action_type == ActionType.ROLL
            and sum(last.result) == 7
        ):
            expected = sum(
                n // 2 for n in self.prev_hands if n > state.discard_limit
            )
            self.expected_discards.append(expected)
            self.actual_discards.append(0)
        if action.action_type == ActionType.DISCARD:
            assert state.is_discarding
            assert action.value in RESOURCES
            assert player_num_resource_cards(state, action.color, action.value) > 0
            self.actual_discards[-1] += 1

        self.prev_hands = [
            player_num_resource_cards(state, color) for color in state.colors
        ]


def test_many_games_preserve_rules():
    rng = random.Random(42)
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    winners = 0
    total_sevens = 0
    for i in range(NUM_GAMES):
        num_players = rng.choice([2, 3, 4])
        players = [
            rng.choice([RandomPlayer, WeightedRandomPlayer])(color)
            for color in colors[:num_players]
        ]
        game = Game(players, seed=i)
        acc = InvariantAccumulator()
        game.play(accumulators=[acc])

        state = game.state
        check_invariants(state)
        assert acc.num_steps == len(state.action_records)
        # termination
        winner = game.winning_color()
        assert winner is not None or state.num_turns >= TURNS_LIMIT
        if winner is not None:
            winners += 1
            # winner is the player whose turn it is and has >= vps_to_win
            assert state.colors[state.current_turn_index] == winner
            assert get_actual_victory_points(state, winner) >= game.vps_to_win
        # discards total floor(n/2) for every player over the limit
        assert acc.expected_discards == acc.actual_discards
        total_sevens += len(acc.expected_discards)
        # no discard owed left over
        assert all(
            state.player_state[f"P{j}_DISCARD_OWED"] == 0 for j in range(num_players)
        )

    assert winners >= NUM_GAMES * 0.95
    assert total_sevens > 0
