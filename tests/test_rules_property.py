"""Property test: many full games with random bots, checking invariants of
the exact-rules engine (termination, non-negative hands, resource
conservation, discard amounts)."""

import random

from catanatron.game import TURNS_LIMIT, Game, GameAccumulator
from catanatron.models.actions import iter_discard_freqdecks
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
    # pending sets are always within the (still untouched) hands
    for color, discard in state.pending_discards.items():
        hand = get_player_freqdeck(state, color)
        assert all(0 <= n <= h for n, h in zip(discard, hand))
        assert sum(discard) == state.player_state[
            f"P{state.color_to_index[color]}_DISCARD_OWED"
        ]
    if not state.is_discarding:
        assert state.pending_discards == {}


class InvariantAccumulator(GameAccumulator):
    def __init__(self):
        self.prev_hands = None
        self.expected_discards = []  # per 7 rolled that triggered discards
        self.actual_discards = []
        self.num_steps = 0
        self.max_discard_actions = 0

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
            index = state.color_to_index[action.color]
            owed = state.player_state[f"P{index}_DISCARD_OWED"]
            hand_size = self.prev_hands[index]
            # each discard is a full set of exactly floor(n/2) cards, within hand
            assert hand_size > state.discard_limit
            assert owed == hand_size // 2
            assert len(action.value) == 5 and sum(action.value) == owed
            hand = get_player_freqdeck(state, action.color)
            assert all(0 <= n <= h for n, h in zip(action.value, hand))
            assert not state.player_state[f"P{index}_DISCARD_SUBMITTED"]
            assert action.color not in state.pending_discards
            # every legal set is listed exactly once
            num_sets = len(list(iter_discard_freqdecks(hand, owed)))
            assert len(game.playable_actions) == num_sets
            assert len(set(a.value for a in game.playable_actions)) == num_sets
            assert action in game.playable_actions
            self.max_discard_actions = max(self.max_discard_actions, num_sets)
            self.actual_discards[-1] += owed

        self.prev_hands = [
            player_num_resource_cards(state, color) for color in state.colors
        ]


def test_many_games_preserve_rules():
    rng = random.Random(42)
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    winners = 0
    total_sevens = 0
    max_discard_actions = 0
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
        max_discard_actions = max(max_discard_actions, acc.max_discard_actions)
        # no discard owed left over
        assert all(
            state.player_state[f"P{j}_DISCARD_OWED"] == 0 for j in range(num_players)
        )

    assert winners >= NUM_GAMES * 0.95
    assert total_sevens > 0
    assert max_discard_actions > 5  # some player chose among many sets
