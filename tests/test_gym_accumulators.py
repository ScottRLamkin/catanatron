from catanatron.game import Game
from catanatron.gym.accumulators import ReinforcementLearningAccumulator
from catanatron.models.enums import ActionType
from catanatron.models.player import Color, RandomPlayer


def test_accumulator_skips_unrepresentable_discards():
    """RandomPlayer discards are not restricted to the 24 keep-rule sets, so
    some of its DISCARD actions cannot be represented as a keep-rule order
    index. The accumulator must drop those transitions (never guess) while
    keeping samples/actions/board_tensors/color_action_indices in lockstep.
    """
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=42)
    accumulator = ReinforcementLearningAccumulator(include_board_tensor=False)
    game.play(accumulators=[accumulator])

    num_discards_in_game = sum(
        1
        for ar in game.state.action_records
        if ar.action.action_type == ActionType.DISCARD
    )
    # All logged data structures stay the same length.
    n = len(accumulator.data["samples"])
    assert len(accumulator.data["actions"]) == n
    assert len(accumulator.data["acting_color"]) == n
    assert sum(len(v) for v in accumulator.data["color_action_indices"].values()) == n

    # Some discards happened (RandomPlayer discards on a 7 in a 4p game),
    # and the accumulator logged at most that many DISCARD entries -- some
    # may have been legitimately dropped as non-keep-rule sets.
    if num_discards_in_game > 0:
        from catanatron.gym.envs.catanatron_env import ACTION_TYPES

        discard_type_index = ACTION_TYPES.index(ActionType.DISCARD)
        logged_discard_count = sum(
            1 for a in accumulator.data["actions"] if a[1] == discard_type_index
        )
        assert logged_discard_count <= num_discards_in_game
