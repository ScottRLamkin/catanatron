import random

import gymnasium
from gymnasium.utils.env_checker import check_env
import numpy as np

from catanatron.features import get_feature_ordering
from catanatron.models.enums import ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    CatanatronEnv,
    OFFER_TRADE_VALUES,
)

features = get_feature_ordering(2)


def get_p0_num_settlements(obs):
    indexes = [
        i
        for i, name in enumerate(features)
        if "NODE" in name and "SETTLEMENT" in name and "P0" in name
    ]
    return sum([obs[i] for i in indexes])


def test_check_env():
    env = CatanatronEnv()
    check_env(env)


def test_gym():
    env = CatanatronEnv()

    first_observation, info = env.reset()  # this forces advanced until p0...
    assert len(info["valid_actions"]) >= 50  # first seat at most blocked 4 nodes
    assert get_p0_num_settlements(first_observation) == 0

    action = random.choice(info["valid_actions"])
    second_observation, reward, terminated, truncated, info = env.step(action)
    assert np.any(first_observation != second_observation)
    assert reward == 0
    assert not terminated
    assert not truncated
    assert len(info["valid_actions"]) in [2, 3]

    assert second_observation[features.index("BANK_DEV_CARDS")] == 25  # type: ignore
    assert second_observation[features.index("BANK_SHEEP")] == 19  # type: ignore
    assert get_p0_num_settlements(second_observation) == 1

    reset_obs, _ = env.reset()
    assert np.any(reset_obs != second_observation)
    assert get_p0_num_settlements(reset_obs) == 0

    env.close()


def test_gym_registration_and_api_works():
    env = gymnasium.make("catanatron/Catanatron-v0")
    observation, info = env.reset()
    done = False
    reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    assert reward in [-1, 1]


def test_invalid_action_reward():
    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"invalid_action_reward": -1234}
    )
    first_obs, info = env.reset()
    invalid_action = next(filter(lambda i: i not in info["valid_actions"], range(1000)))
    observation, reward, terminated, truncated, info = env.step(invalid_action)
    assert reward == -1234
    assert not terminated
    assert not truncated
    assert (observation == first_obs).all()
    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(invalid_action)
        assert (observation == first_obs).all()
    assert not terminated
    assert truncated


def test_custom_reward():
    def custom_reward(game, p0_color):
        return 123

    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"reward_function": custom_reward}
    )
    observation, info = env.reset()
    action = random.choice(info["valid_actions"])
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 123


def test_custom_map():
    env = gymnasium.make("catanatron/Catanatron-v0", config={"map_type": "MINI"})
    observation, info = env.reset()
    assert len(info["valid_actions"]) < 50
    assert len(observation) < 614
    # assert env.action_space.n == 260


def test_enemies():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [
                ValueFunctionPlayer(Color.RED),
                RandomPlayer(Color.ORANGE),
                RandomPlayer(Color.WHITE),
            ]
        },
    )
    observation, info = env.reset()
    assert len(observation) == len(get_feature_ordering(4))

    done = False
    reward = 0
    while not done:
        action = random.choice(info["valid_actions"])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Virtually impossible for a Random bot to beat Value Function Player
    assert env.unwrapped.game.winning_color() == Color.RED  # type: ignore
    assert reward == -1  # ensure we lost
    env.close()


def test_mixed_rep():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={"representation": "mixed"},
    )
    observation, info = env.reset()
    assert "board" in observation
    assert "numeric" in observation


def test_action_space_size_and_no_duplicates():
    assert ACTION_SPACE_SIZE == len(ACTIONS_ARRAY)
    assert len(set(ACTIONS_ARRAY)) == len(ACTIONS_ARRAY)  # no dup entries


def test_offer_trade_templates_are_60_and_valid():
    assert len(OFFER_TRADE_VALUES) == 60
    assert len(set(OFFER_TRADE_VALUES)) == 60
    for value in OFFER_TRADE_VALUES:
        assert len(value) == 10
        offered, asked = value[:5], value[5:]
        assert sum(1 for v in offered if v > 0) == 1
        assert sum(1 for v in asked if v > 0) == 1


def test_reset_seed_determinism():
    env = CatanatronEnv(
        config={
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.WHITE),
                RandomPlayer(Color.ORANGE),
            ]
        }
    )
    obs1, info1 = env.reset(seed=123)
    board1 = dict(env.game.state.board.map.tiles)
    colors1 = env.game.state.colors
    obs2, info2 = env.reset(seed=123)
    board2 = dict(env.game.state.board.map.tiles)
    colors2 = env.game.state.colors

    assert np.array_equal(obs1, obs2)
    assert info1["valid_actions"] == info2["valid_actions"]
    assert colors1 == colors2
    def tile_signature(tile):
        return (getattr(tile, "resource", None), getattr(tile, "number", None))

    for coord in board1:
        assert tile_signature(board1[coord]) == tile_signature(board2[coord])

    # Different seed should (almost certainly) produce a different board.
    env.reset(seed=456)
    board3 = dict(env.game.state.board.map.tiles)
    assert any(
        tile_signature(board1[coord]) != tile_signature(board3[coord])
        for coord in board1
    )


def test_robber_victim_choice_reaches_chosen_victim():
    env = CatanatronEnv(
        config={
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.WHITE),
                RandomPlayer(Color.ORANGE),
            ]
        }
    )
    obs, info = env.reset(seed=7)

    found_choice = False
    for _ in range(2000):
        robber_actions = [
            a
            for a in info["valid_actions"]
            if ACTIONS_ARRAY[a][0] == ActionType.MOVE_ROBBER
            and ACTIONS_ARRAY[a][1][1] is not None
        ]
        if len(robber_actions) >= 1:
            found_choice = True
            action = robber_actions[0]
            _, (tile, relative_seat) = ACTIONS_ARRAY[action]
            colors = env.game.state.colors
            from catanatron.gym.envs.catanatron_env import relative_seat_to_color

            expected_victim = relative_seat_to_color(
                colors, env.p0.color, relative_seat
            )
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.game.state.board.robber_coordinate == tile
            break
        action = random.choice(info["valid_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=_)
    assert found_choice


def test_discard_via_env_chooses_resource():
    env = CatanatronEnv(
        config={
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.WHITE),
                RandomPlayer(Color.ORANGE),
            ]
        }
    )
    obs, info = env.reset(seed=3)

    found_discard = False
    for _ in range(2000):
        discard_actions = [
            a
            for a in info["valid_actions"]
            if ACTIONS_ARRAY[a][0] == ActionType.DISCARD
        ]
        if discard_actions:
            found_discard = True
            action = discard_actions[0]
            order_index = ACTIONS_ARRAY[action][1]
            assert order_index in range(24)
            before = env.invalid_actions_count
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.invalid_actions_count == before  # decoded to a legal set
            break
        action = random.choice(info["valid_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=_)
    assert found_discard


def test_p0_can_accept_and_reject_trades():
    env = CatanatronEnv(
        config={
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.WHITE),
                RandomPlayer(Color.ORANGE),
            ]
        }
    )
    obs, info = env.reset(seed=11)

    seen_decide_trade = False
    for _ in range(3000):
        decide_actions = [
            a
            for a in info["valid_actions"]
            if ACTIONS_ARRAY[a][0] in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
        ]
        if decide_actions:
            seen_decide_trade = True
            action = random.choice(decide_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            continue
        action = random.choice(info["valid_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=_)
    assert seen_decide_trade
