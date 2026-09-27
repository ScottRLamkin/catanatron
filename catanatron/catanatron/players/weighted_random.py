import random

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.players.tree_search_utils import prune_discard_actions


WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}


class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def decide(self, game, playable_actions):
        # Policy: discard one of the keep-rule sets (at most 24) rather than
        # a uniformly random legal set.
        playable_actions = prune_discard_actions(playable_actions, game.state)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
