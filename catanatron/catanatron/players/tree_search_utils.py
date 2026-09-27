import math
from collections import defaultdict

from catanatron.game import Game
from catanatron.models.map import number_probability
from catanatron.models.discard_rule import keep_rule_discard_sets
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    CITY,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.state_functions import (
    player_key,
    get_player_buildings,
    get_dev_cards_in_hand,
    get_player_freqdeck,
    get_enemy_colors,
)
from catanatron.features import build_production_features
from catanatron.players.value import value_production

DETERMINISTIC_ACTIONS = set(
    [
        ActionType.END_TURN,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_CITY,
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.MARITIME_TRADE,
        ActionType.DISCARD,  # player picks the set, so truly deterministic
        ActionType.PLAY_MONOPOLY,  # for simplicity... we assume good card-counting and bank is visible...
        # Domestic trade actions only change prompts / move known cards around.
        ActionType.OFFER_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.CONFIRM_TRADE,
        ActionType.CANCEL_TRADE,
    ]
)


def execute_deterministic(game, action):
    copy = game.copy()
    copy.execute(action, validate_action=False)
    return [(copy, 1)]


def execute_spectrum(game: Game, action: Action):
    """Returns [(game_copy, proba), ...] tuples for result of given action.
    Result probas should add up to 1. Does not modify self"""
    if action.action_type in DETERMINISTIC_ACTIONS:
        return execute_deterministic(game, action)
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        results = []

        # Get the possible deck from the perspective of the current player
        # by getting all face down cards
        current_deck = game.state.development_listdeck.copy()
        for color in get_enemy_colors(game.state.colors, action.color):
            for card in DEVELOPMENT_CARDS:
                number = get_dev_cards_in_hand(game.state, color, card)
                current_deck += [card] * number

        for card in set(current_deck):
            option_action = Action(action.color, action.action_type, card)
            option_game = game.copy()
            try:
                option_game.execute(option_action, validate_action=False)
            except Exception:
                # ignore exceptions, since player might imagine impossible outcomes.
                # ignoring means the value function of this node will be flattened,
                # to the one before.
                pass
            results.append((option_game, current_deck.count(card) / len(current_deck)))
        return results
    elif action.action_type == ActionType.ROLL:
        results = []
        for roll in range(2, 13):
            outcome = (roll // 2, math.ceil(roll / 2))

            option_action = Action(action.color, action.action_type, outcome)
            option_game = game.copy()
            option_game.execute(option_action, validate_action=False)
            results.append((option_game, number_probability(roll)))
        return results
    elif action.action_type == ActionType.MOVE_ROBBER:
        (coordinate, robbed_color) = action.value
        if robbed_color is None:  # no one to steal, then deterministic
            return execute_deterministic(game, action)

        results = []
        opponent_hand = get_player_freqdeck(game.state, robbed_color)
        opponent_hand_size = sum(opponent_hand)
        if opponent_hand_size == 0:
            # Nothing to steal
            return execute_deterministic(game, action)

        # Steal is uniform over the victim's cards, so weight each resource by
        # how many of it the victim holds.
        for card, amount in zip(RESOURCES, opponent_hand):
            if amount == 0:
                continue
            option_action = Action(
                action.color,
                action.action_type,
                (coordinate, robbed_color),
            )
            option_action_record = ActionRecord(action=option_action, result=card)
            option_game = game.copy()
            option_game.execute(
                option_action,
                validate_action=False,
                action_record=option_action_record,
            )
            results.append((option_game, amount / opponent_hand_size))
        return results
    else:
        raise RuntimeError("Unknown ActionType " + str(action.action_type))


def expand_spectrum(game, actions):
    """Consumes game if playable_actions not specified"""
    children = defaultdict(list)
    for action in actions:
        outprobas = execute_spectrum(game, action)
        children[action] = outprobas
    return children  # action => (game, proba)[]


def prune_trade_offers(actions):
    """Policy (not a rule): drop OFFER_TRADE actions. Shallow searches can't
    see a trade's payoff (the cards only move after the opponent answers and
    the offerer confirms), so offering only widens the tree."""
    return [a for a in actions if a.action_type != ActionType.OFFER_TRADE]


def prune_discard_actions(actions, state):
    """Policy (not a rule): when the actions are DISCARD choices, keep only
    the (at most 24) distinct sets produced by the keep rule
    (catanatron.models.discard_rule) for the player's hand, in DISCARD_ORDERS
    order. A 20-card hand has 1001 legal sets, which would swamp any search.
    Non-DISCARD actions pass through unchanged."""
    if not actions or actions[0].action_type != ActionType.DISCARD:
        return actions
    color = actions[0].color
    owed = state.player_state[f"{player_key(state, color)}_DISCARD_OWED"]
    hand = get_player_freqdeck(state, color)
    keep_sets = keep_rule_discard_sets(hand, owed)
    legal = set(a.value for a in actions if a.action_type == ActionType.DISCARD)
    pruned = [
        Action(color, ActionType.DISCARD, discard)
        for discard in keep_sets
        if discard in legal
    ]
    return pruned or actions


def prune_bot_actions(playable_actions, state):
    """The default policy pruning shared by the engine bots: no domestic
    trade offers, and DISCARD restricted to the keep-rule sets. Never empty
    when playable_actions is not empty (except when only offers were
    listed, which cannot happen since END_TURN is always available)."""
    actions = prune_trade_offers(playable_actions)
    return prune_discard_actions(actions, state) or playable_actions


def list_prunned_actions(game: Game):
    current_color = game.state.current_color()
    playable_actions = game.playable_actions
    actions = prune_bot_actions(playable_actions, game.state)
    types = set(map(lambda a: a.action_type, playable_actions))

    # Prune Initial Settlements at 1-tile places
    if ActionType.BUILD_SETTLEMENT in types and game.state.is_initial_build_phase:
        actions = filter(
            lambda a: len(game.state.board.map.adjacent_tiles[a.value]) != 1, actions
        )

    # Prune Trading if can hold for resources. Only for rare resources.
    if ActionType.MARITIME_TRADE in types:
        port_resources = game.state.board.get_player_port_resources(current_color)
        has_three_to_one = None in port_resources
        # TODO: for 2:1 ports, skip any 3:1 or 4:1 trades
        # TODO: if can_safely_hold, prune all
        tmp_actions = []
        for action in actions:
            if action.action_type != ActionType.MARITIME_TRADE:
                tmp_actions.append(action)
                continue
            # has 3:1, skip any 4:1 trades
            if has_three_to_one and action.value[3] is not None:
                continue
            tmp_actions.append(action)
        actions = tmp_actions

    if ActionType.MOVE_ROBBER in types:
        actions = prune_robber_actions(current_color, game, actions)

    return list(actions)


def prune_robber_actions(current_color, game, actions):
    """Eliminate all but the most impactful tile (most enemy production
    blocked, summed over all enemies, minus our own blocked production)."""
    enemy_colors = list(get_enemy_colors(game.state.colors, current_color))
    enemy_owned_tiles = set()
    for enemy_color in enemy_colors:
        for node_id in get_player_buildings(game.state, enemy_color, SETTLEMENT):
            enemy_owned_tiles.update(game.state.board.map.adjacent_tiles[node_id])
        for node_id in get_player_buildings(game.state, enemy_color, CITY):
            enemy_owned_tiles.update(game.state.board.map.adjacent_tiles[node_id])

    robber_moves = [
        a
        for a in actions
        if a.action_type == ActionType.MOVE_ROBBER
        and game.state.board.map.tiles[a.value[0]] in enemy_owned_tiles
    ]
    if len(robber_moves) == 0:
        return actions  # nothing to prune against (e.g. enemies own no tiles)

    production_features = build_production_features(True)
    enemy_names = [f"P{i}" for i in range(1, len(game.state.colors))]

    def impact(action):
        game_copy = game.copy()
        game_copy.execute(action, validate_action=False)

        # production_features is from current_color's perspective: P0 is us,
        # P1..Pn are the enemies.
        sample = production_features(game_copy, current_color)
        production = value_production(sample, "P0")
        enemy_production = sum(value_production(sample, name) for name in enemy_names)

        return enemy_production - production

    most_impactful_robber_action = max(
        robber_moves, key=impact
    )  # most production and variety producing
    actions = filter(
        lambda a: a.action_type != ActionType.MOVE_ROBBER
        or a == most_impactful_robber_action,
        # lambda a: a.action_type != ActionType.MOVE_ROBBER or a in robber_moves,
        actions,
    )
    return actions
