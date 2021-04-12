import logging
import time
from argparse import ArgumentParser
from typing import List

from aiutils import save
from config import GameConfig
from constants import BUY
from enums import Phase
from env import DefaultEnvironment, Environment
from mcts import GameTree
from player import load_players
from simulationdata import SimulationData
from state import DecisionResponse, DecisionState, FeatureType, State
from supply import Supply
from tqdm import tqdm


def simulate(env: Environment, n: int, trees: List[GameTree], turn_log=False, action_log=False, card_log=False) -> SimulationData:
    # TODO: Fix this shit
    sim_data = SimulationData(Supply(env.config).get_supply_card_types())

    for i in tqdm(range(n)):
        state: State = env.reset()
        for tree in trees:
            tree.reset(state)
        done = False
        t_start = time.time()
        starting_player_buy = None

        if card_log:
            sim_data.update_card(i, 0, 0, state.get_card_counts(0))
            sim_data.update_card(i, 1, 0, state.get_card_counts(1))
        if turn_log:
            sim_data.update_turn(i, 0, 0, state.get_player_score(0), None, state.get_coin_density(0))
            sim_data.update_turn(i, 1, 0, state.get_player_score(1), None, state.get_coin_density(1))

        while not done:
            action: DecisionResponse = DecisionResponse([])
            d: DecisionState = state.decision
            pid: int = d.controlling_player
            player = env.players[pid]
            player.makeDecision(state, action)

            if state.phase == Phase.ActionPhase:
                # +1 to turns to get current turn
                sim_data.update_action(i, pid, state.player_states[pid].turns + 1, action.cards[0])

            if state.phase == Phase.BuyPhase and trees:
                for tree in trees:
                    tree.advance(action.single_card)

            log_buy = (state.phase == Phase.BuyPhase)

            obs, reward, done, _ = env.step(action)

            if turn_log and log_buy:
                if pid == 0:
                    starting_player_buy = action.single_card
                else:
                    sim_data.update_turn(i, 0, state.player_states[0].turns, state.get_player_score(0), starting_player_buy, state.get_coin_density(0))
                    sim_data.update_turn(i, 1, state.player_states[1].turns, state.get_player_score(1), action.single_card, state.get_coin_density(1))
            if card_log and log_buy:
                if pid == 1:
                    sim_data.update_card(i, 0, state.player_states[0].turns, state.get_card_counts(0))
                    sim_data.update_card(i, 1, state.player_states[1].turns, state.get_card_counts(1))

        if state.player_states[0].turns > state.player_states[1].turns:
            sim_data.update_card(i, 0, state.player_states[0].turns, state.get_card_counts(0))
            sim_data.update_turn(i, 0, state.player_states[0].turns, state.get_player_score(0), starting_player_buy, state.get_coin_density(0))

        t_end = time.time()
        sim_data.update(env.game, t_end - t_start)

    sim_data.finalize(env.game)

    print('===SUMMARY===')
    print(sim_data.summary)

    return sim_data


def main(args: ArgumentParser):
    if args.debug:
        logging.basicConfig(level=logging.INFO)

    config = GameConfig(prosperity=args.prosperity, num_players=len(args.players), sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    if args.tree_path:
        tree = GameTree.load(args.tree_path, False)
    else:
        tree = None

    players = load_players(args.players, args.models, tree=tree, train=False, rollout_type=args.rollout_type)
    logger = logging.getLogger()

    if args.log_buys:
        logger.setLevel(BUY)

    env = DefaultEnvironment(config, players, logger=logger)
    sim_data = simulate(env, args.n, tree)

    if args.save_data:
        save(args.data_path, sim_data)


if __name__ == '__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('-n', type=int, required=True, help='Number of games to simulate')
    parser.add_argument('-T', type=int, default=None, help='Upper threshold for number of turns in each game')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('--sandbox', action='store_true', help='When set, the supply is limited to the 7 basic kingdom supply cards.')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--tree-path', type=str, help='Path to game tree.')
    parser.add_argument('--players', nargs='+', type=str, help='Strategy of AI opponent.')
    parser.add_argument('--rollout-type', type=str, help='Type of rollout model.')
    parser.add_argument('--device', default='cuda', type=str, help='Hardware to use for neural network models.')
    parser.add_argument('--models', nargs='+', type=str, help='Path to AI models')
    parser.add_argument('--log-buys', action='store_true', help='Whether or not to log buys')
    parser.add_argument('--save_data', action='store_true', help='Whether the data should be saved')
    parser.add_argument('--data_path', type=str, help='Where to save data file')
    parser.add_argument('--debug', action='store_true', help='Turn logging settings to DEBUG')

    args = parser.parse_args()

    main(args)
