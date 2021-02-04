from env import Environment
from player import Player, RolloutPlayer
from state import DecisionResponse, DecisionState
from tqdm import tqdm


def train_elog(env: Environment, epochs: int, train_epochs_interval: int):
    for epoch in tqdm(range(epochs)):
        state = env.reset()
        done = False
        data = {'features': [], 'rewards': [], 'cards': [], 'idxs': state.feature.idxs}
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            player: Player = env.players[d.controlling_player]

            player.makeDecision(state, action)

            x = state.feature.to_numpy()
            data['features'].append(x)
            data['cards'].append(action.single_card)

            obs, reward, done, _ = env.step(action)

        data['rewards'].extend([reward] * (len(data['features']) - len(data['rewards'])))

        for player in env.players:
            if isinstance(player, RolloutPlayer):
                player.rollout.update(**data)
                if (epoch + 1) % train_epochs_interval == 0:
                    player.rollout.learn()
