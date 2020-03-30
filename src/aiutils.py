import pickle
from mcts import Node

def load(checkpoint: str):
    node = pickle.load(open(checkpoint, 'rb'))
    return node

def save(root: Node, i: int):
    with open (f'mcts_chkpt_{i}.pk1', 'wb') as output:
        pickle.dump(root, output, pickle.HIGHEST_PROTOCOL)