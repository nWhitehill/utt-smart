import math
import random
from utt_smart.utt import UTT, DRAW, other_token

EXPLORATION = 1.41

class Node:
    # token is the player who made `move` to reach this node, so wins are
    # counted from their point of view
    def __init__(self, parent, move, token):
        self.parent = parent
        self.token = token
        self.children = None
        self.move = move
        self.visits = 0
        self.wins = 0

    def is_leaf(self):
        return self.children is None

def score(winner):
    if winner == DRAW:
        return {'X': 0.5, 'O': 0.5}
    return {'X': 1 if winner == 'X' else 0,
            'O': 1 if winner == 'O' else 0}

def select_node(parent, nodes):
    # UCB1: always try unvisited children first, then balance win rate
    # against how little a child has been explored
    for node in nodes:
        if node.visits == 0:
            return node
    log_parent = math.log(parent.visits)
    return max(nodes, key=lambda node: node.wins / node.visits
               + EXPLORATION * math.sqrt(log_parent / node.visits))


class MC:
    def __init__(self, game=None, to_move='X'):
        self.game = game.copy() if game is not None else UTT()
        # The root stands for the current position, reached by the other player's move
        self.root = Node(None, None, other_token(to_move))

    def start(self, iterations):
        for _ in range(iterations):
            self.traverse(self.root, self.game.copy())

    def best_move(self):
        best = max(self.root.children, key=lambda node: node.visits)
        return best.move

    def traverse(self, node, curr_game):
        if (node.is_leaf()):
            if (curr_game.winner == None):
                child_token = other_token(node.token)
                node.children = [Node(node, move, child_token) for move in curr_game.valid_moves()]
                selected_child = random.choice(node.children)
                curr_game.apply_move(selected_child.move, selected_child.token, validate=False)
                curr_game.finish_random_game(other_token(selected_child.token))
                result = score(curr_game.winner)
                selected_child.visits += 1
                selected_child.wins += result[selected_child.token]
            else:
                # Game should be over now
                result = score(curr_game.winner)
        else:
            child = select_node(node, node.children)
            curr_game.apply_move(child.move, child.token, validate=False)
            result = self.traverse(child, curr_game)

        node.visits += 1
        node.wins += result[node.token]
        return result
