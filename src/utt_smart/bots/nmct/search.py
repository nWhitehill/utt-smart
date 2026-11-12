import math
import random
import torch
from utt_smart.nn.encoding import encode
from utt_smart.utt import UTT, DRAW, other_token

# How strongly the network's prior pulls the search towards a move, relative
# to the value the search has measured for it
C_PUCT = 1.5

# Self-play mixes Dirichlet noise into the root priors so the search sometimes
# tries moves the network doesn't yet like, as in AlphaZero. The number of legal
# moves swings from 1 to 81, so alpha is scaled to it (KataGo does the same):
# about DIRICHLET_TOTAL / moves, which keeps the noise from piling onto one or
# two moves when there are many, or spreading flat when there are few
DIRICHLET_TOTAL = 10.0
DIRICHLET_MAX_ALPHA = 1.0
NOISE_FRACTION = 0.25


class Node:
    # token is the player who made `move` to reach this node, so value_sum is
    # counted from their point of view: +1 a win, -1 a loss, 0 a draw
    def __init__(self, parent, move, token, prior):
        self.parent = parent
        self.move = move
        self.token = token
        self.prior = prior
        self.children = None
        self.visits = 0
        self.value_sum = 0.0

    def is_leaf(self):
        return self.children is None

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


def terminal_value(winner, token):
    if winner == DRAW:
        return 0.0
    return 1.0 if winner == token else -1.0


def select_child(node):
    # PUCT, as in AlphaZero: the measured value of a child plus a bonus that
    # is large for moves the network likes and shrinks as they get visited
    sqrt_parent = math.sqrt(node.visits)
    return max(node.children, key=lambda child: child.q()
               + C_PUCT * child.prior * sqrt_parent / (1 + child.visits))


class NeuralMC:
    def __init__(self, eval_lambda, game=None, to_move='X', noise=False):
        # Takes an encoded position, returns (move probabilities as an 81-long
        # list, value for the player to move)
        self.eval = eval_lambda
        self.noise = noise
        self.game = game.copy() if game is not None else UTT()
        # The root stands for the current position, reached by the other player's move
        self.root = Node(None, None, other_token(to_move), 1.0)
        self.root_noised = False

    def evaluate(self, game, to_move):
        # Returns the network's move probabilities and its value for to_move
        return self.eval(encode(game, to_move))

    def start(self, iterations):
        # A root kept from the previous move is already expanded, so it gets its
        # noise here rather than when it is first expanded in simulate()
        if self.noise and not self.root_noised and not self.root.is_leaf():
            self.add_root_noise()
        for _ in range(iterations):
            self.simulate()

    def advance(self, move):
        # Plays move and keeps the subtree below it, so the next search starts
        # with the visits this one already spent on that position
        child = next((c for c in self.root.children or [] if c.move == move), None)
        self.game.apply_move(move, other_token(self.root.token))
        if child is None:
            child = Node(None, move, other_token(self.root.token), 1.0)
        child.parent = None
        self.root = child
        self.root_noised = False

    def simulate(self):
        node = self.root
        game = self.game.copy()

        # Walk down the tree until we reach a position we haven't expanded yet
        while not node.is_leaf():
            node = select_child(node)
            game.apply_move(node.move, node.token, validate=False)

        if game.winner is not None:
            value = terminal_value(game.winner, node.token)
        else:
            # Instead of a random playout, ask the network. Its value is for the
            # player to move, who is the opponent of node.token
            to_move = other_token(node.token)
            probs, net_value = self.evaluate(game, to_move)
            node.children = [Node(node, move, to_move, probs[move])
                             for move in game.valid_moves()]
            value = -net_value
            if node is self.root and self.noise:
                self.add_root_noise()

        # Back up the result, flipping sign each level as the players alternate
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def add_root_noise(self):
        children = self.root.children
        alpha = min(DIRICHLET_MAX_ALPHA, DIRICHLET_TOTAL / len(children))
        # random.gammavariate rather than torch's Dirichlet: one torch call per
        # move is slow and releases the GIL (see encode)
        noise = [random.gammavariate(alpha, 1.0) for _ in children]
        total = sum(noise) or 1.0
        for child, n in zip(children, noise):
            child.prior = (1 - NOISE_FRACTION) * child.prior + NOISE_FRACTION * n / total
        self.root_noised = True

    def sample_move(self):
        # Picks a move with probability proportional to its visit count, so
        # self-play games differ from each other
        children = self.root.children
        return random.choices(children, weights=[child.visits for child in children])[0].move

    def best_move(self):
        best = max(self.root.children, key=lambda node: node.visits)
        return best.move

    def root_value(self):
        # The search's value for the player to move
        return -self.root.q()

    def policy(self):
        # Visit counts as an 81-long probability distribution: the training
        # target for the network's policy head
        pi = [0.0] * 81
        for child in self.root.children:
            pi[child.move] = child.visits
        total = sum(pi)
        return torch.tensor([p / total for p in pi])
