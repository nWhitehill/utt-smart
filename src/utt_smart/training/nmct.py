import math
import random
import torch
from torch.nn import MSELoss
from utt_smart.training.utt import UTT, DRAW
from utt_smart.training.mct import other_token

# How strongly the network's prior pulls the search towards a move, relative
# to the value the search has measured for it
C_PUCT = 1.5

# Self-play mixes Dirichlet noise into the root priors so the search sometimes
# tries moves the network doesn't yet like, as in AlphaZero
DIRICHLET_ALPHA = 0.3
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


def encode(game, to_move):
    # The 3 input planes, from the point of view of the player about to move
    opponent = other_token(to_move)
    planes = torch.zeros(3, 81)
    for i, cell in enumerate(game.board):
        if cell == to_move:
            planes[0, i] = 1
        elif cell == opponent:
            planes[1, i] = 1
    planes[2, game.valid_moves()] = 1
    return planes.view(3, 9, 9)


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
    def __init__(self, net, game=None, to_move='X', noise=False):
        self.net = net
        self.noise = noise
        self.game = game.copy() if game is not None else UTT()
        # The root stands for the current position, reached by the other player's move
        self.root = Node(None, None, other_token(to_move), 1.0)

    @torch.inference_mode()
    def evaluate(self, game, to_move):
        # Returns the network's move probabilities and its value for to_move
        self.net.eval()
        logits, value = self.net(encode(game, to_move).unsqueeze(0))
        return torch.softmax(logits[0], dim=0), value.item()

    def start(self, iterations):
        for _ in range(iterations):
            self.simulate()

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
            node.children = [Node(node, move, to_move, probs[move].item())
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
        noise = torch.distributions.Dirichlet(torch.full((len(children),), DIRICHLET_ALPHA)).sample()
        for child, n in zip(children, noise):
            child.prior = (1 - NOISE_FRACTION) * child.prior + NOISE_FRACTION * n.item()

    def sample_move(self):
        # Picks a move with probability proportional to its visit count, so
        # self-play games differ from each other
        children = self.root.children
        return random.choices(children, weights=[child.visits for child in children])[0].move

    def best_move(self):
        best = max(self.root.children, key=lambda node: node.visits)
        return best.move

    def policy(self):
        # Visit counts as an 81-long probability distribution: the training
        # target for the network's policy head
        pi = torch.zeros(81)
        for child in self.root.children:
            pi[child.move] = child.visits
        return pi / pi.sum()