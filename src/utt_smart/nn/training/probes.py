import torch
import torch.nn.functional as F
from utt_smart.nn.encoding import encode
from utt_smart.utt import UTT


# A few fixed positions the network is asked about after every generation, so
# you can watch what it believes change as it trains: where it wants to play,
# and how good it thinks the position is for the side to move


class Probe:
    def __init__(self, name, title, game, to_move, key_move=None, note=""):
        self.name = name
        self.title = title
        self.game = game
        self.to_move = to_move
        self.key_move = key_move  # the move a strong player would make, if there is one
        self.note = note


def _win_now():
    # X has won subgames 0 and 1 (the top row of each) and has cells 6 and 7 of
    # subgame 2. O's last move, cell 32, sends X to subgame 2, where cell 8 wins
    # it and with it the top row of the big board (the position in
    # tests/test_training.py::test_search_takes_a_winning_move)
    game = UTT()
    for move in (0, 1, 2, 3, 4, 5, 6, 7):
        game.board[move] = 'X'
    game.subgames[0] = game.subgames[1] = 'X'
    game.board[32] = 'O'
    game.moves = [32]
    return game


def _after(moves):
    game, token = UTT(), 'X'
    for move in moves:
        game.apply_move(move, token)
        token = 'O' if token == 'X' else 'X'
    return game, token


def probe_positions():
    centre, centre_to_move = _after([40])
    return [
        Probe("opening", "Empty board, X to move", UTT(), 'X',
              note="Where does the network like to start? X moves first, so a value above 0 means it has "
                   "learned that X is favoured."),
        Probe("reply_to_centre", "X took the centre cell, O to move", centre, centre_to_move,
              note="O has to play in the centre subgame. Each cell O picks sends X to that subgame."),
        Probe("win_now", "X wins with cell 8", _win_now(), 'X', key_move=8,
              note="One move wins the game. A network that has learned to finish games puts nearly all "
                   "its probability on cell 8 and a value near +1."),
    ]


@torch.no_grad()
def evaluate_probes(bot, probes, device):
    # The network's own policy and value for each probe, with no search
    was_training = bot.training
    bot.eval()
    states = torch.stack([encode(p.game, p.to_move) for p in probes]).to(device)
    logits, values = bot(states)
    policies = F.softmax(logits, dim=1).cpu()
    bot.train(was_training)
    return [{'name': p.name, 'value': round(values[i].item(), 4),
             'policy': [round(x, 4) for x in policies[i].tolist()]}
            for i, p in enumerate(probes)]
