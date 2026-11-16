import torch

from utt_smart.utt import EMPTY, SUBGAME_CELLS, other_token


def encode(game, to_move):
    # The input planes (see net.IN_PLANES), from the point of view of the player
    # about to move. Built as Python lists and turned into a tensor in one call:
    # every small torch op releases the GIL, which stalls self-play running on
    # many threads
    opponent = other_token(to_move)
    board = game.board
    mine = [1.0 if cell == to_move else 0.0 for cell in board]
    theirs = [1.0 if cell == opponent else 0.0 for cell in board]
    legal = [0.0] * 81
    for move in game.valid_moves():
        legal[move] = 1.0
    # Who owns each subgame, spread over all 9 of its cells so it lines up with
    # the pieces under the network's convolutions
    won = [0.0] * 81
    lost = [0.0] * 81
    drawn = [0.0] * 81
    for s, owner in enumerate(game.subgames):
        if owner != EMPTY:
            plane = won if owner == to_move else lost if owner == opponent else drawn
            for i in SUBGAME_CELLS[s]:
                plane[i] = 1.0
    # X moves first and wins more often, so the value depends on which side is to move
    is_x = [1.0 if to_move == 'X' else 0.0] * 81
    return torch.tensor([mine, theirs, legal, won, lost, drawn, is_x]).view(7, 9, 9)
