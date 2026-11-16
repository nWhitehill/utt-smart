import random

# The 81 cells are stored row-major over the full 9x9 grid: index = row * 9 + col.
# Subgames are numbered 0-8 left-to-right, top-to-bottom, and so are the cells
# within a subgame.
#
#   cell index  -> (row, col)        row = move // 9, col = move % 9
#   cell index  -> subgame           (row // 3) * 3 + (col // 3)
#   cell index  -> cell in subgame   (row % 3) * 3 + (col % 3)
#   subgame     -> top-left cell     (subgame // 3) * 27 + (subgame % 3) * 3

EMPTY = '-'
DRAW = 'd'

# Lookup tables so the hot paths in apply_move and valid_moves avoid recomputing
# the index math on every call
CELL_SUBGAME = [(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)]
CELL_POS = [(i // 9 % 3) * 3 + (i % 9 % 3) for i in range(81)]
SUBGAME_CELLS = [[(s // 3) * 27 + (s % 3) * 3 + r * 9 + c for r in range(3) for c in range(3)]
                 for s in range(9)]

# The 8 winning lines of a 3x3 grid, as positions 0-8
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]
LINES_THROUGH = [[line for line in LINES if pos in line] for pos in range(9)]


def other_token(token):
    return 'O' if token == 'X' else 'X'


class UTT:
    def __init__(self):
        self.board = [EMPTY for _ in range(81)]
        self.subgames = [EMPTY for _ in range(9)]
        self.moves = []
        self.winner = None  # 'X'/'O' token, DRAW, or None while in progress

    @classmethod
    def from_board(cls, board, last_move):
        # A game in progress from its 81 cells ('X', 'O' or EMPTY) and the last
        # move played (None before the first move). The move history isn't
        # needed: the subgame owners follow from the board, and the last move
        # alone decides where the next move may go
        game = cls()
        game.board = list(board)
        game.moves = [] if last_move is None else [last_move]
        for s, cells in enumerate(SUBGAME_CELLS):
            for a, b, c in LINES:
                if board[cells[a]] != EMPTY and board[cells[a]] == board[cells[b]] == board[cells[c]]:
                    game.subgames[s] = board[cells[a]]
                    break
            else:
                if all(board[i] != EMPTY for i in cells):
                    game.subgames[s] = DRAW
        for a, b, c in LINES:
            if game.subgames[a] in ('X', 'O') and game.subgames[a] == game.subgames[b] == game.subgames[c]:
                game.winner = game.subgames[a]
                return game
        if EMPTY not in game.subgames:
            game.winner = DRAW
        return game

    def copy(self):
        game = UTT.__new__(UTT)
        game.board = self.board.copy()
        game.subgames = self.subgames.copy()
        game.moves = self.moves.copy()
        game.winner = self.winner
        return game

    def is_over(self):
        return self.winner is not None

    def finish_random_game(self, curr_token):
        while self.winner is None:
            self.apply_move(random.choice(self.valid_moves()), curr_token, validate=False)
            curr_token = 'O' if curr_token == 'X' else 'X'
        return self.winner

    def apply_move(self, move, token, validate=True):
        # Pass validate=False only for moves taken from valid_moves()
        if validate and move not in self.valid_moves():
            raise ValueError(f"Invalid move: {move}")

        board = self.board
        board[move] = token
        self.moves.append(move)

        # Only the lines through the cell just played can have been completed
        subgame_i = CELL_SUBGAME[move]
        cells = SUBGAME_CELLS[subgame_i]
        subgames = self.subgames
        for a, b, c in LINES_THROUGH[CELL_POS[move]]:
            if board[cells[a]] == board[cells[b]] == board[cells[c]] == token:
                subgames[subgame_i] = token
                break
        else:
            for i in cells:
                if board[i] == EMPTY:
                    return
            subgames[subgame_i] = DRAW

        # A subgame just closed. If it was won, the same check applies to the big board
        for a, b, c in LINES_THROUGH[subgame_i]:
            if subgames[a] == subgames[b] == subgames[c] == token:
                self.winner = token
                return
        # Every open subgame still has an empty cell, so no open subgames means no moves
        if EMPTY not in subgames:
            self.winner = DRAW

    def valid_moves(self):
        if self.winner is not None:
            return []
        if len(self.moves) == 0:
            return [i for i in range(81)]

        board = self.board
        subgames = self.subgames

        # The cell played within its subgame decides which subgame is next
        next_subgame = CELL_POS[self.moves[-1]]

        # else they can only play in open spaces in that subgame
        if subgames[next_subgame] == EMPTY:
            return [i for i in SUBGAME_CELLS[next_subgame] if board[i] == EMPTY]

        # If that subgame is already won or drawn they can play in any open subgame
        return [i for s in range(9) if subgames[s] == EMPTY
                for i in SUBGAME_CELLS[s] if board[i] == EMPTY]

    def __str__(self):
        lines = []
        for row in range(9):
            if row > 0 and row % 3 == 0:
                lines.append('------+-------+------')
            cells = self.board[row * 9:row * 9 + 9]
            lines.append(' | '.join(' '.join(cells[c:c + 3]) for c in range(0, 9, 3)))
        return '\n'.join(lines)

