"""The Bots Playing Games side of a bot: a base class that plays one game over stdin/stdout.

A bot subclasses Harness, implements choose_move, and ends its file with

    if __name__ == "__main__":
        MyBot.main()

so the file itself is the entrypoint the sandbox runner starts. The runner relays
lines between it and the game server (see the ultimate-tic-tac-toe server repo,
RunnerActionService.kt):

  argv     ULTIMATE_TIC_TAC_TOE <X|O> <clock>       clock is the time per move, e.g. "10s"
  startup  print `ready` within 10 s
  stdin    move <flatBoard> <lastMove> <validBoards>          our turn
           invalid_move <flatBoard> <lastMove> <validBoards>  our last reply was rejected
           wait for turn                                       our move landed
           end won | end loss | end draw | end forfeight | end cancelled
  stdout   one cell index 0..80 per turn
  stderr   diagnostics, stored as the game's logs

The server numbers cells board by board: index = boardIndex * 9 + cellIndex, and
flatBoard lists the 81 cells ("", "X" or "O") in that order. UTT numbers them
row by row across the whole 9x9 grid, so the harness converts both ways.

This module only uses the standard library, so a bot that doesn't need torch
can run without it.
"""
import argparse
import random
import sys
import time
import traceback

from utt_smart.utt import CELL_POS, CELL_SUBGAME, EMPTY, SUBGAME_CELLS, UTT

# Share of the move clock spent searching. The rest is margin for reading the
# position, a slow or busy sandbox, and the trip back through the runner
THINK_FRACTION = 0.5
DEFAULT_CLOCK_SECONDS = 10.0

# Cell index conversions: UTT's row-major index <-> the server's board-major index
TO_SERVER = [CELL_SUBGAME[i] * 9 + CELL_POS[i] for i in range(81)]
FROM_SERVER = [SUBGAME_CELLS[s // 9][s % 9] for s in range(81)]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def parse_clock(clock):
    # "10s" or "500ms" to seconds
    try:
        if clock.endswith("ms"):
            return float(clock[:-2]) / 1000
        return float(clock.rstrip("s"))
    except ValueError:
        log(f"can't read clock {clock!r}; assuming {DEFAULT_CLOCK_SECONDS:g}s")
        return DEFAULT_CLOCK_SECONDS


def parse_position(line):
    # A `move`/`invalid_move` line to (game, legal moves), both in UTT's numbering.
    # The legal moves are worked out from the server's validBoards, so they are
    # right even if UTT's rules ever disagree with the server's
    parts = line.split(" ")
    parts += [""] * (4 - len(parts))
    flat, last, boards = parts[1], parts[2], parts[3]
    cells = flat.split(",") if flat else []
    board = [EMPTY] * 81
    for s, cell in enumerate(cells[:81]):
        if cell in ("X", "O"):
            board[FROM_SERVER[s]] = cell
    last_move = FROM_SERVER[int(last)] if last.isdigit() and int(last) < 81 else None
    valid_boards = [int(b) for b in boards.split(",") if b.isdigit() and int(b) < 9]
    legal = [FROM_SERVER[b * 9 + c] for b in valid_boards for c in range(9)
             if board[FROM_SERVER[b * 9 + c]] == EMPTY]
    return UTT.from_board(board, last_move), legal


class Harness:
    # Subclasses set name, load anything slow (torch, weights) in __init__, which
    # main() runs before saying `ready`, and implement choose_move
    name = "bot"

    def choose_move(self, game, to_move, deadline):
        # The move to play in game for to_move ('X' or 'O'), as a cell index in
        # UTT's row-major numbering, by time.monotonic() time deadline. May set
        # self.last_iterations for the log
        raise NotImplementedError

    @classmethod
    def main(cls, argv=None):
        # The entrypoint: reads the runner's arguments, builds the bot, says
        # `ready`, and plays one game on stdin/stdout
        parser = argparse.ArgumentParser(description=f"Play one Bots Playing Games match as {cls.name}")
        parser.add_argument("game", help="must be ULTIMATE_TIC_TAC_TOE")
        parser.add_argument("mark", choices=["X", "O"], help="the side this bot plays")
        parser.add_argument("clock", help='time per move, e.g. "10s"')
        parser.add_argument("--think", type=float,
                            help="seconds to search per move, for local testing (default: half the clock)")
        args = parser.parse_args(argv)
        if args.game != "ULTIMATE_TIC_TAC_TOE":
            parser.error(f"unsupported game {args.game!r}")
        think = args.think if args.think is not None else parse_clock(args.clock) * THINK_FRACTION

        # Build the bot before saying `ready`: the runner allows 10 s for startup,
        # and anything slow (importing torch, loading weights) shouldn't eat into a move
        start = time.monotonic()
        bot = cls()
        log(f"{bot.name} bot playing {args.mark}, {think:g}s per move, loaded in {time.monotonic() - start:.2f}s")
        print("ready", flush=True)
        bot.play(args.mark, think)

    def play(self, mark, think_seconds, inp=sys.stdin, out=sys.stdout):
        self.start_game(mark, think_seconds, out)
        # readline rather than `for line in inp`, which reads ahead and can
        # stall a turn-by-turn exchange over a pipe
        while True:
            raw = inp.readline()
            if raw == "":
                break
            line = raw.rstrip("\r\n")
            if line and not self.handle(line):
                break

    def start_game(self, mark, think_seconds, out=sys.stdout):
        self.mark = mark
        self.think_seconds = think_seconds
        self.out = out
        self.last_sent = None
        self.rejected = set()  # this turn's moves the server turned down, server numbering

    def handle(self, line):
        # Handles one line from the server. Returns False once the game is over
        if line.startswith(("move ", "invalid_move ")):
            deadline = time.monotonic() + self.think_seconds
            if line.startswith("move "):
                self.rejected = set()
            else:
                self.rejected.add(self.last_sent)
                log(f"move {self.last_sent} was rejected")
            game, legal = parse_position(line)
            self.last_sent = self.pick(game, legal, deadline)
            print(self.last_sent, file=self.out, flush=True)
            return True
        if line == "wait for turn":
            return True
        if line.startswith("end"):
            log(f"game over: {line}")
            return False
        log(f"ignoring unrecognised line: {line!r}")
        return True

    def pick(self, game, legal, deadline):
        # The move to send, in the server's numbering. Falls back to a random
        # legal move if choose_move fails or picks something the server won't
        # take, since three rejected moves forfeit the game
        candidates = [m for m in legal if TO_SERVER[m] not in self.rejected]
        if not candidates:
            log("no legal move in the message; playing any empty cell")
            candidates = [m for m in range(81) if game.board[m] == EMPTY] or [0]
            return TO_SERVER[random.choice(candidates)]
        if game.winner is None and set(game.valid_moves()) == set(legal) and not self.rejected:
            start = time.monotonic()
            try:
                move = self.choose_move(game, self.mark, deadline)
            except Exception:
                log(traceback.format_exc())
                move = None
            if move in candidates:
                log(f"{self.name}: played {TO_SERVER[move]} after {time.monotonic() - start:.2f}s, "
                    f"{getattr(self, 'last_iterations', '?')} iterations")
                return TO_SERVER[move]
            log(f"{self.name} picked {move}, which isn't legal here")
        elif not self.rejected:
            log("the position doesn't match UTT's rules; playing a random legal move")
        return TO_SERVER[random.choice(candidates)]
