import pathlib
import random
import subprocess
import sys
import time

from utt_smart.bots.harness import FROM_SERVER, TO_SERVER, Harness, parse_position
from utt_smart.utt import CELL_SUBGAME, EMPTY, UTT, other_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOTS = ROOT / "src" / "utt_smart" / "bots"


def server_message(prefix, game):
    # A board-state line as the server builds it (RunnerActionService.playMessage):
    # cells in board-major order, "" for empty, the last move as a board-major
    # index or "?", and every board the next move may go in
    flat = ",".join("" if game.board[FROM_SERVER[s]] == EMPTY else game.board[FROM_SERVER[s]]
                    for s in range(81))
    last = str(TO_SERVER[game.moves[-1]]) if game.moves else "?"
    boards = sorted({CELL_SUBGAME[m] for m in game.valid_moves()})
    return f"{prefix} {flat} {last} {','.join(map(str, boards))}"


def test_index_conversion_round_trips():
    assert sorted(TO_SERVER) == list(range(81))
    assert all(FROM_SERVER[TO_SERVER[i]] == i for i in range(81))
    # Row 0 of the grid runs across the top rows of boards 0, 1 and 2
    assert [TO_SERVER[i] for i in range(9)] == [0, 1, 2, 9, 10, 11, 18, 19, 20]


def test_parse_position_matches_the_game_it_came_from():
    rng = random.Random(0)
    for _ in range(50):
        game, token = UTT(), 'X'
        while not game.is_over():
            parsed, legal = parse_position(server_message("move", game))
            assert parsed.board == game.board
            assert parsed.subgames == game.subgames
            assert parsed.winner is None
            assert sorted(parsed.valid_moves()) == sorted(legal) == sorted(game.valid_moves())
            game.apply_move(rng.choice(game.valid_moves()), token)
            token = other_token(token)
        # A finished game's board gives the same result
        assert UTT.from_board(game.board, game.moves[-1]).winner == game.winner


def test_first_move_with_an_empty_board_field():
    # The docs describe the first move's flatBoard as empty; the server sends 80
    # commas. Both mean every cell is free
    for flat in ("", "," * 80):
        game, legal = parse_position(f"move {flat} ? 0,1,2,3,4,5,6,7,8")
        assert game.moves == [] and sorted(legal) == list(range(81))


class Recorder:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines += [line for line in text.splitlines() if line]

    def flush(self):
        pass


class BrokenBot(Harness):
    name = "broken"

    def choose_move(self, game, to_move, deadline):
        raise RuntimeError("bot crashed")


def test_harness_falls_back_to_a_legal_move():
    out = Recorder()
    harness = BrokenBot()
    harness.start_game('X', 0.01, out)
    game = UTT()
    assert harness.handle(server_message("move", game))
    first = int(out.lines[-1])
    # After a rejection it plays a different legal move
    assert harness.handle(server_message("invalid_move", game))
    assert int(out.lines[-1]) != first
    assert harness.handle("wait for turn")
    assert not harness.handle("end won")


def start_bot(mark, bot, think):
    process = subprocess.Popen(
        [sys.executable, str(BOTS / bot / "bot.py"), "ULTIMATE_TIC_TAC_TOE", mark, "10s", "--think", str(think)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    start = time.monotonic()
    assert process.stdout.readline().strip() == "ready", process.stderr.read()
    # The runner gives up on a bot that isn't ready within 10 s
    assert time.monotonic() - start < 10
    return process


def referee(x_bot, o_bot, think):
    # Plays one game between two bot processes, each started from its bot.py
    # entrypoint, talking to them the way the runner relays the server's lines.
    # Returns the winner
    players = {'X': start_bot('X', x_bot, think), 'O': start_bot('O', o_bot, think)}
    game, to_move = UTT(), 'X'
    try:
        while not game.is_over():
            player = players[to_move]
            player.stdin.write(server_message("move", game) + "\n")
            player.stdin.flush()
            started = time.monotonic()
            reply = player.stdout.readline().strip()
            assert time.monotonic() - started < 10, "missed the server's move timeout"
            move = FROM_SERVER[int(reply)]
            assert move in game.valid_moves(), f"{to_move} played illegal move {reply}"
            game.apply_move(move, to_move)
            if not game.is_over():
                player.stdin.write("wait for turn\n")
                player.stdin.flush()
            to_move = other_token(to_move)
        ends = {}
        for mark, player in players.items():
            ends[mark] = ("end draw" if game.winner not in ('X', 'O')
                          else "end won" if game.winner == mark else "end loss")
            player.stdin.write(ends[mark] + "\n")
            player.stdin.flush()
            assert player.wait(timeout=10) == 0, player.stderr.read()
        return game.winner
    finally:
        for player in players.values():
            if player.poll() is None:
                player.kill()


def test_full_game_between_mct_bots():
    assert referee('mct', 'mct', 0.02) in ('X', 'O', 'd')


def test_full_game_with_the_network_bot():
    assert referee('nmct', 'mct', 0.05) in ('X', 'O', 'd')


def test_mct_entrypoint_runs_without_torch():
    # The mct bot needs only the standard library. Blocking torch makes any
    # import of it fail, so this catches the harness or mct picking one up
    script = ("import runpy, sys; sys.modules['torch'] = None; "
              f"sys.argv = ['bot.py', 'ULTIMATE_TIC_TAC_TOE', 'X', '10s', '--think', '0.02']; "
              f"runpy.run_path({str(BOTS / 'mct' / 'bot.py')!r}, run_name='__main__')")
    result = subprocess.run([sys.executable, "-c", script], input="end draw\n",
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ready"
