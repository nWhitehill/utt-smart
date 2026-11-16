"""Entrypoint for the mct bot on Bots Playing Games:

    python3 lib/utt_smart/bots/mct/bot.py ULTIMATE_TIC_TAC_TOE X 10s    # in the tarball
    uv run python src/utt_smart/bots/mct/bot.py ULTIMATE_TIC_TAC_TOE X 10s
"""
import pathlib
import sys

if __name__ == "__main__":
    # Run as a script, Python puts this file's directory first on sys.path, where
    # search.py would shadow any top-level `search` module. Swap it for the
    # directory holding utt_smart: lib/ in the tarball, src/ in the repo
    sys.path[0] = str(pathlib.Path(__file__).resolve().parents[3])

from utt_smart.bots import search_until  # noqa: E402
from utt_smart.bots.harness import Harness  # noqa: E402
from utt_smart.bots.mct.search import MC  # noqa: E402


class MCTBot(Harness):
    # Plain Monte Carlo tree search with random playouts. Standard library only,
    # so it runs anywhere Python does
    name = "mct"

    def choose_move(self, game, to_move, deadline):
        search = MC(game, to_move)
        self.last_iterations = search_until(search, deadline)
        return search.best_move()


if __name__ == "__main__":
    MCTBot.main()
