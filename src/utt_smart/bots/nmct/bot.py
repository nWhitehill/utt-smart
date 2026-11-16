"""Entrypoint for the nmct bot on Bots Playing Games:

    python3 lib/utt_smart/bots/nmct/bot.py ULTIMATE_TIC_TAC_TOE X 10s    # in the tarball
    uv run python src/utt_smart/bots/nmct/bot.py ULTIMATE_TIC_TAC_TOE X 10s
"""
import pathlib
import sys

if __name__ == "__main__":
    # Run as a script, Python puts this file's directory first on sys.path, where
    # search.py would shadow any top-level `search` module. Swap it for the
    # directory holding utt_smart: lib/ in the tarball, src/ in the repo
    sys.path[0] = str(pathlib.Path(__file__).resolve().parents[3])

import torch  # noqa: E402
from utt_smart.bots import search_until  # noqa: E402
from utt_smart.bots.harness import Harness  # noqa: E402
from utt_smart.bots.nmct.search import NeuralMC  # noqa: E402
from utt_smart.nn.net import load_net  # noqa: E402

# The network it plays with: models/nmct.pt, next to lib/ in the tarball and to
# src/ in the repo, so four levels up from this file in both
MODEL_PATH = pathlib.Path(__file__).resolve().parents[4] / "models" / "nmct.pt"


class NeuralMCTBot(Harness):
    # Tree search guided by the trained network (PUCT), with no exploration
    # noise. Needs torch and a checkpoint
    name = "nmct"

    def __init__(self, checkpoint=MODEL_PATH):
        # One position at a time is too small a job to spread over threads
        torch.set_num_threads(1)
        self.net = load_net(checkpoint).eval()

    def evaluate(self, encoded):
        with torch.inference_mode():
            logits, value = self.net(encoded.unsqueeze(0))
        return torch.softmax(logits, dim=1)[0].tolist(), value.item()

    def choose_move(self, game, to_move, deadline):
        search = NeuralMC(self.evaluate, game, to_move)
        self.last_iterations = search_until(search, deadline)
        return search.best_move()


if __name__ == "__main__":
    NeuralMCTBot.main()
