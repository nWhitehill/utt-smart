import argparse
import os

from utt_smart.nn.net import TRAINED_PATH, load_net
from utt_smart.nn.training.train import evaluate_vs_mct
from utt_smart.nn.training.train_multi import GamePool


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a saved network against plain MCT with random playouts")
    parser.add_argument("checkpoint", nargs="?", default=str(TRAINED_PATH), help="network to play (default: trained.pt)")
    parser.add_argument("--games", type=int, default=200, help="games to play, half as X and half as O (default: 200)")
    parser.add_argument("--sims", type=int, nargs="+", default=[200],
                        help="the network's simulations per move; several values run one match each (default: 200)")
    parser.add_argument("--mct", type=int, nargs="+", default=[1000],
                        help="MCT iterations per move; several values run one match each (default: 1000)")
    parser.add_argument("--cores", type=int, default=os.cpu_count(), help="worker processes (default: all)")
    args = parser.parse_args()

    bot = load_net(args.checkpoint)
    with GamePool(args.cores) as pool:
        for mct_iterations in args.mct:
            for sims in args.sims:
                evaluate_vs_mct(pool, bot, args.games, sims, mct_iterations)


if __name__ == "__main__":
    main()
