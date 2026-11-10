import argparse
import os
from utt_smart.net import OPTIMIZER
from utt_smart.training.train import training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the network by self-play")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help=f"worker processes for self-play and evaluation, one core each (default: all {os.cpu_count()})")
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be at least 1")

    # 100 games give ~5000 positions per generation, so the 10-generation replay
    # buffer holds ~50k; 2.5 passes over it is ~490 batches of 256
    training(optimizer=OPTIMIZER, generations=50, games_per_gen=100, n_games_simulate=50, batch_size=256,
             passes_per_generation=2.5, buffer_generations=10, mct_iterations=200, workers=args.cores)

if __name__ == "__main__":
    main()
