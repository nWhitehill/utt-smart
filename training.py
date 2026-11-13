import argparse
import os
from utt_smart.nn.training.train import training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the network by self-play. Resumes from trained.pt if it exists")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help=f"worker processes for self-play and evaluation, one core each (default: all {os.cpu_count()})")
    parser.add_argument("--generations", type=int, default=100, help="generations to run (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--device", default=None, help="device to train on (default: cuda if available, else cpu)")
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be at least 1")

    # A quarter of moves get a full 200-simulation search and become policy
    # targets; the rest get 40 simulations, so a move costs ~80 network calls on
    # average. 400 games give ~22k positions per generation, and the 8-generation
    # buffer holds ~175k. Each position is trained on about 4 times while it is
    # in the buffer, which the loss on fresh positions shows doesn't overfit
    training(generations=args.generations, games_per_gen=400, full_sims=200, fast_sims=40, full_search_prob=0.25,
             batch_size=512, samples_per_position=4, buffer_generations=8, eval_games=100, mct_iterations=1000,
             workers=args.cores, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
