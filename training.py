from utt_smart.net import OPTIMIZER
from utt_smart.training.train import training


def main() -> None:
    # 100 games give ~5000 positions per generation, so the 10-generation replay
    # buffer holds ~50k; 200 batches of 256 is about one pass over it
    training(optimizer=OPTIMIZER, generations=75, games_per_gen=100, n_games_simulate=50, batch_samples=200, batch_size=256,
             buffer_generations=10, mct_iterations=200)

if __name__ == "__main__":
    main()
