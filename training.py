from utt_smart.net import OPTIMIZER
from utt_smart.training.train import training


def main() -> None:
    training(optimizer=OPTIMIZER, generations=10, games_per_gen=20, n_games_simulate=100, batch_samples=65, batch_size=64)

if __name__ == "__main__":
    main()
