from utt_smart.net import BOT, TRAINED_PATH
from utt_smart.training.nmct import NeuralMC, encode
from utt_smart.training.utt import UTT, DRAW
from utt_smart.training.mct import MC, other_token
from utt_smart.training.train_multi import GamePool
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
import torch
from collections import Counter, deque
import csv
from functools import partial
import math
import random
import time

# Self-play samples moves from the visit counts for this many moves, then plays
# the most visited move for the rest of the game
SAMPLED_MOVES = 12

# One row per generation, appended across runs, for training_analysis.ipynb
LOG_PATH = TRAINED_PATH.parent / "runs" / "training_log.csv"

class TrainingMove:
    def __init__(self, encoded_game, move_policy, turn_token):
        self.encoded_game = encoded_game
        self.move_policy = move_policy
        self.turn_token = turn_token
        self.target = None

    def add_target(self, results):
        self.target = results[self.turn_token]

def play_game_nmct(nn_call, game_index, n_games_simulate):
    game_moves = []
    game = UTT()
    turn_to_move = 'X'
    while not game.is_over():
        nmc = NeuralMC(nn_call, game, turn_to_move, noise=True)
        nmc.start(n_games_simulate)
        move = nmc.sample_move() if len(game.moves) < SAMPLED_MOVES else nmc.best_move()
        game_moves.append(TrainingMove(encode(game, turn_to_move), nmc.policy(), turn_to_move))
        game.apply_move(move, turn_to_move)
        turn_to_move = 'O' if turn_to_move == 'X' else 'X'

    results = {}
    if game.winner == DRAW:
        results = {'X': 0, 'O': 0} 
    else:
        results = {'X': 1 if game.winner == 'X' else -1,
                    'O': 1 if game.winner == 'O' else -1}
    for move in game_moves:
        move.add_target(results)
    return game_moves


def generate_batch(pool, bot, n_games_to_play, n_games_simulate):
    # Returns each game's list of moves
    return pool.run(bot, n_games_to_play, partial(play_game_nmct, n_games_simulate=n_games_simulate))


def game_winner(game_moves):
    # X makes the first move, so its target is the result from X's point of view
    return {1: 'X', -1: 'O', 0: DRAW}[game_moves[0].target]


def nn_token_for(game_index):
    # The network alternates between X and O so the first-move advantage evens out
    return 'X' if game_index % 2 == 0 else 'O'


def play_vs_mct(nn_call, game_index, n_games_simulate, mct_iterations):
    # The network plays its best move with no noise, against plain MCT with random playouts
    nn_token = nn_token_for(game_index)
    game = UTT()
    to_move = 'X'
    while not game.is_over():
        if to_move == nn_token:
            searcher = NeuralMC(nn_call, game, to_move)
            searcher.start(n_games_simulate)
        else:
            searcher = MC(game, to_move)
            searcher.start(mct_iterations)
        game.apply_move(searcher.best_move(), to_move)
        to_move = other_token(to_move)
    return game.winner


def evaluate_vs_mct(pool, bot, n_games, n_games_simulate, mct_iterations):
    nn_tokens = [nn_token_for(i) for i in range(n_games)]
    winners = pool.run(bot, n_games, partial(play_vs_mct, n_games_simulate=n_games_simulate, mct_iterations=mct_iterations))
    record = {'X': [0, 0, 0], 'O': [0, 0, 0]}  # wins, draws, losses for the network playing that side
    for nn_token, winner in zip(nn_tokens, winners):
        record[nn_token][0 if winner == nn_token else 1 if winner == DRAW else 2] += 1
    wins, draws, losses = (record['X'][k] + record['O'][k] for k in range(3))
    print(f"vs MCT({mct_iterations}): {wins} W / {draws} D / {losses} L"
          f"  (as X {'-'.join(map(str, record['X']))}, as O {'-'.join(map(str, record['O']))})")
    return record


def policy_entropy(training_data):
    # The lowest policy loss possible on this data: cross-entropy can only reach
    # the entropy of the visit-count targets themselves
    pi = torch.stack([move.move_policy for move in training_data])
    return -(pi * torch.log(pi.clamp_min(1e-12))).sum(dim=1).mean().item()


def log_generation(row):
    LOG_PATH.parent.mkdir(exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def random_symmetry(planes, pi):
    # Rotating or mirroring the whole 9x9 grid gives an equally valid position,
    # since every subgame and every cell within it move by the same transform.
    # The policy is transformed exactly like the board so each probability stays
    # on the cell it belongs to; the value target doesn't change
    k = random.randrange(4)
    planes = torch.rot90(planes, k, dims=(1, 2))
    pi = torch.rot90(pi.view(9, 9), k, dims=(0, 1))
    if random.random() < 0.5:
        planes = torch.flip(planes, dims=(2,))
        pi = torch.flip(pi, dims=(1,))
    return planes, pi.reshape(81)


def train(bot: torch.nn.Module, optimizer, training_data):
    bot.train()
    value_criterion = MSELoss()
    policy_criterion = CrossEntropyLoss()

    # Each position gets a random one of its 8 symmetries, so over many batches
    # the network sees every orientation of the positions it trains on
    augmented = [random_symmetry(move.encoded_game, move.move_policy) for move in training_data]
    inputs = torch.stack([planes for planes, _ in augmented])
    move_policy_target = torch.stack([pi for _, pi in augmented])
    value_target = torch.stack([torch.tensor(move.target, dtype=torch.float32) for move in training_data]).unsqueeze(1)

    logits, values = bot(inputs)
    policy_loss = policy_criterion(logits, move_policy_target)
    values_loss = value_criterion(values, value_target)
    loss = policy_loss + values_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), values_loss.item()

def training(optimizer, generations, games_per_gen, n_games_simulate, batch_size, passes_per_generation,
             buffer_generations=10, eval_games=10, mct_iterations=200, workers=None):
    bot = BOT
    # Training samples from the last few generations' self-play, not just the
    # newest, so the net doesn't forget positions it played a while ago
    replay_buffer = deque(maxlen=buffer_generations)
    run = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Starting Training")
    with GamePool(workers) as pool:
        for i in range(generations):
            start = time.time()
            print(f"Generation {i+1}/{generations}")
            games = generate_batch(pool, bot, games_per_gen, n_games_simulate)
            replay_buffer.append([move for game_moves in games for move in game_moves])
            selfplay_done = time.time()
            print(f"Self-play: {len(replay_buffer[-1])} positions from {games_per_gen} games in {selfplay_done - start:.0f}s")
            training_data = [move for generation in replay_buffer for move in generation]
            print(f"Training on {len(training_data)} positions from the last {len(replay_buffer)} generations")
            # Enough batches that each position is sampled passes_per_generation
            # times on average, however full the replay buffer is
            batch_samples = math.ceil(passes_per_generation * len(training_data) / batch_size)
            policy_total = value_total = 0.0
            for _ in range(batch_samples):
                batch = random.choices(training_data, k=batch_size)
                policy_loss, value_loss = train(bot, optimizer, batch)
                policy_total += policy_loss
                value_total += value_loss
            print(f"Avg loss over {batch_samples} batches: policy {policy_total / batch_samples:.4f}, value {value_total / batch_samples:.4f}")
            train_done = time.time()
            record = evaluate_vs_mct(pool, bot, eval_games, n_games_simulate, mct_iterations)
            eval_done = time.time()
            print(f"Generation took {eval_done - start:.0f}s")
            torch.save({
                'model_state_dict': bot.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, TRAINED_PATH)

            selfplay_results = Counter(game_winner(game_moves) for game_moves in games)
            log_generation({
                'run': run, 'generation': i + 1,
                'workers': pool.n_workers, 'n_games_simulate': n_games_simulate, 'mct_iterations': mct_iterations,
                'selfplay_games': games_per_gen, 'selfplay_positions': len(replay_buffer[-1]),
                'selfplay_x_wins': selfplay_results['X'], 'selfplay_o_wins': selfplay_results['O'],
                'selfplay_draws': selfplay_results[DRAW],
                'buffer_positions': len(training_data), 'batches': batch_samples,
                'policy_loss': policy_total / batch_samples, 'value_loss': value_total / batch_samples,
                'policy_target_entropy': policy_entropy(training_data),
                'value_target_mse': sum(move.target ** 2 for move in training_data) / len(training_data),
                **{f'eval_{side.lower()}_{outcome}': record[side][k]
                   for side in ('X', 'O') for k, outcome in enumerate(('wins', 'draws', 'losses'))},
                'selfplay_seconds': selfplay_done - start, 'train_seconds': train_done - selfplay_done,
                'eval_seconds': eval_done - train_done,
            })
