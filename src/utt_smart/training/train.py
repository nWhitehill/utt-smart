from utt_smart.net import BOT, TRAINED_PATH
from utt_smart.training.nmct import NeuralMC, encode
from utt_smart.training.utt import UTT, DRAW
from utt_smart.training.mct import MC, other_token
from utt_smart.training.train_multi import run_on_threads
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
import torch
from collections import deque
import random
import time

# Self-play samples moves from the visit counts for this many moves, then plays
# the most visited move for the rest of the game
SAMPLED_MOVES = 12

class TrainingMove:
    def __init__(self, encoded_game, move_policy, turn_token):
        self.encoded_game = encoded_game
        self.move_policy = move_policy
        self.turn_token = turn_token
        self.target = None

    def add_target(self, results):
        self.target = results[self.turn_token]

def play_game_nmct(nn_call, n_games_simulate):
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


def generate_batch(bot, n_games_to_play, n_games_simulate):
    games = run_on_threads(bot, n_games_to_play, lambda nn_call, _: play_game_nmct(nn_call, n_games_simulate))
    return [move for game_moves in games for move in game_moves]


def play_vs_mct(nn_call, nn_token, n_games_simulate, mct_iterations):
    # The network plays its best move with no noise, against plain MCT with random playouts
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


def evaluate_vs_mct(bot, n_games, n_games_simulate, mct_iterations):
    # The network alternates between X and O so the first-move advantage evens out
    nn_tokens = ['X' if i % 2 == 0 else 'O' for i in range(n_games)]
    winners = run_on_threads(bot, n_games, lambda nn_call, i: play_vs_mct(nn_call, nn_tokens[i], n_games_simulate, mct_iterations))
    record = {'X': [0, 0, 0], 'O': [0, 0, 0]}  # wins, draws, losses for the network playing that side
    for nn_token, winner in zip(nn_tokens, winners):
        record[nn_token][0 if winner == nn_token else 1 if winner == DRAW else 2] += 1
    wins, draws, losses = (record['X'][k] + record['O'][k] for k in range(3))
    print(f"vs MCT({mct_iterations}): {wins} W / {draws} D / {losses} L"
          f"  (as X {'-'.join(map(str, record['X']))}, as O {'-'.join(map(str, record['O']))})")


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

def training(optimizer, generations, games_per_gen, n_games_simulate, batch_samples, batch_size,
             buffer_generations=10, eval_games=10, mct_iterations=200):
    bot = BOT
    # Training samples from the last few generations' self-play, not just the
    # newest, so the net doesn't forget positions it played a while ago
    replay_buffer = deque(maxlen=buffer_generations)
    print("Starting Training")
    for i in range(generations):
        start = time.time()
        print(f"Generation {i+1}/{generations}")
        replay_buffer.append(generate_batch(bot, games_per_gen, n_games_simulate))
        print(f"Self-play: {len(replay_buffer[-1])} positions from {games_per_gen} games in {time.time() - start:.0f}s")
        training_data = [move for generation in replay_buffer for move in generation]
        print(f"Training on {len(training_data)} positions from the last {len(replay_buffer)} generations")
        policy_total = value_total = 0.0
        for _ in range(batch_samples):
            batch = random.choices(training_data, k=batch_size)
            policy_loss, value_loss = train(bot, optimizer, batch)
            policy_total += policy_loss
            value_total += value_loss
        print(f"Avg loss over {batch_samples} batches: policy {policy_total / batch_samples:.4f}, value {value_total / batch_samples:.4f}")
        evaluate_vs_mct(bot, eval_games, n_games_simulate, mct_iterations)
        print(f"Generation took {time.time() - start:.0f}s")
        torch.save({
            'model_state_dict': bot.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, TRAINED_PATH)
