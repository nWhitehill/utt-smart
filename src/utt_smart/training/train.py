from utt_smart.net import BOT, TRAINED_PATH
from utt_smart.training.nmct import NeuralMC, encode
from utt_smart.training.utt import UTT, DRAW
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
import torch
import random

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
       
def generate_batch(bot, n_games_to_play, n_games_simulate):
    batch_data = []
    for _ in range(n_games_to_play):
        game_moves = []
        game = UTT()
        turn_to_move = 'X'
        while not game.is_over():
            nmc = NeuralMC(bot, game, turn_to_move, noise=True)
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
        batch_data.extend(game_moves)
    return batch_data 


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
    print(f"Values Loss: {values_loss.item()}")
    print(f"Policy Loss: {policy_loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def training(optimizer, generations, games_per_gen, n_games_simulate, batch_samples, batch_size):
    bot = BOT
    print("Starting Training")
    for i in range(generations):
        print(f"Training Generation {i+1}/{generations}")
        batch_data = generate_batch(bot, games_per_gen, n_games_simulate)
        print("Training Data Generated")
        for j in range(batch_samples):
            print(f"Updating with batch {j+1}/{batch_samples}")
            batch = random.choices(batch_data, k=batch_size)
            train(bot, optimizer, batch)
        torch.save({
            'model_state_dict': bot.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, TRAINED_PATH)

