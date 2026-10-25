from utt_smart.net import Net
from utt_smart.training.nmct import NeuralMC, encode
from utt_smart.training.utt import UTT, DRAW
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
import torch
import random


class TrainingMove:
    def __init__(self, encoded_game, move_policy, turn_token):
        self.encoded_game = encoded_game
        self.move_policy = move_policy
        self.turn_token = turn_token
        self.target = None

    def add_target(self, results):
        self.target = results[self.turn_token]
       
def generate_batch(bot, n_games_to_play, n_games_simulate, batch_size):
    batch_data = []
    for _ in range(n_games_to_play):
        game_moves = []
        game = UTT()
        turn_to_move = 'X'
        while not game.is_over():
            nmc = NeuralMC(bot, game, turn_to_move)
            nmc.start(n_games_simulate)
            move = nmc.best_move()
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
        batch_data.append(move)
    return random.choices(batch_data, k=batch_size)


def train(bot: torch.nn.Module, optimizer, training_data):
    bot.train()
    value_criterion = MSELoss()
    policy_criterion = CrossEntropyLoss()

    inputs = torch.stack([move.encoded_game for move in training_data])
    move_policy_target = torch.stack([move.move_policy for move in training_data])
    value_target = torch.stack([torch.tensor(move.target, dtype=torch.float32) for move in training_data]).unsqueeze(1)

    logits, values = bot(inputs)
    policy_loss = policy_criterion(logits, move_policy_target)
    values_loss = value_criterion(values, value_target)
    loss = policy_loss + values_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def training(generations, games_per_gen, n_games_simulate):
    nn = Net()
    for _ in range(generations):
        bot = Net()
        generate_batch(bot, n_games_simulate)
        training_data = random.choices(batch_data, k=100)
        batch_input = torch.stack([move.encoded_game for move in training_data])
        batch_policy = torch.stack([move.policy for move in training_data])
        batch_values = torch.stack([torch.tensor(move.value) for move in training_data])
        bot.train()


bot = Net()
optimizer = Adam(bot.parameters(), lr=1e-3)

batch = generate_batch(bot, 2, 20, 20)
train(bot, optimizer, batch)


