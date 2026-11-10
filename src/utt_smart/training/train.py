from utt_smart.net import Net, TRAINED_PATH, load_checkpoint, new_optimizer, save_checkpoint
from utt_smart.training.nmct import NeuralMC, encode
from utt_smart.training.utt import UTT, DRAW
from utt_smart.training.mct import MC, other_token
from utt_smart.training.symmetry import random_symmetries
from utt_smart.training.train_multi import GamePool
import torch
import torch.nn.functional as F
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
# Copies of the network every few generations, to measure progress against later
SNAPSHOT_DIR = TRAINED_PATH.parent / "checkpoints"


class GameRecord:
    # One self-play game, as tensors so it is cheap to send back from a worker
    # and to stack into training batches
    def __init__(self, states, policies, has_policy, values, winner):
        self.states = states          # (n, IN_PLANES, 9, 9) uint8: every plane is 0/1
        self.policies = policies      # (n, 81) visit-count targets, zeros where has_policy is False
        self.has_policy = has_policy  # (n,) bool: the move got a full search
        self.values = values          # (n,) game result for the player to move
        self.winner = winner


def play_game_nmct(nn_call, game_index, full_sims, fast_sims, full_search_prob):
    # Playout cap randomization (KataGo): only some moves get a full search with
    # root noise, and only those become policy targets. The rest get a quick
    # search that is just good enough to pick a decent move. Every position
    # still gets the game result as a value target, so a game costs far fewer
    # network calls while the policy targets stay as good as a full search gives
    states, policies, has_policy, tokens = [], [], [], []
    game = UTT()
    to_move = 'X'
    search = NeuralMC(nn_call, game, to_move)
    while not game.is_over():
        full = random.random() < full_search_prob
        search.noise = full
        search.start(full_sims if full else fast_sims)
        states.append(encode(game, to_move))
        policies.append(search.policy() if full else torch.zeros(81))
        has_policy.append(full)
        tokens.append(to_move)
        move = search.sample_move() if len(game.moves) < SAMPLED_MOVES else search.best_move()
        game.apply_move(move, to_move)
        search.advance(move)
        to_move = other_token(to_move)

    return GameRecord(torch.stack(states).to(torch.uint8), torch.stack(policies), torch.tensor(has_policy),
                      torch.tensor([0.0 if game.winner == DRAW else 1.0 if game.winner == token else -1.0
                                    for token in tokens]),
                      game.winner)


def nn_token_for(game_index):
    # The network alternates between X and O so the first-move advantage evens out
    return 'X' if game_index % 2 == 0 else 'O'


def play_vs_mct(nn_call, game_index, n_games_simulate, mct_iterations):
    # The network plays its best move with no noise, against plain MCT with random
    # playouts. Neither side keeps its tree between moves, so the result measures
    # the network rather than the search tricks
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
    print(f"vs MCT({mct_iterations}), net {n_games_simulate} sims: {wins} W / {draws} D / {losses} L"
          f"  (as X {'-'.join(map(str, record['X']))}, as O {'-'.join(map(str, record['O']))})")
    return record


def log_generation(row):
    LOG_PATH.parent.mkdir(exist_ok=True)
    if LOG_PATH.exists():
        with open(LOG_PATH, newline='') as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            # Written by an older version with other columns: keep it, but start a new file
            old = LOG_PATH.with_name(f"training_log.{time.strftime('%Y%m%d-%H%M%S')}.csv")
            LOG_PATH.rename(old)
            print(f"Log columns changed: moved the old log to {old}")
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


class ReplayBuffer:
    # The last few generations' self-play, kept on the training device so batches
    # are sampled without copying from Python objects
    def __init__(self, generations, device):
        self.device = device
        self.generations = deque(maxlen=generations)
        self._cat = None

    def append(self, data):
        # data is one generation, as returned by stack()
        self.generations.append(data)
        self._cat = None

    def stack(self, games):
        return (torch.cat([g.states for g in games]).to(self.device),
                torch.cat([g.policies for g in games]).to(self.device),
                torch.cat([g.has_policy for g in games]).to(self.device),
                torch.cat([g.values for g in games]).to(self.device))

    def all(self):
        if self._cat is None:
            self._cat = tuple(torch.cat(parts) for parts in zip(*self.generations))
        return self._cat

    def __len__(self):
        return len(self.all()[0])


def losses(bot, states, pi, has_policy, z):
    logits, values = bot(states.float())
    # Soft-target cross-entropy, only on moves that got a full search
    per_position = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1)
    policy_loss = per_position[has_policy].mean() if has_policy.any() else logits.sum() * 0
    value_loss = F.mse_loss(values.squeeze(1), z)
    return policy_loss, value_loss


def train_step(bot, optimizer, batch):
    states, pi, has_policy, z = batch
    # Each position gets a random one of its 8 symmetries, so over many batches
    # the network sees every orientation of the positions it trains on
    states, pi = random_symmetries(states, pi)
    policy_loss, value_loss = losses(bot, states, pi, has_policy, z)
    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


@torch.no_grad()
def measure(bot, data, batch_size=4096):
    # Average losses without training, in eval mode as self-play uses the network
    bot.eval()
    states, pi, has_policy, z = data
    policy_sum = value_sum = 0.0
    for i in range(0, len(states), batch_size):
        part = slice(i, i + batch_size)
        logits, values = bot(states[part].float())
        per_position = -(pi[part] * F.log_softmax(logits, dim=1)).sum(dim=1)
        policy_sum += per_position[has_policy[part]].sum().item()
        value_sum += F.mse_loss(values.squeeze(1), z[part], reduction='sum').item()
    return policy_sum / max(1, has_policy.sum().item()), value_sum / len(states)


def policy_entropy(pi, has_policy):
    # The lowest policy loss possible on this data: cross-entropy can only reach
    # the entropy of the visit-count targets themselves
    pi = pi[has_policy]
    return -(pi * torch.log(pi.clamp_min(1e-12))).sum(dim=1).mean().item()


def training(generations, games_per_gen, full_sims, fast_sims, full_search_prob, batch_size,
             samples_per_position, buffer_generations, eval_games=100, mct_iterations=1000,
             workers=None, lr=1e-3, snapshot_every=10, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    bot = Net().to(device)
    optimizer = new_optimizer(bot, lr=lr)
    checkpoint = load_checkpoint()
    start_generation = 0
    if checkpoint is not None:
        bot = Net(**checkpoint['config']).to(device)
        bot.load_state_dict(checkpoint['model_state_dict'])
        optimizer = new_optimizer(bot, lr=lr)
        if checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        start_generation = checkpoint.get('generation', 0)
        print(f"Resuming from {TRAINED_PATH} at generation {start_generation}")

    # Training samples from the last few generations' self-play, not just the
    # newest, so the net doesn't forget positions it played a while ago
    replay_buffer = ReplayBuffer(buffer_generations, device)
    run = time.strftime("%Y-%m-%d %H:%M:%S")
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    print(f"Starting training on {device}")
    with GamePool(workers) as pool:
        for i in range(generations):
            generation = start_generation + i + 1
            start = time.time()
            print(f"Generation {generation} ({i + 1}/{generations} this run)")
            games = pool.run(bot, games_per_gen, partial(play_game_nmct, full_sims=full_sims, fast_sims=fast_sims,
                                                          full_search_prob=full_search_prob))
            new_data = replay_buffer.stack(games)
            new_positions = len(new_data[0])
            full_positions = int(new_data[2].sum().item())
            selfplay_done = time.time()
            print(f"Self-play: {new_positions} positions ({full_positions} with full searches) "
                  f"from {games_per_gen} games in {selfplay_done - start:.0f}s")

            # Loss on positions the network hasn't trained on yet: unlike the
            # training loss, this goes up if the network is memorising the buffer
            fresh_policy, fresh_value = measure(bot, new_data)
            replay_buffer.append(new_data)
            data = replay_buffer.all()
            buffer_positions = len(data[0])

            # Every position is trained on about samples_per_position times while
            # it is in the buffer, however many generations that is
            batches = math.ceil(samples_per_position * new_positions / batch_size)
            bot.train()
            policy_total = value_total = 0.0
            for _ in range(batches):
                index = torch.randint(buffer_positions, (batch_size,), device=device)
                policy_loss, value_loss = train_step(bot, optimizer, tuple(part[index] for part in data))
                policy_total += policy_loss
                value_total += value_loss
            print(f"Loss on new positions before training: policy {fresh_policy:.4f}, value {fresh_value:.4f}")
            print(f"Training loss over {batches} batches from {buffer_positions} positions: "
                  f"policy {policy_total / batches:.4f}, value {value_total / batches:.4f}")
            train_done = time.time()

            record = evaluate_vs_mct(pool, bot, eval_games, full_sims, mct_iterations)
            eval_done = time.time()
            print(f"Generation took {eval_done - start:.0f}s")
            save_checkpoint(bot, optimizer, generation=generation)
            if generation % snapshot_every == 0:
                save_checkpoint(bot, None, SNAPSHOT_DIR / f"gen{generation:04d}.pt", generation=generation)

            selfplay_results = Counter(game.winner for game in games)
            log_generation({
                'run': run, 'generation': generation,
                'workers': pool.n_workers, 'n_games_simulate': full_sims, 'fast_sims': fast_sims,
                'full_search_prob': full_search_prob, 'mct_iterations': mct_iterations,
                'selfplay_games': games_per_gen, 'selfplay_positions': new_positions,
                'selfplay_full_positions': full_positions,
                'selfplay_x_wins': selfplay_results['X'], 'selfplay_o_wins': selfplay_results['O'],
                'selfplay_draws': selfplay_results[DRAW],
                'buffer_positions': buffer_positions, 'batches': batches,
                'policy_loss': policy_total / batches, 'value_loss': value_total / batches,
                'fresh_policy_loss': fresh_policy, 'fresh_value_loss': fresh_value,
                'policy_target_entropy': policy_entropy(data[1], data[2]),
                'value_target_mse': (data[3] ** 2).mean().item(),
                **{f'eval_{side.lower()}_{outcome}': record[side][k]
                   for side in ('X', 'O') for k, outcome in enumerate(('wins', 'draws', 'losses'))},
                'selfplay_seconds': selfplay_done - start, 'train_seconds': train_done - selfplay_done,
                'eval_seconds': eval_done - train_done,
            })
