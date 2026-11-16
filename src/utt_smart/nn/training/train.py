from utt_smart.bots.mct import MC
from utt_smart.bots.nmct import NeuralMC
from utt_smart.nn.encoding import encode
from utt_smart.nn.net import Net, TRAINED_PATH, load_checkpoint, load_net, new_optimizer, save_checkpoint
from utt_smart.nn.training.probes import evaluate_probes, probe_positions
from utt_smart.nn.training.symmetry import random_symmetries
from utt_smart.nn.training.train_multi import GamePool
from utt_smart.utt import UTT, DRAW, other_token
import torch
import torch.nn.functional as F
from collections import Counter, deque
import csv
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import pathlib
import random
import time

# Self-play samples moves from the visit counts for this many moves, then plays
# the most visited move for the rest of the game
SAMPLED_MOVES = 12

# Ladder matches start from this many random moves, so the games differ. Games
# come in pairs with the same opening and the colours swapped, so a lopsided
# opening favours neither network
LADDER_RANDOM_MOVES = 2

# One row per generation, appended across runs, for training_analysis.ipynb
LOG_PATH = TRAINED_PATH.parent / "runs" / "training_log.csv"
# Copies of the network every few generations, to measure progress against later
SNAPSHOT_DIR = TRAINED_PATH.parent / "checkpoints"


@dataclass
class RunPaths:
    # Where one training run keeps its files
    checkpoint: pathlib.Path  # the latest network and optimizer, to resume from
    snapshots: pathlib.Path   # genNNNN.pt every snapshot_every generations
    log: pathlib.Path         # one CSV row per generation
    probes: pathlib.Path      # the network's answers on the probe positions, one JSON line per generation

    @classmethod
    def in_dir(cls, run_dir):
        run_dir = pathlib.Path(run_dir)
        return cls(run_dir / "trained.pt", run_dir / "checkpoints", run_dir / "training_log.csv",
                   run_dir / "probes.jsonl")

    def snapshot(self, generation):
        return self.snapshots / f"gen{generation:04d}.pt"


# The files used when no run_dir is given: trained.pt, checkpoints/ and runs/ in the project root
DEFAULT_PATHS = RunPaths(TRAINED_PATH, SNAPSHOT_DIR, LOG_PATH, LOG_PATH.parent / "probes.jsonl")


@dataclass
class TrainingConfig:
    # Self-play. A quarter of moves get a full 200-simulation search and become
    # policy targets; the rest get 40 simulations, so a move costs ~80 network
    # calls on average. 400 games give ~22k positions per generation
    games_per_gen: int = 400
    full_sims: int = 200
    fast_sims: int = 40
    full_search_prob: float = 0.25
    # Training. The 8-generation buffer holds ~175k positions, and each position
    # is trained on about 4 times while it is in the buffer
    batch_size: int = 512
    samples_per_position: float = 4
    buffer_generations: int = 8
    lr: float = 1e-3
    # Evaluation against plain MCT, every generation
    eval_games: int = 100
    mct_iterations: int = 1000
    # Every snapshot_every generations the network is saved to checkpoints/ and,
    # if ladder_games > 0, plays the previous snapshot to rate it (Elo)
    snapshot_every: int = 10
    ladder_games: int = 40
    ladder_sims: int = 100


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


def evaluate_vs_mct(pool, bot, n_games, n_games_simulate, mct_iterations, log=print):
    nn_tokens = [nn_token_for(i) for i in range(n_games)]
    winners = pool.run(bot, n_games, partial(play_vs_mct, n_games_simulate=n_games_simulate, mct_iterations=mct_iterations))
    record = {'X': [0, 0, 0], 'O': [0, 0, 0]}  # wins, draws, losses for the network playing that side
    for nn_token, winner in zip(nn_tokens, winners):
        record[nn_token][0 if winner == nn_token else 1 if winner == DRAW else 2] += 1
    wins, draws, losses = (record['X'][k] + record['O'][k] for k in range(3))
    log(f"vs MCT({mct_iterations}), net {n_games_simulate} sims: {wins} W / {draws} D / {losses} L"
        f"  (as X {'-'.join(map(str, record['X']))}, as O {'-'.join(map(str, record['O']))})")
    return record


def play_net_vs_net(nn_calls, game_index, sims):
    # nn_calls[0] is the network being rated, nn_calls[1] its opponent. Both
    # play their best move with no noise and no tree reuse, as in play_vs_mct.
    # That makes a game fixed by its opening, so the opening is random, shared
    # by each pair of games, and the colours swap within the pair
    new_token = nn_token_for(game_index)
    opening = random.Random(game_index // 2)
    game = UTT()
    to_move = 'X'
    while not game.is_over():
        if len(game.moves) < LADDER_RANDOM_MOVES:
            move = opening.choice(game.valid_moves())
        else:
            searcher = NeuralMC(nn_calls[0] if to_move == new_token else nn_calls[1], game, to_move)
            searcher.start(sims)
            move = searcher.best_move()
        game.apply_move(move, to_move)
        to_move = other_token(to_move)
    return game.winner


def elo_difference(score, n_games):
    # The rating gap that predicts this score (wins + draws / 2, as a share of
    # the games). A clean sweep would be an infinite gap, so the score is kept
    # half a game away from 0 and 1
    score = min(max(score, 0.5 / n_games), 1 - 0.5 / n_games)
    return 400 * math.log10(score / (1 - score))


def log_generation(row, path=LOG_PATH, log=print):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, newline='') as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            # Written by an older version with other columns: keep it, but start a new file
            old = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
            path.rename(old)
            log(f"Log columns changed: moved the old log to {old}")
    new_file = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_log(path):
    # The rows log_generation wrote, with numbers turned back into numbers and
    # empty cells into None
    path = pathlib.Path(path)
    if not path.exists():
        return []

    def value(text):
        if text == '':
            return None
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    with open(path, newline='') as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


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

    def resize(self, generations):
        # Keeps the newest generations that still fit
        if generations != self.generations.maxlen:
            self.generations = deque(self.generations, maxlen=generations)
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




class Trainer:
    # One training run, driven a generation at a time so a notebook can watch
    # it, change the config between generations, and stop and carry on. Keeps
    # the worker pool, the network, the optimizer and the replay buffer alive
    # between calls to run(). Resumes from paths.checkpoint if it exists.
    #
    # A generation only counts once it finishes: it is then saved, logged and
    # added to history. Interrupting one part way leaves the previous
    # generation as the last one, although training that already happened on
    # the interrupted generation's games stays in the network
    def __init__(self, config=None, run_dir=None, workers=None, device=None, log=print):
        self.config = config or TrainingConfig()
        self.paths = RunPaths.in_dir(run_dir) if run_dir is not None else DEFAULT_PATHS
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.log = log
        self.workers = workers
        self.phase = 'idle'
        self.paths.snapshots.mkdir(parents=True, exist_ok=True)
        self.paths.log.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = load_checkpoint(self.paths.checkpoint)
        if checkpoint is None:
            self.bot = Net().to(self.device)
            self.optimizer = new_optimizer(self.bot, lr=self.config.lr)
            self.generation = 0
        else:
            self.bot = Net(**checkpoint['config']).to(self.device)
            self.bot.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = new_optimizer(self.bot, lr=self.config.lr)
            if checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.generation = checkpoint.get('generation', 0)

        # Everything logged so far, so charts can show the whole run after a restart
        self.history = read_log(self.paths.log)
        self.probes = probe_positions()
        self.probe_history = []
        if self.paths.probes.exists():
            with open(self.paths.probes) as f:
                self.probe_history = [json.loads(line) for line in f if line.strip()]

        # Elo of each rated snapshot. Generation 0 is the untrained network,
        # the anchor of the ladder at 0
        self.ratings = {row['generation']: row['elo'] for row in self.history if row.get('elo') is not None}
        if self.generation == 0:
            save_checkpoint(self.bot, None, self.paths.snapshot(0), generation=0)
            self._record_probes(0)
        if self.paths.snapshot(0).exists():
            self.ratings.setdefault(0, 0.0)

        if checkpoint is None:
            self.log(f"New network in {self.paths.checkpoint.parent}, training on {self.device}")
        else:
            self.log(f"Resuming from {self.paths.checkpoint} at generation {self.generation}, "
                     f"training on {self.device}. The replay buffer starts empty")

        # Training samples from the last few generations' self-play, not just the
        # newest, so the net doesn't forget positions it played a while ago
        self.replay_buffer = ReplayBuffer(self.config.buffer_generations, self.device)
        self.last_games = []  # the newest generation's self-play, as GameRecords
        self.run_id = time.strftime("%Y-%m-%d %H:%M:%S")
        self.pool = GamePool(workers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.pool.close()

    def run(self, generations, on_generation=None):
        # Runs this many generations, calling on_generation(trainer) after each.
        # Ctrl-C (or the notebook's interrupt button) stops cleanly; call run()
        # again to carry on
        for _ in range(generations):
            try:
                self.run_generation()
                if on_generation is not None:
                    on_generation(self)
            except KeyboardInterrupt:
                phase = self.phase
                self.phase = 'idle'
                if phase in ('self-play', 'evaluation', 'ladder'):
                    # The workers may still be busy with the abandoned games
                    self.log("Interrupted: restarting the worker pool")
                    self.pool.close()
                    self.pool = GamePool(self.workers)
                self.log(f"Stopped during {phase} of generation {self.generation + 1}. "
                         f"Generation {self.generation} is the last one saved")
                return

    def _start(self, phase, message):
        self.phase = phase
        self.log(message)

    def _record_probes(self, generation):
        answers = evaluate_probes(self.bot, self.probes, self.device)
        entry = {'generation': generation, 'probes': answers}
        self.probe_history.append(entry)
        with open(self.paths.probes, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def run_generation(self):
        cfg = self.config
        generation = self.generation + 1
        self.replay_buffer.resize(cfg.buffer_generations)
        # The config's lr wins over the one saved with the optimizer, so it can be changed between runs
        for group in self.optimizer.param_groups:
            group['lr'] = cfg.lr
        start = time.time()

        self._start('self-play', f"Generation {generation}: self-play, {cfg.games_per_gen} games")
        games = self.pool.run(self.bot, cfg.games_per_gen,
                              partial(play_game_nmct, full_sims=cfg.full_sims, fast_sims=cfg.fast_sims,
                                      full_search_prob=cfg.full_search_prob))
        new_data = self.replay_buffer.stack(games)
        new_positions = len(new_data[0])
        full_positions = int(new_data[2].sum().item())
        selfplay_done = time.time()
        self.log(f"Self-play: {new_positions} positions ({full_positions} with full searches) "
                 f"from {cfg.games_per_gen} games in {selfplay_done - start:.0f}s")

        # Loss on positions the network hasn't trained on yet: unlike the
        # training loss, this goes up if the network is memorising the buffer
        self._start('training', f"Generation {generation}: training")
        fresh_policy, fresh_value = measure(self.bot, new_data)
        self.replay_buffer.append(new_data)
        data = self.replay_buffer.all()
        buffer_positions = len(data[0])

        # Every position is trained on about samples_per_position times while
        # it is in the buffer, however many generations that is
        batches = math.ceil(cfg.samples_per_position * new_positions / cfg.batch_size)
        self.bot.train()
        policy_total = value_total = 0.0
        for _ in range(batches):
            index = torch.randint(buffer_positions, (cfg.batch_size,), device=self.device)
            policy_loss, value_loss = train_step(self.bot, self.optimizer, tuple(part[index] for part in data))
            policy_total += policy_loss
            value_total += value_loss
        self.log(f"Loss on new positions before training: policy {fresh_policy:.4f}, value {fresh_value:.4f}")
        self.log(f"Training loss over {batches} batches from {buffer_positions} positions: "
                 f"policy {policy_total / batches:.4f}, value {value_total / batches:.4f}")
        train_done = time.time()

        self._start('evaluation', f"Generation {generation}: {cfg.eval_games} games against MCT({cfg.mct_iterations})")
        record = evaluate_vs_mct(self.pool, self.bot, cfg.eval_games, cfg.full_sims, cfg.mct_iterations, log=self.log)
        eval_done = time.time()

        ladder = {'ladder_opponent': None, 'ladder_wins': None, 'ladder_draws': None, 'ladder_losses': None,
                  'elo': None}
        snapshot = generation % cfg.snapshot_every == 0
        if snapshot and cfg.ladder_games > 0:
            ladder.update(self._ladder(generation))
        ladder_done = time.time()

        self.phase = 'saving'
        save_checkpoint(self.bot, self.optimizer, self.paths.checkpoint, generation=generation)
        if snapshot:
            save_checkpoint(self.bot, None, self.paths.snapshot(generation), generation=generation)
        if ladder['elo'] is not None:
            self.ratings[generation] = ladder['elo']
        self._record_probes(generation)

        selfplay_results = Counter(game.winner for game in games)
        row = {
            'run': self.run_id, 'generation': generation, 'workers': self.pool.n_workers,
            **asdict(cfg),
            'selfplay_positions': new_positions, 'selfplay_full_positions': full_positions,
            'selfplay_x_wins': selfplay_results['X'], 'selfplay_o_wins': selfplay_results['O'],
            'selfplay_draws': selfplay_results[DRAW],
            'buffer_positions': buffer_positions, 'batches': batches,
            'policy_loss': policy_total / batches, 'value_loss': value_total / batches,
            'fresh_policy_loss': fresh_policy, 'fresh_value_loss': fresh_value,
            'policy_target_entropy': policy_entropy(data[1], data[2]),
            'value_target_mse': (data[3] ** 2).mean().item(),
            **{f'eval_{side.lower()}_{outcome}': record[side][k]
               for side in ('X', 'O') for k, outcome in enumerate(('wins', 'draws', 'losses'))},
            **ladder,
            'selfplay_seconds': selfplay_done - start, 'train_seconds': train_done - selfplay_done,
            'eval_seconds': eval_done - train_done, 'ladder_seconds': ladder_done - eval_done,
        }
        log_generation(row, self.paths.log, log=self.log)
        self.history.append(row)
        self.last_games = games
        self.generation = generation
        self.phase = 'idle'
        self.log(f"Generation {generation} took {time.time() - start:.0f}s")
        return row

    def _ladder(self, generation):
        # Plays the newest rated snapshot, and rates this network from the
        # result. The ratings chain from snapshot to snapshot back to the
        # untrained network at 0, so errors add up along the chain: read the
        # trend, not single steps
        cfg = self.config
        rated = [g for g in self.ratings if g < generation and self.paths.snapshot(g).exists()]
        if not rated:
            # Resumed from a run that never rated a snapshot: this one becomes the anchor
            return {'elo': 0.0}
        opponent_generation = max(rated)
        self._start('ladder', f"Generation {generation}: {cfg.ladder_games} games against "
                              f"generation {opponent_generation}")
        opponent = load_net(self.paths.snapshot(opponent_generation))
        winners = self.pool.run([self.bot, opponent], cfg.ladder_games, partial(play_net_vs_net, sims=cfg.ladder_sims))
        tokens = [nn_token_for(i) for i in range(cfg.ladder_games)]
        wins = sum(w == t for w, t in zip(winners, tokens))
        draws = sum(w == DRAW for w in winners)
        lost = cfg.ladder_games - wins - draws
        elo = self.ratings[opponent_generation] + elo_difference((wins + draws / 2) / cfg.ladder_games,
                                                                 cfg.ladder_games)
        self.log(f"vs generation {opponent_generation}, both {cfg.ladder_sims} sims: "
                 f"{wins} W / {draws} D / {lost} L, Elo {elo:.0f}")
        return {'ladder_opponent': opponent_generation, 'ladder_wins': wins, 'ladder_draws': draws,
                'ladder_losses': lost, 'elo': elo}
