import multiprocessing
import os
import pickle
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from utt_smart.nn.net import Net


class BatchedEvaluator:
    # Game threads each ask for one position at a time and block until it's
    # evaluated. A single eval thread waits until every live game has a request
    # pending, then runs them all through the networks, one call per network.
    # There can be several networks (a match between two of them), but one
    # evaluator for all of them: a game waiting on one network still counts as
    # pending, so neither network waits for games that are busy with the other
    def __init__(self, bots, n_games):
        self.bots = bots
        self.pending = []  # (network index, encoded position, Future for its result)
        self.live_games = n_games
        self.cond = threading.Condition()

    def evaluate(self, encoded, model=0):
        # Called from game threads; returns (move probabilities, value)
        future = Future()
        with self.cond:
            self.pending.append((model, encoded, future))
            self.cond.notify()
        # Wait outside the lock so other games can add their requests
        return future.result()

    def game_finished(self):
        # A finished game will never send another request, so stop waiting for it
        with self.cond:
            self.live_games -= 1
            self.cond.notify()

    def run(self):
        # The eval thread. Grad mode is per thread, so inference_mode has to be set here
        for bot in self.bots:
            bot.eval()
        with torch.inference_mode():
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: len(self.pending) >= self.live_games)
                    if self.live_games == 0:
                        return
                    batch, self.pending = self.pending, []
                for model, bot in enumerate(self.bots):
                    requests = [(encoded, future) for m, encoded, future in batch if m == model]
                    if requests:
                        self._evaluate(bot, requests)

    def _evaluate(self, bot, requests):
        try:
            logits, values = bot(torch.stack([encoded for encoded, _ in requests]))
            # Plain lists, so game threads don't make small torch calls on them
            probs = torch.softmax(logits, dim=1).tolist()
            values = values.squeeze(1).tolist()
            for (_, future), p, v in zip(requests, probs, values):
                future.set_result((p, v))
        except Exception as e:
            # Pass the error to the waiting games rather than leaving them hung
            for _, future in requests:
                future.set_exception(e)


def run_on_threads(bots, n_games, play):
    # Runs play(nn_call, game_index) for every game on its own thread, with
    # their network calls batched together. Returns each game's result in order.
    # bots is one network, or a list of them; with a list, play gets a list of
    # nn_calls, one per network
    several = isinstance(bots, (list, tuple))
    evaluator = BatchedEvaluator(list(bots) if several else [bots], n_games)
    nn_calls = [lambda encoded, model=model: evaluator.evaluate(encoded, model)
                for model in range(len(evaluator.bots))]
    eval_thread = threading.Thread(target=evaluator.run)
    eval_thread.start()

    def run(game_index):
        try:
            return play(nn_calls if several else nn_calls[0], game_index)
        finally:
            evaluator.game_finished()

    with ThreadPoolExecutor(max_workers=n_games) as executor:
        results = [game.result() for game in [executor.submit(run, i) for i in range(n_games)]]
    eval_thread.join()
    return results


def init_worker():
    # Ctrl-C (or a notebook's interrupt, which signals the whole process group)
    # is handled by the main process, which then shuts the pool down. A worker
    # that died of it mid-task would leave the pool waiting for a lost result
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # One core per worker, rather than every worker starting a torch thread
    # pool the size of the whole machine
    torch.set_num_threads(1)
    # Spawned processes all start torch's generator from the same fixed seed,
    # which would give every worker the same Dirichlet noise
    torch.seed()


# The worker's copies of the networks, rebuilt only when a network's shape changes
_worker_bots = []


def run_chunk(weights, play, game_indices):
    # Runs in a worker: the chunk's games share this process's copies of the
    # networks, with their calls batched on threads as in run_on_threads
    several, networks = pickle.loads(weights)
    for i, (config, state_dict) in enumerate(networks):
        if i == len(_worker_bots):
            _worker_bots.append(Net(**config))
        elif _worker_bots[i].config != config:
            _worker_bots[i] = Net(**config)
        _worker_bots[i].load_state_dict(state_dict)
    bots = _worker_bots[:len(networks)] if several else _worker_bots[0]
    results = run_on_threads(bots, len(game_indices), lambda nn_call, i: play(nn_call, game_indices[i]))
    # Plain pickle copies tensors into the message. The pool's own pickler would
    # put each tensor in its own shared memory file, and thousands of small
    # ones run out of file descriptors
    return pickle.dumps(results)


class GamePool:
    # Worker processes that play games in parallel, outside the GIL. Start it
    # once per training run, since each worker has to import torch
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or os.cpu_count()
        # spawn rather than fork: forking after torch has started its thread pools can deadlock
        self.pool = multiprocessing.get_context('spawn').Pool(
            self.n_workers, initializer=init_worker)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.pool.terminate()
        self.pool.join()

    def run(self, bots, n_games, play):
        # Like run_on_threads, but the games are split across the workers. play
        # is sent to them, so it must be picklable: a module-level function or
        # a functools.partial of one, not a lambda.
        # Workers run on the CPU, whatever device the network is training on
        several = isinstance(bots, (list, tuple))
        networks = [(bot.config, {k: v.cpu() for k, v in bot.state_dict().items()})
                    for bot in (bots if several else [bots])]
        weights = pickle.dumps((several, networks))
        chunks = [chunk for chunk in (list(range(w, n_games, self.n_workers)) for w in range(self.n_workers)) if chunk]
        chunk_results = self.pool.starmap(run_chunk, [(weights, play, chunk) for chunk in chunks])
        results = [None] * n_games
        for chunk, pickled in zip(chunks, chunk_results):
            for game_index, result in zip(chunk, pickle.loads(pickled)):
                results[game_index] = result
        return results
