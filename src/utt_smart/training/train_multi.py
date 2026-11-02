import threading
from concurrent.futures import Future, ThreadPoolExecutor
import torch


class BatchedEvaluator:
    # Game threads each ask for one position at a time and block until it's
    # evaluated. A single eval thread waits until every live game has a request
    # pending, then runs them all through the network in one call
    def __init__(self, bot, n_games):
        self.bot = bot
        self.pending = []  # (encoded position, Future for its result)
        self.live_games = n_games
        self.cond = threading.Condition()

    def evaluate(self, encoded):
        # Called from game threads; returns (move probabilities, value)
        future = Future()
        with self.cond:
            self.pending.append((encoded, future))
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
        self.bot.eval()
        with torch.inference_mode():
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: len(self.pending) >= self.live_games)
                    if self.live_games == 0:
                        return
                    batch, self.pending = self.pending, []
                try:
                    logits, values = self.bot(torch.stack([encoded for encoded, _ in batch]))
                    # Plain lists, so game threads don't make small torch calls on them
                    probs = torch.softmax(logits, dim=1).tolist()
                    values = values.squeeze(1).tolist()
                    for (_, future), p, v in zip(batch, probs, values):
                        future.set_result((p, v))
                except Exception as e:
                    # Pass the error to the waiting games rather than leaving them hung
                    for _, future in batch:
                        future.set_exception(e)


def run_on_threads(bot, n_games, play):
    # Runs play(nn_call, game_index) for every game on its own thread, with
    # their network calls batched together. Returns each game's result in order
    evaluator = BatchedEvaluator(bot, n_games)
    eval_thread = threading.Thread(target=evaluator.run)
    eval_thread.start()

    def run(game_index):
        try:
            return play(evaluator.evaluate, game_index)
        finally:
            evaluator.game_finished()

    with ThreadPoolExecutor(max_workers=n_games) as executor:
        results = [game.result() for game in [executor.submit(run, i) for i in range(n_games)]]
    eval_thread.join()
    return results
