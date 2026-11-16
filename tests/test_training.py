import random

import torch

from utt_smart.bots.nmct import NeuralMC
from utt_smart.nn.encoding import encode
from utt_smart.nn.net import IN_PLANES, Net
from utt_smart.nn.training.symmetry import PERMUTATIONS, random_symmetries
from utt_smart.nn.training.train import losses, play_game_nmct
from utt_smart.utt import UTT, other_token


def random_position(rng, moves):
    game, token = UTT(), 'X'
    for _ in range(moves):
        if game.is_over():
            break
        game.apply_move(rng.choice(game.valid_moves()), token)
        token = other_token(token)
    return game, token


def test_symmetries_map_positions_to_positions():
    # Replaying a game with every move transformed must give exactly the
    # transformed planes: same pieces, legal moves and subgame owners
    rng = random.Random(0)
    for _ in range(30):
        game, to_move = random_position(rng, rng.randrange(1, 60))
        planes = encode(game, to_move)
        for perm in PERMUTATIONS:
            # perm gathers, so the cell at old position perm[i] lands at i
            new_cell = {int(old): new for new, old in enumerate(perm)}
            transformed, token = UTT(), 'X'
            for move in game.moves:
                transformed.apply_move(new_cell[move], token)
                token = other_token(token)
            expected = planes.reshape(IN_PLANES, 81)[:, perm].view(IN_PLANES, 9, 9)
            assert torch.equal(encode(transformed, to_move), expected)


def test_random_symmetries_keep_policy_on_its_cell():
    planes = torch.zeros(64, IN_PLANES, 9, 9)
    pi = torch.zeros(64, 81)
    cells = torch.randint(81, (64,))
    planes.view(64, IN_PLANES, 81)[torch.arange(64), 2, cells] = 1
    pi[torch.arange(64), cells] = 1
    planes, pi = random_symmetries(planes, pi)
    assert torch.equal(planes[:, 2].reshape(64, 81), pi)


def test_search_takes_a_winning_move():
    # X has won subgames 0 and 1 (top row of each) and has cells 6 and 7 of
    # subgame 2. O's last move, cell 32, is position 2 of its subgame, so X must
    # play in subgame 2, where cell 8 wins it and with it the top row of the big
    # board. With a network that knows nothing, only the search can find it
    game = UTT()
    for move in (0, 1, 2, 3, 4, 5, 6, 7):
        game.board[move] = 'X'
    game.subgames[0] = game.subgames[1] = 'X'
    game.board[32] = 'O'
    game.moves = [32]
    assert 8 in game.valid_moves() and all(m in (8, 15, 16, 17, 24, 25, 26) for m in game.valid_moves())
    def uniform(encoded):
        return [1 / 81] * 81, 0.0

    search = NeuralMC(uniform, game, 'X')
    search.start(300)
    assert search.best_move() == 8


def test_tree_reuse_keeps_the_subtree():
    rng = random.Random(1)

    def uniform(encoded):
        return [1 / 81] * 81, rng.uniform(-1, 1)

    search = NeuralMC(uniform, UTT(), 'X')
    search.start(200)
    move = search.best_move()
    visits = next(c.visits for c in search.root.children if c.move == move)
    search.advance(move)
    assert search.root.visits == visits and search.root.parent is None
    assert search.game.moves == [move] and search.root.token == 'X'


def test_self_play_game_and_losses():
    bot = Net(blocks=1, ch=8).eval()

    def nn_call(encoded):
        with torch.no_grad():
            logits, value = bot(encoded.unsqueeze(0))
        return torch.softmax(logits, 1)[0].tolist(), value.item()

    record = play_game_nmct(nn_call, 0, full_sims=8, fast_sims=2, full_search_prob=0.5)
    n = len(record.values)
    assert record.states.shape == (n, IN_PLANES, 9, 9) and record.states.dtype == torch.uint8
    full = record.policies[record.has_policy]
    assert torch.allclose(full.sum(1), torch.ones(len(full)))
    # Every position gets the final result from its own side: win, draw or loss
    assert set(record.values.tolist()) <= {-1.0, 0.0, 1.0}
    policy_loss, value_loss = losses(bot.train(), record.states, record.policies, record.has_policy, record.values)
    assert torch.isfinite(policy_loss) and torch.isfinite(value_loss)


def test_threads_route_each_network_its_own_calls():
    # Two networks in one batched evaluator: every call must come back from the
    # network it was sent to, even when games wait on different networks
    from utt_smart.nn.training.train_multi import run_on_threads
    nets = [Net(blocks=1, ch=8).eval(), Net(blocks=1, ch=8).eval()]
    position = encode(UTT(), 'X')
    with torch.no_grad():
        expected = [net(position.unsqueeze(0))[1].item() for net in nets]

    def play(nn_calls, game_index):
        # Games ask the networks in different orders
        order = (0, 1, 1, 0) if game_index % 2 else (1, 0, 0, 1)
        return [(model, nn_calls[model](position)[1]) for model in order]

    for calls in run_on_threads(nets, 6, play):
        for model, value in calls:
            assert abs(value - expected[model]) < 1e-5


def test_net_vs_net_pairs_share_openings():
    # Games 2k and 2k + 1 start from the same random opening; the first position
    # either network is asked about is the one right after it
    from utt_smart.nn.training.train import play_net_vs_net
    first_seen = []

    def uniform(encoded):
        if len(first_seen) == game:
            first_seen.append(encoded[:2].sum(0))  # the occupied cells
        return [1 / 81] * 81, 0.0

    for game in range(4):
        assert play_net_vs_net([uniform, uniform], game, sims=4) in ('X', 'O', 'd')
    assert torch.equal(first_seen[0], first_seen[1]) and torch.equal(first_seen[2], first_seen[3])
    assert not torch.equal(first_seen[0], first_seen[2])


def test_elo_difference():
    import math

    from utt_smart.nn.training.train import elo_difference
    assert elo_difference(0.5, 40) == 0
    assert 185 < elo_difference(0.75, 40) < 195
    # A clean sweep is kept finite
    assert math.isclose(elo_difference(1.0, 40), -elo_difference(0.0, 40)) and elo_difference(1.0, 40) < 1000
