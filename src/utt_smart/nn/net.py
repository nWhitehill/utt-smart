import pathlib
import torch
from torch import nn

# Always the project root, whatever directory training is run from
TRAINED_PATH = pathlib.Path(__file__).resolve().parents[3] / "trained.pt"

# Input planes, from the point of view of the player about to move (see encoding.encode):
# my pieces, opponent's pieces, legal moves, subgames I've won, subgames they've
# won, drawn subgames, and a constant plane that is 1 when the player to move is X
IN_PLANES = 7


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class Net(nn.Module):
    def __init__(self, in_planes=IN_PLANES, blocks=6, ch=64, moves=81):
        super().__init__()
        self.config = {'in_planes': in_planes, 'blocks': blocks, 'ch': ch}
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        )
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # Each cell is one move, so a 1x1 conv to a single channel gives one logit per move
        self.policy = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Flatten())
        # Two channels into the value head rather than one, so it isn't squeezed
        # through a single number per cell
        self.value = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(2 * moves, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        # x is (N, IN_PLANES, 9, 9); plane 2 marks the legal moves
        h = self.trunk(self.stem(x))
        # Illegal moves get a huge negative logit so softmax gives them ~0 probability.
        # -1e9 rather than -inf avoids 0 * -inf = NaN in the cross-entropy loss
        legal = x[:, 2].flatten(1).bool()
        logits = self.policy(h).masked_fill(~legal, -1e9)
        return logits, self.value(h)


def new_optimizer(bot, lr=1e-3, weight_decay=1e-4):
    # AdamW's decoupled weight decay plays the part of AlphaZero's L2 term and
    # keeps the network from memorising the replay buffer
    return torch.optim.AdamW(bot.parameters(), lr=lr, weight_decay=weight_decay)


def save_checkpoint(bot, optimizer, path=TRAINED_PATH, **extra):
    torch.save({
        'config': bot.config,
        'model_state_dict': bot.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        **extra,
    }, path)


def load_checkpoint(path=TRAINED_PATH):
    # Returns the checkpoint dict, or None if there is no file. A checkpoint from
    # before the network had a saved config used a different input encoding, so
    # it can't be used with this code
    path = pathlib.Path(path)
    if not path.exists():
        return None
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    config = checkpoint.get('config')
    if config is None or config['in_planes'] != IN_PLANES:
        raise ValueError(f"{path} was saved with a different input encoding. "
                         f"Move it aside to train a new network from scratch.")
    return checkpoint


def load_net(path=TRAINED_PATH):
    # The network saved at path, built with the shape it was saved with
    checkpoint = load_checkpoint(path)
    if checkpoint is None:
        raise FileNotFoundError(path)
    bot = Net(**checkpoint['config'])
    bot.load_state_dict(checkpoint['model_state_dict'])
    return bot
