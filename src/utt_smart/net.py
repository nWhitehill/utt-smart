import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class Net(nn.Module):
    def __init__(self, blocks=5, ch=64, moves = 81):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        )
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # Each cell is one move, so a 1x1 conv to a single channel gives one logit per move
        self.policy = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Flatten())
        self.value = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Flatten(), nn.Linear(moves, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        # x is (N, 3, 9, 9): my pieces, opponent's pieces, legal moves
        h = self.trunk(self.stem(x))
        # Illegal moves get a huge negative logit so softmax gives them ~0 probability.
        # -1e9 rather than -inf avoids 0 * -inf = NaN in the cross-entropy loss
        legal = x[:, 2].flatten(1).bool()
        logits = self.policy(h).masked_fill(~legal, -1e9)
        return logits, self.value(h)
