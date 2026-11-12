import torch

# Rotating or mirroring the whole 9x9 grid gives an equally valid position: every
# subgame and every cell within it move by the same transform, so the rule that
# the cell you play picks the opponent's next subgame still holds. That gives 8
# versions of each training position for free.
#
# Each symmetry is a permutation of the 81 cells, used as a gather:
# transformed[..., i] = original[..., perm[i]]. The same permutation applies to
# the input planes and the policy target, so every probability stays on the
# cell it belongs to; the value target doesn't change.


def _permutations():
    grid = torch.arange(81).view(9, 9)
    perms = []
    for flip in (False, True):
        g = torch.flip(grid, dims=(1,)) if flip else grid
        for k in range(4):
            perms.append(torch.rot90(g, k, dims=(0, 1)).reshape(81))
    return torch.stack(perms)


PERMUTATIONS = _permutations()  # (8, 81), PERMUTATIONS[0] is the identity


def random_symmetries(planes, pi):
    # planes (N, C, 9, 9), pi (N, 81): gives each position its own random symmetry
    n, c = planes.shape[:2]
    perms = PERMUTATIONS.to(planes.device)[torch.randint(len(PERMUTATIONS), (n,), device=planes.device)]
    planes = torch.gather(planes.reshape(n, c, 81), 2, perms.unsqueeze(1).expand(n, c, 81)).view(n, c, 9, 9)
    pi = torch.gather(pi, 1, perms)
    return planes, pi
