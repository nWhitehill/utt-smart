# utt-smart

Python workspace for training a neural network to play Ultimate Tic Tac Toe
using Monte Carlo simulations and tree search.

## Development container

Open this repository in GitHub Codespaces, or use **Dev Containers: Reopen in
Container** in VS Code with Docker and the Dev Containers extension installed.
The container provides:

- Python 3.12 with the Python and Pylance editor extensions.
- [uv](https://docs.astral.sh/uv/) for package management. On creation it runs
  `uv sync --extra training`, which creates `.venv/` and installs this package
  in editable mode with NumPy and PyTorch from `uv.lock`.
- The Claude Code CLI and the Claude Code VS Code extension.

Claude Code settings and credentials are stored in `~/.claude`
(`CLAUDE_CONFIG_DIR`), which is a Docker volume, so you only need to sign in
once per container: the login survives rebuilds of the same codespace or local
container. A brand-new codespace starts with an empty volume and needs a fresh
login.

The default container does not require a GPU. Installing PyTorch can take a few
minutes and requires internet access.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --extra training
uv run python -c "import utt_smart, numpy, torch; print('Training workspace ready')"
```

`uv sync` creates `.venv/`, installs Python 3.12 if needed, and installs the
locked dependencies. Use `uv run <command>` to run inside the environment, or
activate it with `source .venv/bin/activate` (`.venv\Scripts\Activate.ps1` in
PowerShell on Windows).

For a lightweight package-only installation, run `uv sync` without
`--extra training`.

### Managing dependencies

- `uv add <package>` adds a runtime dependency.
- `uv add --optional training <package>` adds a training dependency.
- `uv lock --upgrade` refreshes locked versions.

Commit `uv.lock` alongside any `pyproject.toml` changes.

## Training

```sh
uv run python training.py                    # 100 generations; resumes from trained.pt if it exists
uv run python training.py --generations 20 --lr 3e-4 --cores 16
uv run python evaluate.py checkpoints/gen0100.pt --games 200 --sims 50 200 --mct 1000 5000
uv run pytest
```

Training is AlphaZero-style self-play. Each generation:

1. **Self-play** (CPU, one worker process per core): 400 games of the network
   against itself, with the network guiding Monte Carlo tree search. Each worker
   batches its games' network calls. The search tree is kept from move to move.
   Moves use *playout cap randomization* (from KataGo): a quarter of moves get a
   full 200-simulation search with Dirichlet noise at the root, and only those
   become policy targets. The rest get a quick 40-simulation search. Every
   position gets the game result as its value target.
2. **Training** (GPU if there is one): the replay buffer holds the last 8
   generations. Each position is trained on about 4 times while it is in the
   buffer, each time in a random one of the board's 8 rotations and reflections.
   AdamW with weight decay.
3. **Evaluation**: 100 games against plain MCT with random playouts
   (1000 iterations a move), with the network on 200 simulations and no tree reuse.

The network (`src/utt_smart/nn/net.py`) is a 6-block, 64-channel residual CNN over
the 9x9 grid. It takes 7 input planes from the side to move: its pieces, the
opponent's pieces, the legal moves, subgames won by each side, drawn subgames,
and whether it is playing X. It outputs a policy over the 81 cells and a value
in [-1, 1].

Training saves `trained.pt` every generation, keeps a copy in
`checkpoints/genNNNN.pt` every 10 generations, and appends a row per generation
to `runs/training_log.csv`. A checkpoint from before the 7-plane input can't be
loaded, so move it aside to start over.

## Project layout

- `src/utt_smart/utt.py`: the game rules, shared by the bots and training.
- `src/utt_smart/bots/`: the bots, one package each.
  - `mct/`: Monte Carlo tree search with random playouts (`MC`), also the
    evaluation opponent.
  - `nmct/`: tree search guided by the network (PUCT, `NeuralMC`).
- `src/utt_smart/nn/`: the network (`net.py`) and its input planes
  (`encoding.py`).
  - `training/`: the self-play training loop (`train.py`); `train_multi.py`
    runs games on worker processes; `symmetry.py` holds the board symmetries
    used for augmentation.
- `training.py`, `evaluate.py`: command-line entry points.
- `tests/`: pytest checks for the symmetries, the search, and the self-play
  records.
- `.devcontainer/devcontainer.json`: reproducible development workspace.
- `pyproject.toml`: package metadata and optional `training` dependencies.
- `uv.lock`: locked dependency versions used by `uv sync`.
- `training_analysis.ipynb`: charts of the per-generation log that training
  appends to `runs/training_log.csv` (win rate against MCT, losses, self-play
  results, timings).

Training checkpoints and run outputs are ignored by Git.
