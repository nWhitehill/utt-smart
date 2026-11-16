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

### Tests and lint

```sh
uv run pytest
uv run ruff check .        # --fix to apply the safe fixes
```

The **Test** GitHub workflow runs both on every pull request and push to `main`.
Ruff's rules are in `pyproject.toml`; notebooks are not linted.

## Training

Training runs from `training_analysis.ipynb` (see *Training from the notebook*).
From the command line:

```sh
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
4. **Ladder** (every snapshot, so every 10 generations by default): 40 games
   against the previous snapshot, both on 100 simulations, from 2-move random
   openings shared by each pair of games with the colours swapped. The result
   gives an Elo rating, chained back to the untrained network at 0. Unlike the
   MCT win rate, which tops out once the network beats MCT nearly every game,
   it keeps rising as long as the network improves.
5. **Probes**: the network's policy and value on a few fixed positions
   (`training/probes.py`), saved to `probes.jsonl`.

The network (`src/utt_smart/nn/net.py`) is a 6-block, 64-channel residual CNN over
the 9x9 grid. It takes 7 input planes from the side to move: its pieces, the
opponent's pieces, the legal moves, subgames won by each side, drawn subgames,
and whether it is playing X. It outputs a policy over the 81 cells and a value
in [-1, 1].

Training saves `trained.pt` every generation, keeps a copy in
`checkpoints/genNNNN.pt` every 10 generations (including `gen0000.pt`, the
untrained network, for a new run), and appends a row per generation to
`runs/training_log.csv`. Given a run directory (the notebook uses one), all of
these go in that directory instead. When the log's columns change, the old log is renamed and a new one
started. A checkpoint from before the 7-plane input can't be loaded, so move
it aside to start over.

### Training from the notebook

`training_analysis.ipynb` runs the loop through `Trainer`
(`training/train.py`) one generation at a time, in its own run directory
(`runs/from-scratch` by default). A dashboard is redrawn after every generation:
the win rate against MCT, the Elo ladder, both losses against their floors, and
the network's policy on the probe positions. Interrupting the cell stops cleanly
after shutting down the workers; running it again carries on, and changes to
`trainer.config` apply from the next generation. Later sections show the full
charts, how the probes changed over training, and single self-play positions
with the network's prior next to the search's policy target. The replay buffer
is only kept in memory, so after a kernel restart it starts empty.

## Playing on Bots Playing Games

Each bot's `bot.py` is its entrypoint on [Bots Playing Games](https://github.com/bots-playing-games):
pick the bot by giving the runner that file. Its bot class inherits from `Harness`
(`src/utt_smart/bots/harness.py`), which speaks the platform's stdin/stdout
protocol: it prints `ready`, then answers each `move` line with a cell index, and
exits on `end ...`. It converts between the server's board-by-board cell numbering
and `UTT`'s row-by-row numbering, checks every move against the server's legal
boards, and falls back to a random legal move if the bot fails, because three
rejected moves forfeit the game. Each move is searched for half the clock, 5 s of
the platform's 10 s. The runner passes no extra arguments, so a bot's settings
live in its `bot.py`.

To try a bot locally (`--think` sets the seconds per move, for testing only):

```sh
uv run python src/utt_smart/bots/nmct/bot.py ULTIMATE_TIC_TAC_TOE X 10s
uv run python src/utt_smart/bots/mct/bot.py ULTIMATE_TIC_TAC_TOE X 10s --think 1
```

then type a position, such as `move ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ? 0,1,2,3,4,5,6,7,8`.

| Bot | Entrypoint (in the tarball) | What it is | Needs |
|---|---|---|---|
| `nmct` | `lib/utt_smart/bots/nmct/bot.py` | tree search guided by the trained network | torch, and the network in `models/nmct.pt` |
| `mct` | `lib/utt_smart/bots/mct/bot.py` | Monte Carlo tree search with random playouts | standard library only |

`models/nmct.pt` is the network the bot plays with, committed to Git (the
`.gitignore` ignores every other `.pt` file). To play a newer network, copy a
snapshot over it, such as `runs/from-scratch/checkpoints/gen0150.pt`. The
checkpoint records its generation (`torch.load(...)["generation"]`).

To add a bot, make a package under `src/utt_smart/bots/` with a `bot.py` whose
class inherits from `Harness` and implements `choose_move`, as described in
`src/utt_smart/bots/__init__.py`. Copy the `sys.path` lines at the top of an
existing `bot.py` (they let the file run as a script) and end it with
`if __name__ == "__main__": YourBot.main()`. The tarball picks it up with no
other changes.

### Packaging the bot

`package.sh` builds the bot artifact, a self-contained tarball that holds every
bot:

```
models/nmct.pt   the network
lib/             utt_smart and its runtime dependencies (a CPU-only PyTorch build,
                 at the versions in uv.lock), including each bot's bot.py
```

Extract it anywhere and run a bot with the matching Python, for example
`python3 lib/utt_smart/bots/nmct/bot.py ULTIMATE_TIC_TAC_TOE X 10s`. Nothing else
needs to be installed: `bot.py` puts `lib/` on `sys.path` itself.

```sh
./package.sh                                   # dist/utt-smart-bot.tar.gz
./package.sh --model runs/from-scratch/checkpoints/gen0150.pt   # a different network
./package.sh --python 3.12 -o dist/local.tar.gz
```

The defaults match the runner's Python image: Python 3.14 on linux/amd64
(`--platform x86_64-manylinux_2_28`). Compiled wheels are tied to one Python
minor version, so the tarball runs only on the version it was built for. The
tarball is about 190 MB, and 780 MB extracted, almost all of it PyTorch.

## Project layout

- `src/utt_smart/utt.py`: the game rules, shared by the bots and training.
- `src/utt_smart/bots/`: the bots, one package each, with a `bot.py` that is
  the bot's entrypoint; `harness.py` holds the platform protocol they inherit.
  - `mct/`: Monte Carlo tree search with random playouts (`MC`), also the
    evaluation opponent.
  - `nmct/`: tree search guided by the network (PUCT, `NeuralMC`).
- `src/utt_smart/nn/`: the network (`net.py`) and its input planes
  (`encoding.py`).
  - `training/`: the self-play training loop (`train.py`, with `Trainer` and
    `TrainingConfig`); `train_multi.py` runs games on worker processes, with
    one or two networks; `symmetry.py` holds the board symmetries used for
    augmentation; `probes.py` the fixed probe positions; `plots.py` the
    notebook's charts and board drawings.
- `evaluate.py`: plays a network against MCT from the command line.
- `package.sh`: builds the bot tarball (see *Packaging the bot*).
- `models/nmct.pt`: the trained network `nmct` plays with.
- `tests/`: pytest checks for the harness and protocol, the symmetries, the
  search, the self-play records, the two-network evaluator and the ladder.
- `.devcontainer/devcontainer.json`: reproducible development workspace.
- `pyproject.toml`: package metadata and optional `training` dependencies.
- `uv.lock`: locked dependency versions used by `uv sync`.
- `training_analysis.ipynb`: runs training with a live dashboard, and charts
  the run's log, probes and self-play games (see *Training from the notebook*).

Training checkpoints and run outputs are ignored by Git.
