# utt-smart

Python workspace for training a neural network to play Ultimate Tic Tac Toe
using Monte Carlo simulations and tree search.

## Development container

Open this repository in GitHub Codespaces, or use **Dev Containers: Reopen in
Container** in VS Code with Docker and the Dev Containers extension installed.
The container provides Python 3.12, Python editor extensions, and installs this
package in editable mode with NumPy and PyTorch.

The default container does not require a GPU. Installing PyTorch can take a few
minutes and requires internet access.

## Local setup

With Python 3.12 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[training]'
python -c "import utt_smart, numpy, torch; print('Training workspace ready')"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead.

For a lightweight package-only installation, use `python -m pip install -e .`.

## Project layout

- `src/utt_smart/`: package for future game rules, Monte Carlo tree search,
  neural-network models, and self-play training.
- `.devcontainer/devcontainer.json`: reproducible development workspace.
- `pyproject.toml`: package metadata and optional `training` dependencies.

This is the initial project scaffold; game logic, search, and training are not
implemented yet. There is no test suite yet. Training checkpoints and run
outputs are ignored by Git.