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

## Project layout

- `src/utt_smart/`: package for future game rules, Monte Carlo tree search,
  neural-network models, and self-play training.
- `.devcontainer/devcontainer.json`: reproducible development workspace.
- `pyproject.toml`: package metadata and optional `training` dependencies.
- `uv.lock`: locked dependency versions used by `uv sync`.

This is the initial project scaffold; game logic, search, and training are not
implemented yet. There is no test suite yet. Training checkpoints and run
outputs are ignored by Git.