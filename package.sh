#!/usr/bin/env bash
# Builds the self-contained bot tarball for Bots Playing Games: the network,
# and the utt_smart package with its runtime dependencies (torch and what it
# needs) installed into lib/ for one Python version and platform. Each bot's
# bot.py is an entrypoint: extract the tarball anywhere and run, say,
# `python3 lib/utt_smart/bots/nmct/bot.py ULTIMATE_TIC_TAC_TOE X 10s` with that
# Python; nothing else needs installing. bot.py puts lib/ on sys.path itself.
#
#   ./package.sh                         # dist/utt-smart-bot.tar.gz for the runner image
#   ./package.sh --model runs/from-scratch/checkpoints/gen0150.pt
#   ./package.sh --python 3.12 --platform x86_64-manylinux_2_28 -o dist/local.tar.gz
#
# Layout inside the tarball:
#   models/nmct.pt                 the network nmct plays with (--model, default models/nmct.pt)
#   lib/                           utt_smart, torch (CPU-only build) and the other runtime deps
#   lib/utt_smart/bots/nmct/bot.py entrypoint for nmct
#   lib/utt_smart/bots/mct/bot.py  entrypoint for mct (standard library only)
#
# The defaults match sandbox-runner's python3_14 image (python:3.14-slim, linux/amd64).
# Wheels are compiled per Python minor version, so the tarball only runs on the
# version it was built for.
set -euo pipefail

cd "$(dirname "$0")"

python_version=3.14
platform=x86_64-manylinux_2_28
# CPU-only torch. The default PyPI torch wheel for Linux depends on several GB of
# CUDA libraries, and the bot runs on CPU.
torch_index=https://download.pytorch.org/whl/cpu
model=models/nmct.pt
output=dist/utt-smart-bot.tar.gz

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_version="$2"; shift 2 ;;
        --platform) platform="$2"; shift 2 ;;
        --torch-index) torch_index="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        -o|--output) output="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$model" ]]; then
    echo "model not found: $model" >&2
    exit 1
fi

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

# The runtime versions pinned in uv.lock (no training extras, no dev group), used
# as constraints rather than requirements: the lock also pins the CUDA packages
# the default torch wheel needs, which the CPU build doesn't, and a constraint
# only applies to a package something actually requires.
uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements.txt \
    > "$stage/constraints.txt"

# This project as a wheel (pure Python, so it installs on any target).
uv build --wheel --out-dir "$stage/wheel" --quiet

# Installs that wheel and its deps into lib/,
# with wheels for the target Python and platform rather than this machine's.
# The torch index also mirrors older copies of common packages (setuptools,
# sympy, ...), so versions are picked from both indexes, not first-index-wins.
# That is safe here because the constraints pin every version, and it picks
# torch 2.x.y+cpu over PyPI's 2.x.y because the +cpu local version sorts higher.
uv pip install \
    --target "$stage/bundle/lib" \
    --python-version "$python_version" \
    --python-platform "$platform" \
    --only-binary :all: \
    --index "$torch_index" \
    --default-index https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -c "$stage/constraints.txt" \
    "$stage"/wheel/*.whl

# Compiled bytecode and console scripts aren't needed at runtime.
find "$stage/bundle/lib" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$stage/bundle/lib/bin"

mkdir -p "$stage/bundle/models"
cp "$model" "$stage/bundle/models/nmct.pt"

mkdir -p "$(dirname "$output")"
tar -czf "$output" -C "$stage/bundle" .

echo "Wrote $output ($(du -h "$output" | cut -f1)) for Python $python_version on $platform"
