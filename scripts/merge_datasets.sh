#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python scripts/merge_datasets.py "$@"
fi

exec uv run --no-project --with h5py --with numpy \
  python scripts/merge_datasets.py "$@"
