#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${AMP_PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/scripts/workflow.py" evaluate "$@"
