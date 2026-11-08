#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "--training" ]]; then
  if [[ -z "${AMP_ISAACLAB:-}" || ! -f "$AMP_ISAACLAB/isaaclab.sh" ]]; then
    echo 'Set AMP_ISAACLAB to the IsaacLab 2.1.0 installation directory.' >&2
    exit 1
  fi
  exec "$AMP_ISAACLAB/isaaclab.sh" -p -m amp_lab.cli doctor --training
fi
exec "${AMP_PYTHON:-$ROOT/.venv/bin/python}" -m amp_lab.cli doctor "$@"
