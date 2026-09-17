#!/usr/bin/env bash
# Durable local GPU queue; see experiment-log.md section 55.
set -euo pipefail
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
exec "$PY" -u "$ROOT/code/cluster/epoch_length_20x.py" --root "$ROOT" "$@"
