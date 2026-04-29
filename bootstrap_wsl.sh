#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap script must be run inside WSL2/Linux." >&2
  exit 1
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
VENV_DIR="${VENV_DIR:-.venv-wsl}"

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Missing $PYTHON_CMD. Install Python 3 inside WSL first." >&2
  exit 1
fi

"$PYTHON_CMD" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
"$VENV_DIR/bin/python" -m pip install -r requirements-wsl.txt

echo "WSL bootstrap complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Compile with:  python compile.py memory"
echo "Search with:   python consume.py \"你的查询\""
echo "Service with:  ./launch_memory_service.sh"
