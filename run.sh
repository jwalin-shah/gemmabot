#!/usr/bin/env bash
# Launch all three GemmaBot demos.
#   :8000  landing page + real-image viewer
#   :8001  PushT hybrid controller
#   :8002  Panda pick-and-place
# Logs stream to runs/*.log. Ctrl-C stops everything.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING=1

if [ -z "${CEREBRAS_API_KEY:-}" ]; then
  if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
  else
    echo "ERROR: CEREBRAS_API_KEY not set and no .env file found."
    echo "       cp .env.example .env  # then add your key"
    exit 1
  fi
fi

mkdir -p "$PROJECT_DIR/runs"
cd "$PROJECT_DIR"

cleanup() {
  echo
  echo "Stopping demos..."
  kill $LANDING_PID $PUSHT_PID $PANDA_PID $REPLAY_PID 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "Starting GemmaBot demos..."
uv run python -m src.web.robosuite_server > runs/panda.log 2>&1 &
LANDING_PID=$!
uv run python -m src.web.pusht_server     > runs/pusht.log   2>&1 &
PUSHT_PID=$!
uv run python -m src.web.robosuite_server > runs/panda.log   2>&1 &
uv run python -m src.web.replay_server   > runs/replay.log 2>&1 &
PANDA_PID=$!

sleep 2
echo
echo "  Panda    http://localhost:8002/robot_live"
echo "  PushT    http://localhost:8001/"
echo "  Replay   http://localhost:8003/"
echo
echo "Logs in runs/*.log. Ctrl-C to stop."
wait
