#!/usr/bin/env bash
# Launch all GemmaBot demos locally.
#   :8002  Panda pick-and-place (robosuite)
#   :8001  PushT hybrid controller
#   :8003  ZTP replay viewer (LeRobot datasets)
# Logs stream to runs/*.log. Ctrl-C stops everything.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING=1

if [ -z "${CEREBRAS_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
  fi
fi

mkdir -p "$PROJECT_DIR/runs"
cd "$PROJECT_DIR"

cleanup() {
  echo
  echo "Stopping demos..."
  kill $PANDA_PID $PUSHT_PID $REPLAY_PID 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "Starting GemmaBot demos..."
echo "  Panda    http://localhost:8002/robot_live"
echo "  PushT    http://localhost:8001/"
echo "  Replay   http://localhost:8003/"
echo "Logs in runs/*.log. Ctrl-C to stop."
echo

uv run python -m src.web.robosuite_server > runs/panda.log 2>&1 &
PANDA_PID=$!

uv run python -m src.web.pusht_server > runs/pusht.log 2>&1 &
PUSHT_PID=$!

uv run python -m src.web.replay_server > runs/replay.log 2>&1 &
REPLAY_PID=$!

wait
