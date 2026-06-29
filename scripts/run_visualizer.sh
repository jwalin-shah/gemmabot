#!/bin/bash
# ---------------------------------------------------------------------------
# run_visualizer.sh — Start the Gemma 4 Experiment Visualizer
#
# Usage:
#   ./scripts/run_visualizer.sh
#
# Opens http://localhost:8899 in the default browser after startup.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISUALIZER="$SCRIPT_DIR/visualizer.py"

if [ ! -f "$VISUALIZER" ]; then
    echo "Error: $VISUALIZER not found."
    echo "Run this script from the project root or scripts/ directory."
    exit 1
fi

echo "Starting Gemma 4 Experiment Visualizer..."
echo ""

# Check for python3
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi

# Launch the visualizer in the background
"$PYTHON" "$VISUALIZER" &
SERVER_PID=$!

# Wait a moment for the server to start, then open the browser
sleep 1

# Open browser (macOS default)
if command -v open &>/dev/null; then
    open "http://localhost:8899"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8899"
elif command -v sensible-browser &>/dev/null; then
    sensible-browser "http://localhost:8899"
fi

echo "Server PID: $SERVER_PID"
echo "Press Ctrl+C to stop."

# Wait for the server process
wait "$SERVER_PID"
