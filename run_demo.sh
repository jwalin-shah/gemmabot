#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== GemmaBot: Cerebras × Gemma 4 Multi-Agent Robotics Demo ==="
echo ""

# Check API key
if [ -z "${CEREBRAS_API_KEY:-}" ]; then
  if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading .env file..."
    set -a; source "$PROJECT_DIR/.env"; set +a
  else
    echo "❌  CEREBRAS_API_KEY not set and no .env file found."
    echo "    Copy .env.example to .env and add your key."
    exit 1
  fi
fi

# Ensure dependencies
echo "📦  Installing dependencies..."
uv pip install --quiet -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || \
  pip install --quiet -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "🚀  Starting demo..."
echo ""

cd "$PROJECT_DIR"
python -m src.demo "$@"