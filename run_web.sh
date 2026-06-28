#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== GemmaBot Web UI ==="
if [ -z "${CEREBRAS_API_KEY:-}" ]; then
  if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
  else
    echo "❌ CEREBRAS_API_KEY not set"
    exit 1
  fi
fi
echo "📦 Installing deps..."
uv pip install --quiet -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || pip install --quiet -r "$PROJECT_DIR/requirements.txt"
pip install --quiet fastapi uvicorn sse-starlette aiofiles python-multipart 2>/dev/null || true
echo ""
echo "🌐 Starting web server at http://localhost:8000"
echo ""
cd "$PROJECT_DIR"
uvicorn src.web.server:app --host 0.0.0.0 --port 8000 --reload
