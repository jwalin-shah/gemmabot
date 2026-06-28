#!/usr/bin/env bash
# Install Rust toolchain via rustup (if not already installed).
set -euo pipefail

echo "=== Installing Rust toolchain ==="

if command -v rustc &>/dev/null; then
    echo "✓ Rust already installed: $(rustc --version)"
    exit 0
fi

echo "→ Installing rustup (Rust toolchain installer)..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

source "$HOME/.cargo/env" 2>/dev/null || true

echo "✓ Rust installed: $(rustc --version)"
echo "✓ Cargo installed: $(cargo --version)"
