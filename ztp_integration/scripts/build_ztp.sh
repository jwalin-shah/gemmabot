#!/usr/bin/env bash
# Clone and build the ztp-runtime shared library.
#
# Prerequisites: Rust toolchain (run install_rust.sh first if needed).
#
# Output: ztp_integration/vendor/ztp-runtime/target/release/libztp_runtime.dylib
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INTEGRATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_DIR="$INTEGRATION_DIR/vendor"
ZTP_DIR="$VENDOR_DIR/ztp-runtime"

echo "=== Building John Kruze ztp-runtime ==="
echo ""

# Check for Rust
if ! command -v rustc &>/dev/null; then
    echo "❌  Rust not found. Run scripts/install_rust.sh first."
    exit 1
fi
echo "✓  Rust: $(rustc --version)"
echo ""

# Clone if needed
if [ ! -d "$ZTP_DIR" ]; then
    echo "→  Cloning ztp-runtime into vendor/"
    mkdir -p "$VENDOR_DIR"
    git clone https://github.com/johnkruze/ztp-runtime.git "$ZTP_DIR"
    echo ""
else
    echo "✓  ztp-runtime already cloned in vendor/"
    echo "   (delete vendor/ztp-runtime to re-clone fresh)"
    echo ""
fi

# Build release
echo "→  Building release (this may take a few minutes)..."
cd "$ZTP_DIR"
cargo build --release
echo ""

# Verify the shared library exists
LIB_NAME="libztp_runtime.dylib"
LIB_PATH="$ZTP_DIR/target/release/$LIB_NAME"
if [ -f "$LIB_PATH" ]; then
    FILE_SIZE=$(stat -f%z "$LIB_PATH" 2>/dev/null || stat -c%s "$LIB_PATH" 2>/dev/null)
    echo "✓  Build complete!"
    echo "   Library: $LIB_PATH"
    echo "   Size: $FILE_SIZE bytes"
else
    echo "⚠️  Release library not found at expected path."
    echo "   Trying debug build..."
    cargo build
    LIB_PATH="$ZTP_DIR/target/debug/$LIB_NAME"
    if [ -f "$LIB_PATH" ]; then
        echo "✓  Debug build complete: $LIB_PATH"
    else
        echo "❌  Build seems to have produced unexpected output."
        ls "$ZTP_DIR/target/release/" 2>/dev/null || echo "   (no target/release/)"
    fi
fi

echo ""
echo "=== ztp-runtime ready ==="
echo "Set ZTP_LIB_PATH=$LIB_PATH to use this build"
