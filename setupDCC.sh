#!/usr/bin/env bash
set -e

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $1"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $1"; }
error() { echo -e "${RED}[setup]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── PATH ─────────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# ─── Python ───────────────────────────────────────────────────────────────────
info "Detecting Python 3.9+..."
PY=""
for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done
[ -z "$PY" ] && error "Python 3.9+ not found."
info "Using $PY ($(${PY} --version))"

# ─── pip install dependencies ─────────────────────────────────────────────────
info "Upgrading pip..."
$PY -m pip install --user --upgrade pip

info "Installing project dependencies from requirements.txt..."
$PY -m pip install --user -r "$SCRIPT_DIR/requirements.txt"

# ─── PyTorch (CUDA-aware) ─────────────────────────────────────────────────────
info "Checking CUDA availability..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    info "CUDA GPU detected — installing default torch/torchvision/timm wheels..."
    $PY -m pip install --user --upgrade torch torchvision timm
else
    info "No CUDA GPU detected — installing CPU-only torch/torchvision/timm..."
    $PY -m pip install --user --upgrade \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision
    $PY -m pip install --user --upgrade --no-deps timm
fi

# ─── mmcv stub ────────────────────────────────────────────────────────────────
info "Installing mmcv stub (delegates to mmengine)..."
STUB_SRC="$SCRIPT_DIR/generator/mmcv_stub/mmcv"
SITE=$($PY -c "import site; print(site.getusersitepackages())")
if [ -d "$STUB_SRC" ]; then
    mkdir -p "$SITE"
    cp -r "$STUB_SRC" "$SITE/"
    info "mmcv stub installed to $SITE/mmcv"
else
    warn "mmcv stub not found at $STUB_SRC — skipping"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "To run generation:"
echo "  cd $SCRIPT_DIR"
echo "  python -m generator.generate --count 100 --workers 14"
