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

# ─── PATH: ensure ~/.local/bin is available ───────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# ─── Python ───────────────────────────────────────────────────────────────────
info "Detecting Python 3.10+..."
PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done
[ -z "$PY" ] && error "Python 3.10+ not found."
info "Using $PY ($(${PY} --version))"

# ─── Core dependencies ────────────────────────────────────────────────────────
info "Installing core dependencies from requirements.txt..."
$PY -m pip install --user --upgrade pip
$PY -m pip install --user -r "$SCRIPT_DIR/requirements.txt"

# ─── PyTorch / torchvision / timm (CUDA-aware) ───────────────────────────────
info "Checking CUDA availability for PyTorch install..."
CUDA_AVAILABLE=0
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    CUDA_AVAILABLE=1
fi

if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    info "CUDA GPU detected — installing CUDA torch/torchvision..."
    $PY -m pip install --user --upgrade torch torchvision
else
    info "No CUDA GPU detected — installing CPU-only torch/torchvision..."
    $PY -m pip install --user --upgrade \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision
fi

info "Installing timm (no-deps to avoid overriding torch)..."
$PY -m pip install --user --upgrade --no-deps timm

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
echo "To run the app inside Apptainer:"
echo "  cd $SCRIPT_DIR"
echo "  FLASK_APP=app flask run --host=0.0.0.0 --port=5000"
