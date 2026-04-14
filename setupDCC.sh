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
CONDA_ENV_NAME="pest_gen"
PYTHON_VERSION="3.11"

# ─── Locate or install Miniconda ──────────────────────────────────────────────
info "Looking for conda..."
CONDA_BIN=""
for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/hpc/group/$(whoami)/miniconda3/bin/conda" \
    "$(which conda 2>/dev/null)"; do
    if [ -x "$candidate" ]; then
        CONDA_BIN="$candidate"
        break
    fi
done

if [ -z "$CONDA_BIN" ]; then
    info "conda not found — installing Miniconda to $HOME/miniconda3..."
    INSTALLER="$HOME/miniconda_installer.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$INSTALLER"
    CONDA_BIN="$HOME/miniconda3/bin/conda"
    info "Miniconda installed."
fi

info "Using conda: $CONDA_BIN"
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ─── Create / reuse conda environment ─────────────────────────────────────────
if "$CONDA_BIN" env list | grep -q "^${CONDA_ENV_NAME} "; then
    info "Conda env '${CONDA_ENV_NAME}' already exists — reusing."
else
    info "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
    "$CONDA_BIN" create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$CONDA_ENV_NAME"
PY="$(which python)"
info "Using $PY ($(${PY} --version))"

# ─── Install PyTorch (CUDA provided by Apptainer container) ──────────────────
info "Installing PyTorch + torchvision + timm..."
pip install --upgrade torch torchvision timm

# ─── Install project dependencies ─────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# ─── mmcv stub ────────────────────────────────────────────────────────────────
info "Installing mmcv stub..."
STUB_SRC="$SCRIPT_DIR/generator/mmcv_stub/mmcv"
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
if [ -d "$STUB_SRC" ]; then
    cp -r "$STUB_SRC" "$SITE/"
    info "mmcv stub installed to $SITE/mmcv"
else
    warn "mmcv stub not found at $STUB_SRC — skipping"
fi

# ─── Persist activation in .bashrc ────────────────────────────────────────────
if ! grep -q "conda activate ${CONDA_ENV_NAME}" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# pest_gen environment" >> "$HOME/.bashrc"
    echo "source \"$CONDA_BASE/etc/profile.d/conda.sh\"" >> "$HOME/.bashrc"
    echo "conda activate ${CONDA_ENV_NAME}" >> "$HOME/.bashrc"
    info "Added conda activate to ~/.bashrc"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "To run generation:"
echo "  cd $SCRIPT_DIR"
echo "  python -m generator.generate --count 100 --workers 14"
echo ""
echo "To run the web app:"
echo "  FLASK_APP=app flask run --host=0.0.0.0 --port=5000"
