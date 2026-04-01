#!/usr/bin/env bash
set -e

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $1"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $1"; }
error()   { echo -e "${RED}[setup]${NC} $1"; exit 1; }

# ─── Python ───────────────────────────────────────────────────────────────────
info "Checking for Python 3.10+..."
PROJECT_PYTHON=""
if command -v python3.12 &>/dev/null; then
    PROJECT_PYTHON="python3.12"
elif command -v python3.11 &>/dev/null; then
    PROJECT_PYTHON="python3.11"
elif command -v python3.10 &>/dev/null; then
    PROJECT_PYTHON="python3.10"
elif command -v python3 &>/dev/null; then
    if python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
        PROJECT_PYTHON="python3"
    fi
fi

if [ -z "$PROJECT_PYTHON" ]; then
    error "Python 3.10+ not found. Install Python 3.10+ and re-run."
fi

PYTHON_VERSION=$($PROJECT_PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
info "Using $PROJECT_PYTHON (Python $PYTHON_VERSION)"

# ─── Linux runtime deps (OpenCV) ─────────────────────────────────────────────
# opencv-python requires system libs such as libGL and glib on Linux.
if [ "$(uname -s)" = "Linux" ]; then
    info "Checking Linux runtime libraries required by OpenCV..."
    MISSING_DEPS=()

    if ! $PROJECT_PYTHON -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1; then
        MISSING_DEPS+=("libgl1")
    fi
    if ! $PROJECT_PYTHON -c "import ctypes; ctypes.CDLL('libglib-2.0.so.0')" >/dev/null 2>&1; then
        MISSING_DEPS+=("libglib2.0-0")
    fi

    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        info "OpenCV runtime libraries are present"
    elif command -v apt-get >/dev/null 2>&1; then
        info "Missing runtime packages: ${MISSING_DEPS[*]}"
        if [ "${EUID:-$(id -u)}" -eq 0 ]; then
            apt-get update
            apt-get install -y "${MISSING_DEPS[@]}"
            info "Installed Linux runtime packages for OpenCV"
        elif command -v sudo >/dev/null 2>&1; then
            info "Attempting to install runtime packages with sudo..."
            if sudo apt-get update && sudo apt-get install -y "${MISSING_DEPS[@]}"; then
                info "Installed Linux runtime packages for OpenCV"
            else
                warn "Automatic install failed. Run manually:"
                warn "sudo apt-get update && sudo apt-get install -y ${MISSING_DEPS[*]}"
            fi
        else
            warn "Missing runtime packages and no sudo available. Run as root:"
            warn "apt-get update && apt-get install -y ${MISSING_DEPS[*]}"
        fi
    else
        warn "Missing runtime libraries for OpenCV and apt-get is unavailable."
        warn "Install equivalents for: ${MISSING_DEPS[*]}"
    fi
fi

# ─── Poetry ───────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
POETRY_BIN="$HOME/.local/bin/poetry"

if command -v poetry &>/dev/null; then
    info "Poetry already installed ($(poetry --version))"
else
    info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | $PROJECT_PYTHON -

    # Persist to shell config
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_RC" ]; then
        grep -q 'HOME/.local/bin' "$SHELL_RC" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        info "Added Poetry to PATH in $SHELL_RC"
    fi

    if [ ! -x "$POETRY_BIN" ]; then
        error "Poetry install did not produce $POETRY_BIN."
    fi
    info "Poetry installed ($($POETRY_BIN --version))"
fi

POETRY_CMD="${POETRY_BIN}"
if command -v poetry &>/dev/null; then
    POETRY_CMD="poetry"
fi

# ─── Poetry shell plugin ──────────────────────────────────────────────────────
if $POETRY_CMD self show plugins 2>/dev/null | grep -q 'poetry-plugin-shell'; then
    info "poetry-plugin-shell already installed"
else
    info "Installing poetry-plugin-shell..."
    $POETRY_CMD self add poetry-plugin-shell
fi

# ─── Install dependencies ─────────────────────────────────────────────────────
cd "$(dirname "$0")"
info "Configuring Poetry to use $PROJECT_PYTHON..."
$POETRY_CMD env use "$PROJECT_PYTHON"

info "Refreshing poetry.lock to match pyproject.toml..."
$POETRY_CMD lock --no-interaction

info "Installing project dependencies..."
$POETRY_CMD install --no-interaction

# ─── PyTorch / torchvision / timm (CUDA-aware) ───────────────────────────────
# These are intentionally installed outside pyproject.toml so setup can choose
# CPU-only wheels on machines without CUDA and avoid pulling large nvidia-* deps.
info "Checking CUDA availability for PyTorch install..."
CUDA_AVAILABLE=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CUDA_AVAILABLE=1
fi

if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    info "CUDA-capable GPU detected; installing default torch/torchvision wheels..."
    $POETRY_CMD run pip install --upgrade torch torchvision
else
    info "No CUDA-capable GPU detected; installing CPU-only torch/torchvision wheels..."
    $POETRY_CMD run pip uninstall -y torch torchvision >/dev/null 2>&1 || true
    $POETRY_CMD run python - <<'PY'
import importlib.metadata as m
import subprocess
import sys

pkgs = sorted(
    d.metadata["Name"]
    for d in m.distributions()
    if d.metadata["Name"] and d.metadata["Name"].lower().startswith("nvidia-")
)
if pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])
PY
    $POETRY_CMD run pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi

info "Installing timm without re-resolving torch dependencies..."
$POETRY_CMD run pip install --upgrade --no-deps timm

# ─── mmcv stub ────────────────────────────────────────────────────────────────
# mmcv cannot be pip-installed on Python 3.12 + PyTorch 2.7 because OpenMMLab
# has not released pre-built wheels for this combination yet.  Metric3D v2 only
# uses mmcv.utils.{Config, DictAction} for inference, both of which are provided
# by mmengine (the official successor).  We copy a minimal stub into the
# virtualenv so torch.hub.load("YvanYin/Metric3D", ...) works out of the box.
info "Installing mmcv stub (delegates to mmengine)..."
STUB_SRC="$(dirname "$0")/generator/mmcv_stub/mmcv"
SITE=$($POETRY_CMD run python -c "import site; print(site.getsitepackages()[0])")
cp -r "$STUB_SRC" "$SITE/"
info "mmcv stub installed to $SITE/mmcv"

# ─── ffmpeg (via imageio-ffmpeg — no sudo required) ──────────────────────────
# imageio-ffmpeg ships a static ffmpeg binary inside the venv.
# System ffmpeg is used if present; imageio-ffmpeg acts as a no-sudo fallback.
info "ffmpeg bundled via imageio-ffmpeg (already in pyproject.toml dependencies)"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Setup complete! Launching poetry shell...${NC}"
echo ""
exec $POETRY_CMD shell
