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
info "Checking for Python 3..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Please install Python 3.12+ and re-run."
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Found Python $PYTHON_VERSION"

# ─── Poetry ───────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

if command -v poetry &>/dev/null; then
    info "Poetry already installed ($(poetry --version))"
else
    info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -

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

    info "Poetry installed ($(poetry --version))"
fi

# ─── Install dependencies ─────────────────────────────────────────────────────
info "Installing project dependencies..."
cd "$(dirname "$0")"
poetry install

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "  Activate your environment with:"
echo -e "    ${YELLOW}poetry shell${NC}"
echo ""
echo "  Or run a single command without activating:"
echo -e "    ${YELLOW}poetry run python <script.py>${NC}"
echo ""
