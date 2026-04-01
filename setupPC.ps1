Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($Message) {
    Write-Host "[setup]" -ForegroundColor Green -NoNewline
    Write-Host " $Message"
}

function Warn($Message) {
    Write-Host "[setup]" -ForegroundColor Yellow -NoNewline
    Write-Host " $Message"
}

function Fail($Message) {
    Write-Host "[setup]" -ForegroundColor Red -NoNewline
    Write-Host " $Message"
    exit 1
}

function Add-UserPathIfMissing($Dir) {
    if (-not (Test-Path $Dir)) {
        return
    }

    $sessionParts = @($env:Path -split ';' | Where-Object { $_ -ne "" })
    if ($sessionParts -notcontains $Dir) {
        $env:Path = "$Dir;$env:Path"
        Info "Added to current PATH: $Dir"
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userParts = @($userPath -split ';' | Where-Object { $_ -ne "" })
    if ($userParts -notcontains $Dir) {
        $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $Dir } else { "$userPath;$Dir" }
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Info "Added Poetry path to user PATH: $Dir"
    }
}

Info "Checking for Python 3.10+..."
$projectPythonPath = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
    $preferredVersions = @("3.12", "3.11", "3.10")
    foreach ($v in $preferredVersions) {
        try {
            $pyPath = (& py "-$v" -c "import sys; print(sys.executable)").Trim()
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($pyPath)) {
                $projectPythonPath = $pyPath
                break
            }
        } catch {}
    }
}

if (-not $projectPythonPath -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $projectPythonPath = (Get-Command python).Source
        }
    } catch {}
}

if (-not $projectPythonPath) {
    Fail "Python 3.10+ not found. Install Python 3.10+ and re-run."
}

$pythonVersion = & $projectPythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Info "Using Python $pythonVersion at $projectPythonPath"

$poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
if ($poetryCmd) {
    Info "Poetry already installed ($(& poetry --version))"
} else {
    Info "Installing Poetry..."
    $installer = (Invoke-WebRequest -Uri "https://install.python-poetry.org" -UseBasicParsing).Content
    $installer | & $projectPythonPath -

    $poetryPaths = @(
        (Join-Path $env:APPDATA "Python\Scripts"),
        (Join-Path $env:APPDATA "pypoetry\venv\Scripts"),
        (Join-Path $env:USERPROFILE ".local\bin")
    )
    foreach ($p in $poetryPaths) {
        Add-UserPathIfMissing $p
    }

    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        Fail "Poetry was installed but is not on PATH yet. Open a new PowerShell session and re-run."
    }

    Info "Poetry installed ($(& poetry --version))"
}

$hasShellPlugin = $false
try {
    $pluginList = poetry self show plugins 2>$null
    if ($pluginList | Select-String -Quiet "poetry-plugin-shell") {
        $hasShellPlugin = $true
    }
} catch {
    $hasShellPlugin = $false
}

if ($hasShellPlugin) {
    Info "poetry-plugin-shell already installed"
} else {
    Info "Installing poetry-plugin-shell..."
    poetry self add poetry-plugin-shell
}

Info "Refreshing poetry.lock to match pyproject.toml..."
Set-Location -Path $PSScriptRoot
Info "Configuring Poetry to use Python 3.10+..."
poetry env use "$projectPythonPath"
poetry lock --no-interaction

Info "Installing project dependencies..."
poetry install --no-interaction

# ─── PyTorch / torchvision / timm (CUDA-aware) ───────────────────────────────
# These are intentionally installed outside pyproject.toml so setup can choose
# CPU-only wheels on machines without CUDA and avoid pulling large nvidia-* deps.
Info "Checking CUDA availability for PyTorch install..."
$cudaAvailable = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        & nvidia-smi -L *> $null
        if ($LASTEXITCODE -eq 0) {
            $cudaAvailable = $true
        }
    } catch {
        $cudaAvailable = $false
    }
}

if ($cudaAvailable) {
    Info "CUDA-capable GPU detected; installing default torch/torchvision wheels..."
    poetry run pip install --upgrade torch torchvision
} else {
    Info "No CUDA-capable GPU detected; installing CPU-only torch/torchvision wheels..."
    poetry run pip uninstall -y torch torchvision *> $null
    poetry run python -c "import importlib.metadata as m, subprocess, sys; pkgs=sorted(d.metadata['Name'] for d in m.distributions() if d.metadata['Name'] and d.metadata['Name'].lower().startswith('nvidia-')); subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', *pkgs]) if pkgs else None"
    poetry run pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
}

Info "Installing timm without re-resolving torch dependencies..."
poetry run pip install --upgrade --no-deps timm

# ─── mmcv stub ────────────────────────────────────────────────────────────────
# mmcv cannot be pip-installed on Python 3.12 + PyTorch 2.7 because OpenMMLab
# has not released pre-built wheels for this combination yet.  Metric3D v2 only
# uses mmcv.utils.{Config, DictAction} for inference, both of which are provided
# by mmengine (the official successor).  We copy a minimal stub into the
# virtualenv so torch.hub.load("YvanYin/Metric3D", ...) works out of the box.
Info "Installing mmcv stub (delegates to mmengine)..."
$stubSrc = Join-Path $PSScriptRoot "generator\mmcv_stub\mmcv"
$site = & poetry run python -c "import site; print(site.getsitepackages()[0])"
$dest = Join-Path $site "mmcv"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
Copy-Item $stubSrc $dest -Recurse
Info "mmcv stub installed to $dest"

Write-Host ""
Write-Host "Setup complete! Launching poetry shell..." -ForegroundColor Green
Write-Host ""
poetry shell
