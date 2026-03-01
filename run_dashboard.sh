#!/usr/bin/env bash
# Run the Sign Language dashboard locally on macOS.
# Creates a virtual environment (venv/) on first run, installs requirements,
# then launches Streamlit on http://localhost:8501

set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"

# ── Find Python 3.8+ ────────────────────────────────────────────────────────
for py in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$py" &>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "❌  Python 3 not found. Install it via: brew install python3"
    exit 1
fi

echo "✅  Using Python: $($PYTHON --version)"

# ── Create / reuse virtual environment ─────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "🔧  Creating virtual environment in $VENV_DIR …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Bootstrap pip if it's missing from the venv
if ! "$VENV_DIR/bin/python" -m pip --version &>/dev/null 2>&1; then
    echo "🔧  Bootstrapping pip …"
    # Try ensurepip first, then fall back to get-pip.py
    if ! "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null; then
        echo "   ensurepip failed – recreating venv with --copies …"
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv --copies "$VENV_DIR"
        "$VENV_DIR/bin/python" -m ensurepip --upgrade || \
            curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python"
    fi
fi

source "$VENV_DIR/bin/activate"

# ── Install / upgrade requirements ─────────────────────────────────────────
echo "📦  Installing requirements …"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
# Install core dashboard dependencies
"$VENV_DIR/bin/python" -m pip install --quiet streamlit numpy pandas pillow scikit-learn matplotlib seaborn "kagglehub[pandas-datasets]"

echo ""
echo "🚀  Starting dashboard at http://localhost:8501"
echo "    Press Ctrl+C to stop."
echo ""

# Skip Streamlit's first-run email prompt
mkdir -p ~/.streamlit
if [[ ! -f ~/.streamlit/credentials.toml ]]; then
    printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
fi

"$VENV_DIR/bin/python" -m streamlit run dashboard.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
