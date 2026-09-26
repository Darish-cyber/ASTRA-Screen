#!/bin/bash
# ==============================================================================
# ASTRA-Screen: AI-Driven Spaceflight Component Burn-In & Screening System
# Indian Space Research Organisation (ISRO) | PS ID: 26170 | SIH 2026
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "    Launching ASTRA-Screen (ISRO Problem Statement ID: 26170)     "
echo "=================================================================="

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv .venv
    echo "[2/4] Installing dependencies..."
    .venv/bin/pip install -r requirements.txt
else
    echo "[1/4] Virtual environment verified."
fi

# Ensure datasets exist
if [ ! -f "data/sample_lot_burnin.csv" ]; then
    echo "[2/4] Generating physics-grounded burn-in datasets..."
    .venv/bin/python data/generate_datasets.py
else
    echo "[2/4] Sample burn-in datasets verified."
fi

# Run verification test
echo "[3/4] Running end-to-end aerospace pipeline verification..."
.venv/bin/python test_pipeline.py

# Ensure local light cleanroom theme config exists
if [ ! -f ".streamlit/config.toml" ]; then
    mkdir -p .streamlit
    cat << 'EOF' > .streamlit/config.toml
[theme]
base = "light"
primaryColor = "#0284c7"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#0f172a"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
EOF
fi

# Launch Streamlit GUI
echo "[4/4] Starting ASTRA-Screen Interactive Mission Control Dashboard..."
echo "Opening web interface at http://localhost:8501"
.venv/bin/streamlit run frontend/app.py \
    --server.port=8501 \
    --server.headless=true \
    --theme.base="light" \
    --theme.primaryColor="#0284c7" \
    --theme.backgroundColor="#f8fafc" \
    --theme.secondaryBackgroundColor="#f1f5f9" \
    --theme.textColor="#0f172a"
