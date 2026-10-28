#!/usr/bin/env bash
# run.sh — SterileSpace Bimanual Spatial Computing Launcher
# Usage:
#   bash run.sh           -> launches app.py      (2D OpenCV viewer)
#   bash run.sh 3d        -> launches sterile_3d.py (PyVista 3D viewer)
#   bash run.sh sterile   -> launches SterileSpace Web App (start_web.sh)
set -e
ENV_NAME="sterile-space"

# Always run from the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "❌ conda not found. Please activate conda environment sterile-space."
    exit 1
}

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Pick which script to launch
if [ "${1}" = "3d" ]; then
    echo "🧠 Starting SterileSpace 3D — PyVista Volumetric Viewer (sterile_3d.py) ..."
    python sterile_3d.py
elif [ "${1}" = "sterile" ] || [ "${1}" = "wonder" ] || [ "${1}" = "web" ]; then
    echo "🫲 Starting SterileSpace Web App (start_web.sh) ..."
    ./start_web.sh "${@:2}"
else
    echo "🩻 Starting SterileSpace 2D — Bimanual Spatial Clinical Viewer (app.py) ..."
    python app.py
fi
