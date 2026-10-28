#!/usr/bin/env bash
# start_web.sh — Launch WonderSnap Web Application
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/wondersnap_web"

if [ ! -d "$WEB_DIR" ]; then
    echo "❌ Error: '$WEB_DIR' does not exist."
    exit 1
fi

cd "$WEB_DIR"

# Verify Node.js and npm availability
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed or not in PATH."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed or not in PATH."; exit 1; }

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

PORT="${1:-${PORT:-5173}}"
echo "🚀 Starting WonderSnap Web Server on http://localhost:${PORT} ..."
echo "🌐 Open http://localhost:${PORT} in your web browser (Chrome or Edge recommended for WebGL2 and camera access)."
echo "💡 Press Ctrl+C to terminate."

exec node server.mjs "$PORT"
