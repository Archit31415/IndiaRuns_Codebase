#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "REDROB HACKATHON: TEAM STARTING ONLINE RANKER"

# Ensure we are executing from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Set Python path so the src/ module can be imported properly
export PYTHONPATH="$SCRIPT_DIR"

# Execute the main orchestrator script
python3 src/online_ranker/main.py

echo "====================================================="
echo "✅ PIPELINE COMPLETE: Submission file generated."
echo "====================================================="