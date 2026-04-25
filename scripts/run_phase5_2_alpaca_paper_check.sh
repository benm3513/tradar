#!/usr/bin/env bash
set -euo pipefail

# Controlled Phase 5.2 Alpaca paper run.
# Run from repo root:
#
#   bash scripts/run_phase5_2_alpaca_paper_check.sh
#
# Stop with Ctrl+C after 3-10 minutes, then run the report command printed below.

export PYTHONPATH=.

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/phase5_2_alpaca_paper_${STAMP}.log"

echo "Writing log to: $LOG_FILE"
echo "Starting Tradar. Stop with Ctrl+C after 3-10 minutes."
echo

./.venv/bin/python run.py 2>&1 | tee "$LOG_FILE"

echo
echo "Run stopped."
echo "Now generate report:"
echo "  PYTHONPATH=. ./.venv/bin/python scripts/phase5_2_live_paper_report.py --db tradarbot.db --since-minutes 60"
