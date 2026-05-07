#!/usr/bin/env bash
set -euo pipefail
systemctl status tradar --no-pager
journalctl -u tradar -n 100 --no-pager
