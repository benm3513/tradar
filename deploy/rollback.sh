#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
BACKUP_CONFIG=1
SERVICE_NAME="${TRADAR_SERVICE_NAME:-tradar}"
CONFIG_PATH="${TRADAR_CONFIG:-}"
ENV_FILE="${TRADAR_ENV_FILE:-/etc/tradar/tradar.env}"
APP_DIR="${APP_DIR:-/opt/tradar}"

log() { echo "[rollback] $*"; }
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[rollback][dry-run] $*"
  else
    "$@"
  fi
}
fail() {
  echo "ROLLBACK_FAILED reason=$*" >&2
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-config-backup) BACKUP_CONFIG=0 ;;
    *) fail "unknown_arg_${arg}" ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  if [[ -f "${APP_DIR}/config/tradar.live.yaml" ]]; then
    CONFIG_PATH="${APP_DIR}/config/tradar.live.yaml"
  elif [[ -f "${APP_DIR}/config/tradar.yaml" ]]; then
    CONFIG_PATH="${APP_DIR}/config/tradar.yaml"
  else
    CONFIG_PATH="config/tradar.yaml"
  fi
fi

log "starting rollback service=${SERVICE_NAME} config=${CONFIG_PATH} env_file=${ENV_FILE} dry_run=${DRY_RUN}"

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
    run systemctl stop "${SERVICE_NAME}" || true
  else
    log "systemd service ${SERVICE_NAME} not found; continuing"
  fi
fi

if [[ -f "${CONFIG_PATH}" && "${BACKUP_CONFIG}" == "1" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  run cp "${CONFIG_PATH}" "${CONFIG_PATH}.rollback_backup.${ts}"
fi

# Preserve DB/logs/artifacts. Only change runtime/live switches.
if [[ -f "${ENV_FILE}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "would update ${ENV_FILE}: TRADAR_PROFILE=paper, TRADAR_CONFIRM_LIVE=false, EXECUTION_MODE=paper"
  else
    touch "${ENV_FILE}"
    chmod 600 "${ENV_FILE}" || true

    set_kv() {
      local key="$1"
      local val="$2"
      if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=${val}|g" "${ENV_FILE}"
      else
        echo "${key}=${val}" >> "${ENV_FILE}"
      fi
    }

    set_kv "TRADAR_PROFILE" "paper"
    set_kv "TRADAR_CONFIRM_LIVE" "false"
    set_kv "TRADAR_LIVE_ACK" ""
    set_kv "EXECUTION_MODE" "paper"
    set_kv "EXECUTION_BROKER" "paper"
    set_kv "TRADAR_CONFIG" "${APP_DIR}/config/tradar.yaml"
  fi
else
  log "env file not found at ${ENV_FILE}; no env mutation performed"
fi

if [[ -f "${CONFIG_PATH}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "would set runtime.profile=paper and execution_live broker/mode paper in ${CONFIG_PATH}"
    else
      python3 - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML required for rollback config mutation: {exc}")

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text()) or {}
cfg.setdefault("runtime", {})["profile"] = "${TRADAR_PROFILE:-paper}"
cfg.setdefault("execution_live", {})["enabled"] = True
cfg["execution_live"]["broker"] = "paper"
cfg["execution_live"]["mode"] = "${EXECUTION_MODE:-paper}"
cfg.setdefault("ml_live", {})["mode"] = "shadow"
cfg.setdefault("rollout", {})["stage"] = 0
cfg.setdefault("live_startup", {})["require_live_confirm"] = True
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
    fi
  else
    log "python3 unavailable; config file not mutated"
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  run systemctl daemon-reload || true
fi

echo "ROLLBACK_COMPLETE"
