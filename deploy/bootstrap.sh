#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/tradar}"
ENV_DIR="${ENV_DIR:-/etc/tradar}"
LOG_DIR="${LOG_DIR:-/var/log/tradar}"
SERVICE_SRC="${APP_DIR}/deploy/tradar.service"
SERVICE_DST="/etc/systemd/system/tradar.service"
USER_NAME="${TRADAR_USER:-tradar}"
GROUP_NAME="${TRADAR_GROUP:-tradar}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash deploy/bootstrap.sh"
  exit 1
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip git sqlite3 build-essential

if ! id -u "${USER_NAME}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${USER_NAME}"
fi

mkdir -p "${APP_DIR}" "${ENV_DIR}" "${LOG_DIR}" "${APP_DIR}/data"
chown -R "${USER_NAME}:${GROUP_NAME}" "${APP_DIR}" "${LOG_DIR}" || true

if [[ ! -f "${APP_DIR}/run.py" ]]; then
  echo "Project files are not present in ${APP_DIR}."
  echo "Copy or git clone the Tradar repo into ${APP_DIR}, then rerun this script."
  exit 1
fi

python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

if [[ ! -f "${ENV_DIR}/tradar.env" ]]; then
  cp "${APP_DIR}/deploy/env.example" "${ENV_DIR}/tradar.env"
  chmod 600 "${ENV_DIR}/tradar.env"
  echo "Created ${ENV_DIR}/tradar.env from deploy/env.example. Edit it before live use."
else
  echo "Preserving existing ${ENV_DIR}/tradar.env"
fi

cp "${SERVICE_SRC}" "${SERVICE_DST}"
systemctl daemon-reload

set +e
"${APP_DIR}/.venv/bin/python" "${APP_DIR}/deploy/healthcheck.py" --config "${APP_DIR}/config/tradar.yaml"
HC_STATUS=$?
set -e

cat <<EOF

Bootstrap complete.

Next commands:
  sudo nano ${ENV_DIR}/tradar.env
  sudo systemctl daemon-reload
  sudo systemctl enable tradar
  sudo systemctl start tradar
  sudo journalctl -u tradar -f

Healthcheck exit code: ${HC_STATUS}
EOF

exit "${HC_STATUS}"
