# Tradar Phase 5.5 Deployment

Phase 5.5 packages Tradar for continuous operation on a VPS or Docker host while keeping paper/dry-run defaults.

## Local Docker

```bash
cp deploy/env.example .env
mkdir -p data artifacts
docker compose up --build
```

Logs:

```bash
docker logs -f tradar
```

Healthcheck:

```bash
docker exec tradar python deploy/healthcheck.py --config config/tradar.yaml
```

## Native VPS install

Copy or clone the repo into `/opt/tradar`, then run:

```bash
sudo bash /opt/tradar/deploy/bootstrap.sh
sudo nano /etc/tradar/tradar.env
sudo systemctl enable tradar
sudo systemctl start tradar
sudo journalctl -u tradar -f
```

## Environment variables

Use `deploy/env.example` as the template. Do not commit real `.env` files or `/etc/tradar/tradar.env`.

Important variables:

- `TRADAR_PROFILE=paper` by default.
- `TRADAR_DB_PATH=/data/tradarbot.db` for Docker or `/opt/tradar/data/tradarbot.db` for native service.
- `TRADAR_ARTIFACT_DIR=/app/artifacts/models/v50_sig_importance` in Docker.
- Provider API keys are read from environment variables only.

## Model artifacts

The Docker image ignores `artifacts/`. Mount artifacts read-only:

```yaml
./artifacts:/app/artifacts:ro
```

If `ml_live.mode` or `ml_live.predictor_mode` is `artifacts`, healthcheck expects:

- `model_6h.joblib`
- `model_24h.joblib`
- `model_72h.joblib`

## SQLite data

Use a persistent mounted directory. Do not store the production DB only inside the container filesystem.

Docker default:

```text
./data:/data
TRADAR_DB_PATH=/data/tradarbot.db
```

## Safety notes

- Default profile is `paper`.
- Real live mode requires `TRADAR_CONFIRM_LIVE=true`.
- Do not commit secrets, `.env`, `tradarbot.db`, `*.db-wal`, or `*.db-shm`.
- Keep `flatten_on_shutdown` enabled unless deliberately disabling it in config.

## Rollback

```bash
sudo systemctl stop tradar
git checkout <known-good-commit>
/opt/tradar/.venv/bin/pip install -r /opt/tradar/requirements.txt
sudo systemctl start tradar
```
