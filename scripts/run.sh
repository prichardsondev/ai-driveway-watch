#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app --host "${WEB_HOST:-127.0.0.1}" --port "${WEB_PORT:-8000}"

