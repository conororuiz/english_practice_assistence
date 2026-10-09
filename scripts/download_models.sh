#!/usr/bin/env bash
# Atajo para macOS/Linux; la descarga real (multiplataforma) está en download_models.py.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python scripts/download_models.py
