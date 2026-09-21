#!/usr/bin/env bash
# Deploy the V14 ABSA service (api + web) on a shared server, on its own port.
# Safe to run from any checkout location -- container/volume/network names are
# pinned via COMPOSE_PROJECT_NAME so this never collides with other stacks
# (e.g. crawlerService) on the same host.
#
# Usage:
#   ./deploy.sh [PORT]
#
# Examples:
#   ./deploy.sh          # uses V14_ABSA_PORT from .env (or 8080 default)
#   ./deploy.sh 9090      # exposes the service on port 9090 instead

git checkout absa
git pull origin absa

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export COMPOSE_PROJECT_NAME="v14-absa"
COMPOSE="docker compose --env-file .env -f docker-compose.v14.yml"

if [[ ! -f .env ]]; then
  echo "No .env found, seeding it from .env.v14.example."
  cp .env.v14.example .env
fi

if [[ $# -ge 1 ]]; then
  PORT="$1"
  if grep -q '^V14_ABSA_PORT=' .env; then
    sed -i.bak "s/^V14_ABSA_PORT=.*/V14_ABSA_PORT=${PORT}/" .env && rm -f .env.bak
  else
    echo "V14_ABSA_PORT=${PORT}" >> .env
  fi
fi

PORT="$(grep '^V14_ABSA_PORT=' .env | cut -d= -f2)"
PORT="${PORT:-8080}"

echo "==> Target port: ${PORT}"
if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${PORT} )" | grep -q LISTEN; then
  echo "ERROR: port ${PORT} is already in use on this host (check what crawlerService or others are bound to)." >&2
  exit 1
fi

echo "==> Validating docker-compose.v14.yml"
$COMPOSE config >/dev/null

echo "==> Building images"
$COMPOSE build

echo "==> Starting containers"
$COMPOSE up -d

echo "==> Waiting for the api service to report healthy (up to 3 minutes)"
for _ in $(seq 1 36); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "${COMPOSE_PROJECT_NAME}-api-1" 2>/dev/null || echo "starting")"
  [[ "$status" == "healthy" ]] && break
  sleep 5
done

$COMPOSE ps

if [[ "${status:-}" != "healthy" ]]; then
  echo "WARNING: api did not report healthy in time. Check logs with:"
  echo "  $COMPOSE logs --tail=200 api"
  exit 1
fi

echo "==> Smoke test"
curl -sf "http://localhost:${PORT}/api/health" && echo
curl -sf -X POST "http://localhost:${PORT}/api/inference/single" \
  -H 'Content-Type: application/json' \
  -d '{"review":"Makanannya enak tetapi pelayanannya lambat.","engine_version":"v14","profile":"maps_high_recall","confidence_threshold":0.1}' \
  | head -c 500 && echo

echo
echo "==> Deployed. Service is reachable at http://<server-ip>:${PORT}/api/..."
echo "    Operate it with: $COMPOSE [ps|logs|restart|down]"
