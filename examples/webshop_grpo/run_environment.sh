#!/usr/bin/env bash
set -euo pipefail
WS=${WS:-/home/aiscuser/shiqiyan}
ROOT=${ROOT:-$WS/verl-upstream-latest}
source "$WS/StartLux-Decision/experiments/webshop_choice/webshop_env.sh"
LOGDIR="$WS/runs/webshop_grpo/environment-upstream-b17218ba"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/service-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec "$WEBSHOP_PYTHON" -I "$ROOT/examples/webshop_grpo/webshop_env_service.py" \
 --manifest "$WS/StartLux-Decision/runs/webshop_choice/manifest-v1.json" \
 --eval-helpers "$WEBSHOP_EVAL" "${WEBSHOP_FULL_ARGS[@]}" \
 --index "$WS/webshop-runtime/index-full-v1/index" --goal-type human --seed 0 \
 --host 127.0.0.1 --port 8093 >"$LOG" 2>&1
