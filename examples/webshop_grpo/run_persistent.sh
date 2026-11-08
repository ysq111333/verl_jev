#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/aiscuser/shiqiyan/verl-upstream-latest}
WS=${WS:-/home/aiscuser/shiqiyan}
MODE=${1:-smoke}
shift || true
DIR="$WS/runs/webshop_grpo/launch-upstream-b17218ba"
mkdir -p "$DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$DIR/$MODE-$STAMP.log"
STATUS="$DIR/$MODE-$STAMP.status"
printf 'running mode=%s log=%s\n' "$MODE" "$LOG" >"$STATUS"
cd "$ROOT"
set +e
bash "$ROOT/examples/webshop_grpo/run_training.sh" "$MODE" "$@" 2>&1 | tee "$LOG"
RESULT=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s finished=%s\n' "$RESULT" "$(date -u +%FT%TZ)" >>"$STATUS"
exit "$RESULT"
