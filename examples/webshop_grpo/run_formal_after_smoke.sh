#!/usr/bin/env bash
# Run independently in tmux; fail closed unless the smoke and checkpoint completed.
set -euo pipefail
ROOT=${ROOT:-/home/aiscuser/shiqiyan/verl-upstream-latest}
WS=${WS:-/home/aiscuser/shiqiyan}
SMOKE_STATUS=${1:?Pass the exact active smoke status file}
while [[ $(tmux display-message -p -t webshop-grpo-upstream '#{pane_dead}') != 1 ]]; do
 sleep 5
done
if ! grep -q '^exit_code=0 ' "$SMOKE_STATUS"; then
 printf 'Formal run not launched: smoke failed; inspect %s\n' "$SMOKE_STATUS" >&2
 exit 1
fi
"$ROOT/.venv/bin/python" -I "$ROOT/examples/webshop_grpo/checkpoint_ready.py" \
 "$WS/checkpoints/webshop_grpo/upstream_webshop_qwen35_native_gdn_smoke_v1"
curl --fail --silent http://127.0.0.1:8093/health
printf '\nSmoke passed; launching full 150-update run from original weights.\n'
exec bash "$ROOT/examples/webshop_grpo/run_persistent.sh" train
