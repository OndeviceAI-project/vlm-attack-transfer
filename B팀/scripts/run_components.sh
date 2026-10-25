#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${1:-common_v1}"; action="${2:---plan}"; backend="${3:-both}"; repeat="${4:-r01}"
[[ "$protocol" == common_v1 || "$protocol" == extended_v1 ]] || { echo 'Use common_v1 or extended_v1'; exit 2; }
[[ "$action" == --plan || "$action" == --check || "$action" == --start ]] || { echo 'Use --plan, --check or --start'; exit 2; }
[[ "$backend" == rtn || "$backend" == bnb || "$backend" == both ]] || { echo 'Use rtn, bnb or both'; exit 2; }
[[ "$repeat" =~ ^r[0-9]{2,}$ ]] || { echo 'Use r01, r02, ...'; exit 2; }
export ONDEVICE_AI_ROOT="${ONDEVICE_AI_ROOT:-$HOME/ondevice-ai}"
if [[ -f "$ONDEVICE_AI_ROOT/activate.sh" ]]; then source "$ONDEVICE_AI_ROOT/activate.sh"; fi
export PYTHONUTF8=1
cd "$base"
if [[ "$action" != --start ]]; then
 python src/components.py "$protocol" "$action" --backend "$backend" --run-number "$repeat"
 exit
fi
python src/components.py "$protocol" --check --backend "$backend" --run-number "$repeat"
command -v tmux >/dev/null; command -v flock >/dev/null
session="bteam-components-$protocol-$backend-$repeat"
if tmux has-session -t "$session" 2>/dev/null; then echo "Already exists: $session"; exit 2; fi
printf -v task 'bash %q %q %q %q' "$base/scripts/component_worker.sh" "$protocol" "$backend" "$repeat"
tmux new-session -d -s "$session" -c "$base" "$task"
printf 'Started: %s\n' "$session"
