#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${1:-}"; run_name="${2:-}"; image_set="${3:-}"; action="${4:---check}"; reuse="${5:-}"
[[ "$protocol" == common_v1 || "$protocol" == extended_v1 ]] || { echo 'Use common_v1 or extended_v1'; exit 2; }
for value in "$run_name" "$image_set"; do
 [[ "$value" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$ ]] || { echo 'Explicit safe run/image-set names required'; exit 2; }
done
[[ "$action" == --check || "$action" == --start ]] || { echo 'Use --check or --start'; exit 2; }
[[ -z "$reuse" || "$reuse" == --measure-only ]] || { echo 'Optional last argument: --measure-only'; exit 2; }
export ONDEVICE_AI_ROOT="${ONDEVICE_AI_ROOT:-$HOME/ondevice-ai}"
if [[ -f "$ONDEVICE_AI_ROOT/activate.sh" ]]; then source "$ONDEVICE_AI_ROOT/activate.sh"; fi
export PYTHONPATH="$base/src${PYTHONPATH:+:$PYTHONPATH}"
export BTEAM_RUN_NAME="$run_name" BTEAM_IMAGE_SET="$image_set"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$base"
extra=(); if [[ -n "$reuse" ]]; then extra+=("$reuse"); fi
if [[ "$action" == --check ]]; then python src/pipeline.py "$protocol" --check "${extra[@]}"; exit; fi
command -v tmux >/dev/null; command -v flock >/dev/null
session="bteam-$run_name"
if tmux has-session -t "$session" 2>/dev/null; then echo "Already exists: $session"; exit; fi
printf -v task 'bash %q %q %q %q %q; exec bash' "$base/scripts/worker.sh" "$protocol" "$run_name" "$image_set" "$reuse"
tmux new-session -d -s "$session" -c "$base" "$task"
printf 'Started: %s\n' "$session"
