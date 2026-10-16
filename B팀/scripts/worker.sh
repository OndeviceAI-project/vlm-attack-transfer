#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${1:?}"; run_name="${2:?}"; image_set="${3:?}"; reuse="${4:-}"
[[ "$protocol" == common_v1 || "$protocol" == extended_v1 ]]
for value in "$run_name" "$image_set"; do [[ "$value" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$ ]]; done
[[ -z "$reuse" || "$reuse" == --measure-only ]]
export ONDEVICE_AI_ROOT="${ONDEVICE_AI_ROOT:-$HOME/ondevice-ai}"
if [[ -f "$ONDEVICE_AI_ROOT/activate.sh" ]]; then source "$ONDEVICE_AI_ROOT/activate.sh"; fi
export PYTHONPATH="$base/src${PYTHONPATH:+:$PYTHONPATH}" BTEAM_RUN_NAME="$run_name" BTEAM_IMAGE_SET="$image_set"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$base"; mkdir -p "results/$run_name"
exec >> >(tee -a "results/$run_name/pipeline.log") 2>&1
exec 9>"$ONDEVICE_AI_ROOT/.qwen-dev.lock"
flock -n 9 || { echo 'GPU lock held by another experiment'; exit 73; }
trap 'rc=$?; printf "%s\n" "$rc" > "results/$run_name/exit.code"; flock -u 9' EXIT
extra=(); if [[ -n "$reuse" ]]; then extra+=("$reuse"); fi
python -u src/pipeline.py "$protocol" --run "${extra[@]}"
