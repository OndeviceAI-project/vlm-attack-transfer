#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${1:?}"; backend="${2:?}"; repeat="${3:?}"
[[ "$protocol" == common_v1 || "$protocol" == extended_v1 ]]
[[ "$backend" == rtn || "$backend" == bnb || "$backend" == both ]]
[[ "$repeat" =~ ^r[0-9]{2,}$ ]]
export ONDEVICE_AI_ROOT="${ONDEVICE_AI_ROOT:-$HOME/ondevice-ai}"
if [[ -f "$ONDEVICE_AI_ROOT/activate.sh" ]]; then source "$ONDEVICE_AI_ROOT/activate.sh"; fi
export PYTHONUTF8=1
cd "$base"
suite="components-$protocol-$backend-$repeat"
mkdir -p "results/$suite"
exec >> >(tee -a "results/$suite/pipeline.log") 2>&1
exec 9>"$ONDEVICE_AI_ROOT/.qwen-dev.lock"
flock -n 9 || { echo 'GPU lock held by another experiment'; exit 73; }
trap 'rc=$?; printf "%s\n" "$rc" > "results/$suite/exit.code"; flock -u 9' EXIT
python -u src/components.py "$protocol" --run --backend "$backend" --run-number "$repeat"
