#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "$0")/.." && pwd)"
action="${1:---status}"
repeat="${2:-r01}"
[[ "$action" == --start || "$action" == --worker || "$action" == --status ]] || { echo 'Use --start or --status'; exit 2; }
[[ "$repeat" =~ ^r[0-9]{2,}$ ]] || { echo 'Use r01, r02, ...'; exit 2; }
export ONDEVICE_AI_ROOT="${ONDEVICE_AI_ROOT:-$HOME/ondevice-ai}"
if [[ -f "$ONDEVICE_AI_ROOT/activate.sh" ]]; then source "$ONDEVICE_AI_ROOT/activate.sh"; fi
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ONDEVICE_AI_REPO="$(cd -- "$base/.." && pwd)"
cd "$base"
suite="components-common-then-extended-both-$repeat"
session="bteam-components-512-then-5000-both-$repeat"
directory="$base/results/$suite"
status="$directory/status.json"

if [[ "$action" == --status ]]; then
 python - "$base" "$status" "$repeat" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]); path=Path(sys.argv[2]); repeat=sys.argv[3]
sequence=json.loads(path.read_text()) if path.exists() else {'state':'not_started'}
experiments=[]
for protocol,cap in [('common',512),('extended',5000)]:
 for component in ['connector','language','vcl']:
  label=f'qwen2vl_{component}_{protocol}_components_v1_both_{repeat}'
  current=base/'results'/label/'status.json'
  data=json.loads(current.read_text()) if current.exists() else {}
  experiments.append(dict(experiment_id=label,max_new_tokens=cap,
   state=data.get('state','not_started'),stage=data.get('stage'),
   measurement_rows=data.get('measurement_rows',0),expected_rows=720,
   current_case=data.get('current_case'),current_image=data.get('current_image'),
   updated_at=data.get('updated_at'),error=data.get('error')))
print(json.dumps(dict(sequence=sequence,experiments=experiments,
 measured_rows=sum(e['measurement_rows'] for e in experiments),
 expected_rows=4320),ensure_ascii=False,indent=2))
PY
 exit
fi

if [[ "$action" == --start ]]; then
 command -v tmux >/dev/null
 command -v flock >/dev/null
 if tmux has-session -t "$session" 2>/dev/null; then echo "Already exists: $session"; exit 2; fi
 python - "$ONDEVICE_AI_REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
repo=Path(sys.argv[1])
ready=json.loads((repo/'B\uD300/experiments/extended_v1_server_readiness.json').read_text())
assert ready['state']=='ready'
for name,digest in ready['code_sha256'].items():
 if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
  raise ValueError(f'Code differs from GPU-verified preparation: {name}')
PY
 for protocol in common_v1 extended_v1; do
  python src/components.py "$protocol" --check --backend both --run-number "$repeat"
 done
 printf -v task 'bash %q --worker %q' "$base/scripts/run_component_sequence.sh" "$repeat"
 tmux new-session -d -s "$session" -c "$base" "$task"
 tmux set-option -t "$session" remain-on-exit on
 printf 'Started: %s\nStatus: %s\n' "$session" "$status"
 exit
fi

mkdir -p "$directory"
exec >> >(tee -a "$directory/sequence.log") 2>&1
completed=0
current=preflight
locked=0
update_status() {
 python - "$status" "$1" "$current" "$completed" "$repeat" "$session" "$$" "${2:-}" <<'PY'
from datetime import datetime,timezone
import json,os,sys
from pathlib import Path
path=Path(sys.argv[1])
data=json.loads(path.read_text()) if path.exists() else {}
now=datetime.now(timezone.utc).isoformat()
data.update(state=sys.argv[2],current_protocol=sys.argv[3],
 completed_protocols=['common_v1','extended_v1'][:int(sys.argv[4])],
 protocols=['common_v1','extended_v1'],backend='both',run_number=sys.argv[5],
 tmux_session=sys.argv[6],pid=int(sys.argv[7]),expected_rows=4320,
 expected_rows_per_protocol=2160,updated_at=now)
data.setdefault('started_at',now)
if sys.argv[8]:
 data.update(exit_code=int(sys.argv[8]),finished_at=now)
temp=path.with_name(path.name+'.tmp')
temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
os.replace(temp,path)
PY
}
finish() {
 rc=$?
 trap - EXIT
 if ((rc==0 && completed==2)); then
  current=complete
  update_status complete "$rc"
 else
  update_status error "$rc"
 fi
 printf '%s\n' "$rc" > "$directory/exit.code"
 if ((locked)); then flock -u 9; fi
 printf 'SEQUENCE_EXIT code=%s completed_protocols=%s\n' "$rc" "$completed"
 exit "$rc"
}
trap finish EXIT
update_status starting
exec 9>"$ONDEVICE_AI_ROOT/.qwen-dev.lock"
flock -n 9 || { echo 'GPU lock held by another experiment'; exit 73; }
locked=1
for protocol in common_v1 extended_v1; do
 current="$protocol"
 update_status running
 phase="$base/results/components-$protocol-both-$repeat"
 mkdir -p "$phase"
 printf 'SEQUENCE_PROTOCOL_START %s\n' "$protocol"
 python -u src/components.py "$protocol" --run --backend both --run-number "$repeat" 2>&1 | tee -a "$phase/pipeline.log"
 completed=$((completed+1))
 update_status running
 printf 'SEQUENCE_PROTOCOL_COMPLETE %s\n' "$protocol"
done

