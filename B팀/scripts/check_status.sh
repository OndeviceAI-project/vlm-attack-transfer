#!/usr/bin/env bash
set -Eeuo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python - "$base" "${1:?RUN_NAME required}" <<'PYTHON'
import json,re,sys
from pathlib import Path
name=sys.argv[2]
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}',name):raise SystemExit('Invalid run name')
p=Path(sys.argv[1])/'results'/name/'status.json'
print(p.read_text() if p.exists() else json.dumps({'state':'not_started'}))
PYTHON
