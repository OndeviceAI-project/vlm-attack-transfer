"""1-3단계 경우 점검(README 1-3단계). 두 팀이 같은 코드를 자기 실험 기계에서 돌린다.

실행: python 공통/check_cases.py --team A        (B팀은 --team B)
결과: <팀>/results/check_cases.json
      하나라도 통과하지 못하면 결과를 적은 뒤 오류로 끝난다. 그때는 4~7단계로 가지 않는다.

먼저 되어 있어야 하는 것
- 2단계: dev 1번 정상 PNG와 <팀>/results/input-manifest.json
- main을 받은 그대로일 것. 공통·이미지생성·팀 code 폴더에 커밋하지 않은 수정이 있으면 결과의 code_commit에 '-dirty'가 붙는다

settings.CASES_ALL의 경우마다 모델을 새로 올려 네 가지를 점검한다.
1. 입력: dev 1번 정상 PNG의 SHA-256이 input-manifest.json과 같다.
2. census: 대상 부품의 Linear는 전부 바뀌고 나머지는 하나도 바뀌지 않는다. 부품별 Linear 수가 settings와 같다.
3. 가중치 변화: rtn은 settings.RTN_CHANGE_PCT와, rand는 같은 부품·비트의 rtn을 잰 값과 CHANGE_TOL 안에서 같다.
   bnb는 값을 재서 기록만 한다.
4. 생성: 경우마다 dev 1번 정상 사진의 답을 두 번 생성해 기록한다. ref·rtn·rand는 두 답이 같아야 한다.
   bnb는 두 답이 다르면 그 사실만 기록한다.
"""
import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import cases
import settings

ROOT = Path(__file__).resolve().parent.parent
TEAMS = {"A": "A팀", "B": "B팀"}
ANSWER_KEYS = ("n_tokens", "ended_eos", "hit_cap", "latency_s", "text")
SAME_KEYS = ("n_tokens", "ended_eos", "hit_cap", "text")


def load_runtime(team):
    """팀의 runtime.py를 불러온다."""
    code = ROOT / TEAMS[team] / "code"
    if not (code / "runtime.py").exists():
        sys.exit(f"멈춤: {code.name}/runtime.py가 없다")
    sys.path.insert(0, str(code))
    import runtime
    return runtime


def dev_photo(rt):
    """dev 1번 정상 PNG와, 그것이 2단계에서 만든 파일인지(input-manifest와 같은지)의 기록."""
    order, coco_id = rt.image_rows("dev")[0]
    path = rt.png_path("dev", "clean", coco_id)
    manifest_path = rt.TEAM / "results" / "input-manifest.json"
    if not path.exists() or not manifest_path.exists():
        sys.exit("멈춤: dev 1번 정상 PNG 또는 input-manifest.json이 없다 → 2단계를 먼저 한다")
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]["dev"][0]
    except (OSError, ValueError, KeyError, IndexError) as err:
        sys.exit(f"멈춤: input-manifest.json을 읽지 못했다 ({type(err).__name__})")
    actual = rt.sha256(path)
    record = {"split": "dev", "order": order, "coco_id": coco_id, "clean_sha256": actual,
              "matches_manifest": recorded.get("coco_id") == coco_id and recorded.get("clean_sha256") == actual}
    return path, record


def _short(err):
    """오류 글에서 절대 경로를 파일 이름만 남기고 지운다(README 2장: 절대 경로를 올리지 않는다)."""
    text = f"{type(err).__name__}: {err}"
    text = re.sub(r"(?:/[^\s/'\":]+){2,}", lambda m: ".../" + m.group(0).rsplit("/", 1)[-1], text)
    return text[:300]


def run_case(rt, processor, img, name, reference):
    """경우 하나를 만들고 dev 1번 사진의 답을 두 번 생성한다. 오류가 나면 내용을 기록하고 다음 경우로 간다."""
    record, model = {}, None
    try:
        model, info = cases.build(rt, name, reference)
        record.update(info)
        first, second = rt.answer(model, processor, img), rt.answer(model, processor, img)
        record["answer"] = {k: first[k] for k in ANSWER_KEYS}
        record["repeat_same"] = all(first[k] == second[k] for k in SAME_KEYS)
    except Exception as err:      # 어느 경우가 왜 안 되는지 빠짐없이 남기려고 넓게 잡는다. 결과와 화면에 모두 남긴다
        record["error"] = _short(err)
        traceback.print_exc()
    finally:
        model = None
        cases.free()
    return record


def check_census(records):
    failed = [name for name, r in records.items() if "census" not in r]
    return {"pass": not failed, "failed": failed}


def check_weight_change(model_key, records):
    table, tol = settings.RTN_CHANGE_PCT[model_key], settings.CHANGE_TOL
    rows = []
    for name, record in records.items():
        method, targets, bits = cases.parse(name)
        for part in targets:
            measured = record.get("weight_change_pct", {}).get(part)
            expected, basis = None, "기록만"
            if method == "rtn":
                expected, basis = table[part][bits], "settings.RTN_CHANGE_PCT"
            elif method == "rand":
                twin = f"rtn-{targets}-{bits}"
                expected, basis = records.get(twin, {}).get("weight_change_pct", {}).get(part), f"{twin}의 잰 값"
            if measured is None or measured <= 0:
                ok = False
            elif method == "bnb":
                ok = True
            else:
                ok = expected is not None and abs(measured - expected) <= tol + 1e-9
            rows.append({"case": name, "part": part, "measured": measured, "expected": expected,
                         "basis": basis, "pass": ok})
    return {"pass": all(r["pass"] for r in rows), "tolerance": tol, "rows": rows}


def check_generation(records):
    failed = [name for name, r in records.items() if r.get("answer", {}).get("n_tokens", 0) < 1]
    differs = [name for name, r in records.items() if r.get("repeat_same") is False]
    must_match = [name for name in differs if cases.parse(name)[0] != "bnb"]
    return {"pass": not failed and not must_match, "failed": failed, "repeat_differs": must_match,
            "bnb_repeat_differs": [name for name in differs if name not in must_match]}


def main():
    parser = argparse.ArgumentParser(description="1-3단계 경우 점검")
    parser.add_argument("--team", required=True, choices=sorted(TEAMS))
    args = parser.parse_args()
    rt = load_runtime(args.team)
    path, inputs = dev_photo(rt)

    rt.setup()
    processor, img = rt.load_processor(), rt.read_png(path)
    reference = cases.reference_weights(rt)
    records = {}
    for name in settings.CASES_ALL:
        record = records[name] = run_case(rt, processor, img, name, reference)
        tokens = record.get("answer", {}).get("n_tokens")
        print(f"{name:10s} 가중치 변화 {record.get('weight_change_pct')} | 답 {tokens}토큰 | "
              f"두 번 같음 {record.get('repeat_same')} | {record.get('error', '오류 없음')}")

    checks = {"input": {"pass": inputs["matches_manifest"]}, "census": check_census(records),
              "weight_change": check_weight_change(rt.MODEL_KEY, records), "generation": check_generation(records)}
    passed = all(c["pass"] for c in checks.values())
    meta = {"environment": rt.environment(), "inputs": inputs, "cases": records, "checks": checks, "passed": passed,
            "note": "README 1-3단계. checks의 pass가 모두 true면 통과. census는 부품별 [바뀐 Linear 수, 전체 Linear 수]"}
    out = rt.TEAM / "results" / "check_cases.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except OSError as err:
        sys.exit(f"멈춤: {out.name}을 쓰지 못했다 ({type(err).__name__})")
    for name, check in checks.items():
        print(f"{name}: {'통과' if check['pass'] else '통과하지 못함'}")
    print(f"저장: {TEAMS[args.team]}/results/check_cases.json | "
          f"{'모두 통과' if passed else '통과하지 못한 점검이 있다 → 4~7단계로 가지 않는다'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
