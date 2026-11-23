"""4·5·5-2·7단계 답 생성(README 4·5·5-2·7단계). 두 팀이 같은 코드를 자기 실험 기계에서 돌린다.

실행: python 공통/run_answers.py --team A --step accuracy          (B팀은 --team B)
      --step 은 vision(4단계) · language(5단계) · both(5-2단계) · accuracy(7단계) 중 하나
      python 공통/run_answers.py --team A --step vision --limit 2 --cases ref,rtn-V-4
                                   짧은 시험. 사진 2장, 경우 2개만 돌리고 결과 폴더에는 쓰지 않는다
결과: <팀>/results/<step>.csv, <step>.json      모든 경우가 끝났을 때만 생긴다

하는 일
1. 시작 조건을 확인한다. 하나라도 맞지 않으면 모델을 올리지 않고 멈춘다.
   - 실행 환경이 그 팀 check_ref.json과 같고, 코드에 커밋하지 않은 수정이 없다(README 5.1)
   - 그 팀 check_cases.json의 passed가 true다(1-3단계)
   - 넣을 사진의 SHA-256이 기록과 같다. 정상·노이즈 PNG는 input-manifest.json, 공격 PNG는 adv/SHA256SUMS
2. 경우(settings 목록 순서)마다 모델을 새로 올려, 사진(순서대로)마다 clean·noise·adv 순으로 답을 하나씩 만든다.
3. 4·5·5-2단계는 첫 경우인 ref가 끝나면 3단계 기록과 맞는지 본다. 공격 사진 답이 모두 512에 닿고, 정상 사진 답은
   하나도 닿지 않으며, 정상 사진 답의 토큰 수가 adv/attempts.csv의 ref_clean_n_tokens와 같아야 한다. 다르면 멈춘다.
4. 경우 하나가 끝날 때마다 <팀>/data/runs/<step>/ 에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는
   건너뛴다). 환경·코드·입력이 처음과 다르면 이어서 하지 않고 멈춘다. 처음부터 다시 하려면 --restart 를 붙인다.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import cases
import check_cases
import settings

ROOT = check_cases.ROOT
# 단계 이름 → (settings의 경우 목록 이름, 공격 사진 세트를 쓰는가)
STEPS = {"vision": ("CASES_VISION", True), "language": ("CASES_LANGUAGE", True),
         "both": ("CASES_BOTH", True), "accuracy": ("CASES_ALL", False)}
COLUMNS = ["model", "case", "image_id", "input", "n_tokens", "ended_eos", "hit_cap", "latency_s", "text"]
csv.field_size_limit(10**9)      # 512토큰짜리 답이 CSV 기본 한도를 넘을 수 있다


def stop(message):
    sys.exit(f"멈춤: {message}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, what):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        stop(f"{what}을(를) 읽지 못했다({type(err).__name__}) → 앞 단계가 끝났는지 확인한다")


def read_csv(path, what):
    try:
        with open(path, newline="", encoding="utf-8") as fp:
            return list(csv.DictReader(fp))
    except OSError as err:
        stop(f"{what}을(를) 읽지 못했다({type(err).__name__})")


def photo_plan(rt, with_adv, limit):
    """넣을 사진의 순서와 파일. [{"image_id", "paths": {입력: 경로}}], 입력 순서, split 이름."""
    if with_adv:
        rows = sorted(read_csv(ROOT / "공통" / "attack_set.csv", "attack_set.csv"), key=lambda r: int(r["set_order"]))
        ids, split, kinds = [f"{int(r['coco_id']):012d}" for r in rows], "cand", ("clean", "noise", "adv")
    else:
        ids, split, kinds = [coco_id for _, coco_id in rt.image_rows("acc")], "acc", ("clean", "noise")
    plan = []
    for coco_id in ids[:limit]:
        paths = {kind: rt.png_path(split, kind, coco_id) for kind in kinds if kind != "adv"}
        if with_adv:
            paths["adv"] = rt.TEAM / "adv" / f"{coco_id}.png"
        plan.append({"image_id": coco_id, "paths": paths})
    return plan, kinds, split


def verify_inputs(rt, plan, split, with_adv):
    """넣을 사진의 SHA-256을 기록과 맞대 보고, 같으면 그 해시들을 돌려준다. 하나라도 다르면 멈춘다."""
    manifest = read_json(rt.TEAM / "results" / "input-manifest.json", "input-manifest.json")
    recorded = {e["coco_id"]: e for e in manifest["images"][split]}
    sums = {}
    if with_adv:
        try:
            lines = (rt.TEAM / "adv" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        except OSError as err:
            stop(f"adv/SHA256SUMS를 읽지 못했다({type(err).__name__})")
        sums = {line.split()[1][:12]: line.split()[0] for line in lines if line.strip()}
    hashes, bad = {}, []
    for photo in plan:
        coco_id = photo["image_id"]
        for kind, path in photo["paths"].items():
            want = sums.get(coco_id) if kind == "adv" else recorded.get(coco_id, {}).get(f"{kind}_sha256")
            got = sha256(path.read_bytes()) if path.exists() else None
            hashes.setdefault(f"{kind}_sha256", {})[coco_id] = got
            if got is None or got != want:
                bad.append(f"{coco_id}:{kind}")
    if bad:
        stop(f"사진 {len(bad)}개가 없거나 기록과 다르다(예: {bad[:3]}) → 2·3단계에서 만든 파일을 그대로 쓴다")
    return hashes


def preconditions(rt, env, trial):
    """실행 환경, 코드, 1-3단계 통과 여부. 시험(--limit)에서는 알리기만 하고 계속한다."""
    try:
        text = (rt.TEAM / "results" / "check_cases.json").read_text(encoding="utf-8")
        done = json.loads(text).get("passed") is True
    except (OSError, ValueError):
        done = False              # 파일이 없거나 읽을 수 없으면 1-3단계가 끝나지 않은 것으로 본다
    checks = {"environment": check_cases.check_environment(rt, env), "code": check_cases.check_code(env),
              "check_cases": {"pass": done}}
    failed = {name: check for name, check in checks.items() if not check["pass"]}
    if failed and not trial:
        stop(f"시작 조건이 맞지 않는다: {failed}")
    for name, check in failed.items():
        print(f"알림(시험이라 계속한다): {name} 조건이 맞지 않는다. {check}")
    return checks


def run_case(rt, processor, name, reference, plan, kinds):
    """경우 하나의 모델을 올려 모든 사진의 답을 만든다. (경우 기록, CSV 줄들)"""
    started = time.time()
    model, info = cases.build(rt, name, reference)
    rows = []
    try:
        for photo in plan:
            for kind in kinds:
                ans = rt.answer(model, processor, rt.read_png(photo["paths"][kind]))
                rows.append([rt.MODEL_KEY, name, photo["image_id"], kind] + [ans[k] for k in COLUMNS[4:]])
    finally:
        model = None
        cases.free()
    info.update({"rows": len(rows), "seconds": round(time.time() - started, 1),
                 "clean_hit_cap": sum(r[6] for r in rows if r[3] == "clean")})
    return info, rows


def ref_check(rt, rows, plan):
    """ref의 답이 3단계 기록과 맞는가(같은 코드·환경이면 그대로 재현된다)."""
    recorded = {}
    for row in read_csv(rt.TEAM / "adv" / "attempts.csv", "adv/attempts.csv"):
        recorded.setdefault(row["coco_id"], int(row["ref_clean_n_tokens"]))
    answers = {(r[2], r[3]): r for r in rows}
    ids = [photo["image_id"] for photo in plan]
    adv_cap = sum(answers[(i, "adv")][6] for i in ids)
    clean_cap = sum(answers[(i, "clean")][6] for i in ids)
    differs = [i for i in ids if answers[(i, "clean")][4] != recorded.get(i)]
    return {"pass": adv_cap == len(ids) and clean_cap == 0 and not differs, "photos": len(ids),
            "adv_hit_cap": adv_cap, "clean_hit_cap": clean_cap, "clean_tokens_differ": differs}


def append_rows(path, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)
        fp.flush()
        os.fsync(fp.fileno())


def save_state(path, state):
    """기록을 통째로 새로 쓴다. 쓰다가 끊겨도 옛 파일이 남도록 다른 이름으로 쓴 뒤 바꾼다."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def load_state(work, identity, restart):
    """이어 할 기록을 읽는다. 없거나 --restart면 새로 시작한다. 환경·코드·입력이 다르면 멈춘다."""
    state_path, rows_path = work / "state.json", work / "rows.csv"
    if restart or not state_path.exists():
        for stale in (rows_path, state_path, work / "ref_check_failed.json"):   # 이전 실행의 작업 파일을 지운다
            if stale.exists():
                stale.unlink()
        return {"identity": identity, "cases": {}, "runs": [], "ref_check": None}
    state = read_json(state_path, "이어 할 기록(state.json)")
    if state["identity"] != identity:
        changed = [k for k in identity if state["identity"].get(k) != identity[k]]
        stop(f"처음 돌릴 때와 다른 것이 있어 이어서 하지 않는다: {changed}. 처음부터 다시 하려면 --restart 를 붙인다")
    done = sum(c["rows"] for c in state["cases"].values())
    try:
        with open(rows_path, newline="", encoding="utf-8") as fp:
            data = list(csv.reader(fp))
    except OSError:
        data = [COLUMNS]          # 작업 파일이 없다: 끝난 경우가 없을 때만 괜찮다(아래에서 줄 수로 가른다)
    if len(data) - 1 < done:
        stop("작업 파일(rows.csv)의 줄이 기록보다 적다 → --restart 로 처음부터 다시 한다")
    if len(data) - 1 > done:          # 저장 직후 끊겨 기록에 없는 줄이 남은 경우: 기록된 줄까지만 남긴다
        with open(rows_path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerows(data[:done + 1])
    return state


def run_all(rt, args, names, plan, kinds, state, work):
    """남은 경우를 차례로 돌린다. 경우 하나가 끝날 때마다 줄과 기록을 저장한다."""
    todo = [name for name in names if name not in state["cases"]]
    if not todo:
        return
    first_row = sum(c["rows"] for c in state["cases"].values()) + 1
    run = {"run": len(state["runs"]) + 1, "first_row": first_row, "rows_added": 0,
           "code_commit": state["identity"]["environment"]["code_commit"]}
    state["runs"].append(run)
    processor, reference = rt.load_processor(), cases.reference_weights(rt)
    for count, name in enumerate(todo, 1):
        info, rows = run_case(rt, processor, name, reference, plan, kinds)
        if name == "ref" and "adv" in kinds:
            state["ref_check"] = ref_check(rt, rows, plan)
            if not state["ref_check"]["pass"] and args.limit is None:
                save_state(work / "ref_check_failed.json", state["ref_check"])
                stop(f"ref의 답이 3단계 기록과 다르다 → 환경이 달라졌는지 확인한다. {state['ref_check']}")
        append_rows(work / "rows.csv", rows)
        state["cases"][name] = info
        run["rows_added"] += len(rows)
        save_state(work / "state.json", state)
        print(f"[{count}/{len(todo)}] {name:10s} {len(rows)}줄, {info['seconds'] / 60:.1f}분 | "
              f"정상 사진 답이 512인 것 {info['clean_hit_cap']}개 | 가중치 변화 {info['weight_change_pct']}")


def finalize(rt, args, state, work, checks, inputs):
    """모든 경우가 끝났으면 결과 파일을 쓴다. 시험이면 작업 폴더에, 아니면 <팀>/results/ 에 쓴다."""
    data = (work / "rows.csv").read_bytes()
    meta = {"environment": state["identity"]["environment"], "step": args.step, "cases": state["cases"],
            "inputs": inputs, "ref_check": state["ref_check"], "checks": checks,
            "csv": {"rows": sum(c["rows"] for c in state["cases"].values()), "sha256": sha256(data)},
            "runs": state["runs"],
            "note": "README 4·5·5-2·7단계. csv 한 줄 = 답 하나. runs의 first_row는 그 실행이 처음 쓴 줄 번호(머리줄 제외)"}
    out = work if args.limit is not None else rt.TEAM / "results"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.step}.csv").write_bytes(data)
        (out / f"{args.step}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except OSError as err:
        stop(f"결과 파일을 쓰지 못했다({type(err).__name__})")
    print(f"저장: {out.relative_to(ROOT).as_posix()}/{args.step}.csv, {args.step}.json | {meta['csv']['rows']}줄")


def parse_args():
    parser = argparse.ArgumentParser(description="4·5·5-2·7단계 답 생성")
    parser.add_argument("--team", required=True, choices=sorted(check_cases.TEAMS))
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--limit", type=int, help="시험: 앞에서 이 수만큼의 사진만 돌린다. 결과 폴더에 쓰지 않는다")
    parser.add_argument("--cases", help="시험(--limit과 함께): 쉼표로 나눈 경우 이름만 돌린다")
    parser.add_argument("--restart", action="store_true", help="이어 하지 않고 처음부터 다시 한다")
    args = parser.parse_args()
    if args.cases and args.limit is None:
        parser.error("--cases 는 시험(--limit)에서만 쓴다")
    return args


def main():
    args = parse_args()
    rt = check_cases.load_runtime(args.team)
    list_name, with_adv = STEPS[args.step]
    names = list(getattr(settings, list_name))
    if args.cases:
        names = [name for name in names if name in args.cases.split(",")]
    plan, kinds, split = photo_plan(rt, with_adv, args.limit)
    rt.setup()
    env = rt.environment()
    checks = preconditions(rt, env, trial=args.limit is not None)
    inputs = verify_inputs(rt, plan, split, with_adv)
    identity = {"environment": env, "step": args.step, "cases": names, "images": [p["image_id"] for p in plan],
                "inputs": inputs, "prompt": settings.PROMPT, "gen_kwargs": settings.GEN_KWARGS}
    identity = json.loads(json.dumps(identity))      # 저장했다 읽은 기록과 같은 꼴로 맞춘다(비교용)
    work = rt.DATA / ("trial" if args.limit is not None else "runs") / args.step
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work, identity, args.restart)
    run_all(rt, args, names, plan, kinds, state, work)
    finalize(rt, args, state, work, checks, inputs)


if __name__ == "__main__":
    main()
