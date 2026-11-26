"""9-2단계: 압축 모델에서 만든 공격 사진을 여러 경우에 넣어 답을 기록한다(README 9단계). 두 팀이 같은 코드를 쓴다.

실행: python 공통/run_direct.py --team A                (B팀은 --team B)
      python 공통/run_direct.py --team A --limit 2 --cases ref,rtn-V-4
                                   짧은 시험. 사진 2장, 도착 경우 2개만 돌리고 결과 폴더에는 쓰지 않는다
결과: <팀>/results/direct.csv, direct.json      모든 도착 경우가 끝났을 때만 생긴다

먼저 되어 있어야 하는 것(9-1단계): <팀>/adv_direct/ 의 attempts.csv, attempts.json, SHA256SUMS, <출발 경우>/<coco_id>.png

하는 일
1. 시작 조건을 확인한다. 하나라도 맞지 않으면 모델을 올리지 않고 멈춘다.
   - 실행 환경이 그 팀 check_ref.json과 같고, 코드에 커밋하지 않은 수정이 없으며, check_cases.json의 passed가 true다
     (run_answers.py와 같은 확인)
   - 9-1단계 기록이 온전하다. attempts.csv가 attempts.json에 적힌 줄 수·SHA-256과 같고, 만든 때의 실행 환경이 지금과
     같으며, 출발 경우(settings.CASES_DIRECT)마다 사진 N_DIRECT장이 모두 끝까지 처리되어 있다
   - 넣을 PNG가 기록과 같다. success 시도의 PNG가 모두 있고 SHA-256이 attempts.csv, SHA256SUMS 둘 다와 같다.
     SHA256SUMS에 그 밖의 파일이 없다
2. 도착 경우(settings.CASES_DIRECT_TARGETS 순서)마다 모델을 새로 올려, 출발 경우(CASES_DIRECT 순서)마다 success가
   나온 사진(set_order 순서)을 넣어 답을 하나씩 만든다.
3. 도착 경우가 출발 경우이기도 하면, 그 경우에서 만든 사진의 답이 모두 512에 닿는지 본다. 9-1단계의 판정이 같은
   코드·환경에서 그대로 재현된다는 확인이다. 다르면 멈춘다.
4. 도착 경우 하나가 끝날 때마다 <팀>/data/runs/direct/ 에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는
   건너뛴다). 환경·코드·입력이 처음과 다르면 이어서 하지 않고 멈춘다. 처음부터 다시 하려면 --restart 를 붙인다.
"""
import argparse
import csv
import json
import os
import time

import cases
import check_cases
import run_answers as common
import settings

ROOT = check_cases.ROOT
STEP = "direct"
COLUMNS = ["model", "source_case", "case", "image_id", "input", "n_tokens", "ended_eos", "hit_cap", "latency_s", "text"]
# 9-1단계 attempts.csv에 있어야 하는 열(README 7장 '9단계 시도 기록'). 열이 더 있어도 된다
ATTEMPT_COLUMNS = ["source_case", "set_order", "coco_id", "attempt", "status", "clean_n_tokens", "clean_hit_cap",
                   "adv_n_tokens", "adv_hit_cap", "linf", "png_sha256"]
stop = common.stop


def set_photos(limit):
    """쓰는 사진: attack_set.csv의 set_order 1~N_DIRECT. [(set_order, 'coco_id 12자리')]"""
    rows = sorted(common.read_csv(ROOT / "공통" / "attack_set.csv", "attack_set.csv"), key=lambda r: int(r["set_order"]))
    photos = [(int(r["set_order"]), f"{int(r['coco_id']):012d}") for r in rows[:settings.N_DIRECT]]
    return photos if limit is None else photos[:limit]


def finished(row):
    """그 사진의 처리가 끝났는가: excluded, success, 또는 마지막 시도까지 fail."""
    return row["status"] in ("excluded", "success") or (
        row["status"] == "fail" and int(row["attempt"]) >= settings.ATTACK_MAX_TRIES)


def read_attempts(rt, photos, env, trial):
    """9-1단계 시도 기록을 읽고 확인한다.
    ({출발 경우: {사진: 마지막 시도 줄}}, 출발 경우별 사진 수 기록, 확인 기록, attempts.csv의 SHA-256)"""
    folder = rt.TEAM / "adv_direct"
    rows = common.read_csv(folder / "attempts.csv", "adv_direct/attempts.csv")
    meta = common.read_json(folder / "attempts.json", "adv_direct/attempts.json")
    missing = [c for c in ATTEMPT_COLUMNS if not rows or c not in rows[0]]
    if missing:
        stop(f"adv_direct/attempts.csv에 줄이 없거나 열이 없다: {missing} (README 7장)")
    digest = common.sha256((folder / "attempts.csv").read_bytes())
    recorded = meta.get("csv") or {}
    if recorded.get("sha256") != digest or recorded.get("rows") != len(rows):
        stop("adv_direct/attempts.csv가 attempts.json에 적힌 줄 수·SHA-256과 다르다 → 9-1단계가 끝난 파일을 그대로 쓴다")
    now = json.loads(json.dumps(env))
    differs = [k for k in check_cases.ENV_KEYS if (meta.get("environment") or {}).get(k) != now.get(k)]

    order_of = {coco_id: order for order, coco_id in set_photos(None)}
    last = {source: {} for source in settings.CASES_DIRECT}
    for row in rows:
        source, coco_id = row["source_case"], f"{int(row['coco_id']):012d}"
        if source not in last:
            stop(f"attempts.csv에 settings.CASES_DIRECT에 없는 출발 경우가 있다: {source}")
        if order_of.get(coco_id) != int(row["set_order"]):
            stop(f"attempts.csv의 사진 {coco_id}(set_order {row['set_order']})이 attack_set.csv의 앞 "
                 f"{settings.N_DIRECT}장과 맞지 않는다")
        if row["status"] == "success" and (int(row["linf"]) > settings.EPS or row["adv_hit_cap"] != "True"):
            stop(f"{source} {coco_id}: success 줄인데 판정 조건(linf ≤ EPS, adv_hit_cap)과 맞지 않는다")
        last[source][coco_id] = row      # 같은 사진의 뒤 시도가 앞 시도를 덮는다

    unfinished = [f"{source}:{coco_id}" for source in last for _, coco_id in photos
                  if coco_id not in last[source] or not finished(last[source][coco_id])]
    counts = {}
    for source in last:
        final = [last[source][coco_id]["status"] for _, coco_id in photos if coco_id in last[source]]
        counts[source] = {"photos": len(photos), **{s: final.count(s) for s in ("success", "fail", "excluded")}}
    check = {"pass": not differs and not unfinished, "environment_differs": differs, "unfinished": len(unfinished)}
    if not check["pass"] and not trial:
        stop(f"9-1단계 기록이 시작 조건과 맞지 않는다: 실행 환경이 다른 항목 {differs}, 끝나지 않은 (출발 경우:사진) "
             f"{len(unfinished)}개(예: {unfinished[:3]})")
    if not check["pass"]:
        print(f"알림(시험이라 계속한다): 9-1단계 기록이 시작 조건과 맞지 않는다. {check}")
    return last, counts, check, digest


def photo_plan(rt, photos, last):
    """출발 경우마다 넣을 사진(success가 나온 것, set_order 순서). {출발 경우: [{"image_id", "path", "sha256"}]}"""
    plan = {}
    for source in settings.CASES_DIRECT:
        plan[source] = [{"image_id": coco_id, "path": rt.TEAM / "adv_direct" / source / f"{coco_id}.png",
                         "sha256": last[source][coco_id]["png_sha256"]}
                        for _, coco_id in photos
                        if coco_id in last[source] and last[source][coco_id]["status"] == "success"]
    return plan


def verify_pngs(rt, plan, trial):
    """넣을 PNG의 SHA-256을 attempts.csv, SHA256SUMS와 맞대 본다. 같으면 그 해시들을 돌려주고, 다르면 멈춘다."""
    folder = rt.TEAM / "adv_direct"
    try:
        lines = (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as err:
        stop(f"adv_direct/SHA256SUMS를 읽지 못했다({type(err).__name__})")
    sums = {}
    for line in lines:                        # 'SHA-256  <출발 경우>/<coco_id>.png' (shasum -a 256, sha256sum 모두 이 꼴)
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest
    hashes, bad, wanted = {}, [], set()
    for source, items in plan.items():
        hashes[source] = {}
        for item in items:
            name = f"{source}/{item['image_id']}.png"
            wanted.add(name)
            got = common.sha256(item["path"].read_bytes()) if item["path"].exists() else None
            hashes[source][item["image_id"]] = got
            if got is None or got != item["sha256"] or got != sums.get(name):
                bad.append(name)
    if bad:
        stop(f"공격 PNG {len(bad)}개가 없거나 기록(attempts.csv, SHA256SUMS)과 다르다(예: {bad[:3]}) "
             "→ 9-1단계에서 만든 파일을 그대로 쓴다")
    extra = sorted(set(sums) - wanted)
    if extra and not trial:
        stop(f"SHA256SUMS에 success 시도의 PNG가 아닌 파일이 {len(extra)}개 있다(예: {extra[:3]})")
    return hashes


def run_case(rt, processor, target, reference, plan):
    """도착 경우 하나의 모델을 올려, 출발 경우마다 만든 사진을 모두 넣는다. (경우 기록, CSV 줄들)"""
    started = time.time()
    model, info = cases.build(rt, target, reference)
    rows = []
    try:
        for source, items in plan.items():
            for item in items:
                ans = rt.answer(model, processor, rt.read_png(item["path"]))
                rows.append([rt.MODEL_KEY, source, target, item["image_id"], "adv"] + [ans[k] for k in COLUMNS[5:]])
    finally:
        model = None
        cases.free()
    info.update({"rows": len(rows), "seconds": round(time.time() - started, 1),
                 "hit_cap": {source: sum(r[7] for r in rows if r[1] == source) for source in plan}})
    return info, rows


def self_check(target, rows):
    """도착 경우가 출발 경우이기도 할 때: 그 경우에서 만든 사진의 답이 모두 512에 닿는가(9-1단계 판정의 재현)."""
    own = [r for r in rows if r[1] == target]
    hit = sum(r[7] for r in own)
    return {"pass": hit == len(own), "photos": len(own), "hit_cap": hit, "not_hit": [r[3] for r in own if not r[7]]}


def append_rows(path, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)
        fp.flush()
        os.fsync(fp.fileno())


def run_all(rt, args, targets, plan, state, work):
    """남은 도착 경우를 차례로 돌린다. 경우 하나가 끝날 때마다 줄과 기록을 저장한다."""
    todo = [name for name in targets if name not in state["cases"]]
    if not todo:
        return
    first_row = sum(c["rows"] for c in state["cases"].values()) + 1
    run = {"run": len(state["runs"]) + 1, "first_row": first_row, "rows_added": 0,
           "code_commit": state["identity"]["environment"]["code_commit"]}
    state["runs"].append(run)
    processor, reference = rt.load_processor(), cases.reference_weights(rt)
    for count, name in enumerate(todo, 1):
        info, rows = run_case(rt, processor, name, reference, plan)
        if name in plan:
            check = self_check(name, rows)
            state["self_check"][name] = check
            if not check["pass"] and args.limit is None:
                common.save_state(work / "self_check_failed.json", {name: check})
                stop(f"{name}에서 만든 사진의 답이 9-1단계 판정과 다르다 → 환경이 달라졌는지 확인한다. "
                     f"설정을 바꾸지 말고 그대로 알린다. {check}")
            if not check["pass"]:
                print(f"알림(시험이라 계속한다): {name}에서 만든 사진의 답이 9-1단계 판정과 다르다. {check}")
        append_rows(work / "rows.csv", rows)
        state["cases"][name] = info
        run["rows_added"] += len(rows)
        common.save_state(work / "state.json", state)
        print(f"[{count}/{len(todo)}] {name:10s} {len(rows)}줄, {info['seconds'] / 60:.1f}분 | "
              f"출발 경우별 512에 닿은 답 {info['hit_cap']}")


def finalize(rt, args, state, work, checks, inputs, sources):
    """모든 도착 경우가 끝났으면 결과 파일을 쓴다. 시험이면 작업 폴더에, 아니면 <팀>/results/ 에 쓴다."""
    data = (work / "rows.csv").read_bytes()
    meta = {"environment": state["identity"]["environment"], "step": STEP, "sources": sources,
            "cases": state["cases"], "inputs": inputs, "self_check": state["self_check"], "checks": checks,
            "csv": {"rows": sum(c["rows"] for c in state["cases"].values()), "sha256": common.sha256(data)},
            "runs": state["runs"],
            "note": "README 9단계. csv 한 줄 = 출발 경우에서 만든 사진 하나를 도착 경우(case)에 넣은 답. sources는 출발 "
                    "경우마다 9-1단계에서 처리한 사진 수와 마지막 상태별 수. cases의 hit_cap은 출발 경우별로 512에 닿은 "
                    "답의 수. runs의 first_row는 그 실행이 처음 쓴 줄 번호(머리줄 제외)"}
    out = work if args.limit is not None else rt.TEAM / "results"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{STEP}.csv").write_bytes(data)
        (out / f"{STEP}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except OSError as err:
        stop(f"결과 파일을 쓰지 못했다({type(err).__name__})")
    print(f"저장: {out.relative_to(ROOT).as_posix()}/{STEP}.csv, {STEP}.json | {meta['csv']['rows']}줄")


def parse_args():
    parser = argparse.ArgumentParser(description="9-2단계: 압축 모델에서 만든 공격 사진을 여러 경우에 넣기")
    parser.add_argument("--team", required=True, choices=sorted(check_cases.TEAMS))
    parser.add_argument("--limit", type=int, help="시험: 앞에서 이 수만큼의 사진만 돌린다. 결과 폴더에 쓰지 않는다")
    parser.add_argument("--cases", help="시험(--limit과 함께): 쉼표로 나눈 도착 경우 이름만 돌린다")
    parser.add_argument("--restart", action="store_true", help="이어 하지 않고 처음부터 다시 한다")
    args = parser.parse_args()
    if args.cases and args.limit is None:
        parser.error("--cases 는 시험(--limit)에서만 쓴다")
    unknown = [name for name in (args.cases or "").split(",") if name and name not in settings.CASES_DIRECT_TARGETS]
    if unknown:
        parser.error(f"settings.CASES_DIRECT_TARGETS에 없는 경우: {unknown}")
    return args


def main():
    args = parse_args()
    trial = args.limit is not None
    rt = check_cases.load_runtime(args.team)
    targets = list(settings.CASES_DIRECT_TARGETS)
    if args.cases:
        targets = [name for name in targets if name in args.cases.split(",")]
    photos = set_photos(args.limit)
    rt.setup()
    env = rt.environment()
    checks = common.preconditions(rt, env, trial)
    last, sources, attempts_check, attempts_sha = read_attempts(rt, photos, env, trial)
    checks["direct_attempts"] = attempts_check
    plan = photo_plan(rt, photos, last)
    inputs = {"attempts_csv_sha256": attempts_sha, "adv_sha256": verify_pngs(rt, plan, trial)}
    identity = {"environment": env, "step": STEP, "cases": targets, "sources": list(plan),
                "images": [coco_id for _, coco_id in photos], "inputs": inputs,
                "prompt": settings.PROMPT, "gen_kwargs": settings.GEN_KWARGS}
    identity = json.loads(json.dumps(identity))      # 저장했다 읽은 기록과 같은 꼴로 맞춘다(비교용)
    work = rt.DATA / ("trial" if trial else "runs") / STEP
    work.mkdir(parents=True, exist_ok=True)
    state = common.load_state(work, identity, args.restart)
    state.setdefault("self_check", {})
    if not state["cases"]:           # 새로 시작한다: 이전 실행이 남긴 확인 실패 기록을 지운다
        (work / "self_check_failed.json").unlink(missing_ok=True)
    run_all(rt, args, targets, plan, state, work)
    finalize(rt, args, state, work, checks, inputs, sources)


if __name__ == "__main__":
    main()
