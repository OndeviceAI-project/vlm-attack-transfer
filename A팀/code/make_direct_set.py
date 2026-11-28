"""9-1단계: 압축 모델에서 공격 사진 만들기(README 9-1단계, 이미지생성/README.md의 제작 방식).

실행: python A팀/code/make_direct_set.py
          세트 사진(공통/attack_set.csv의 set_order 1~settings.N_DIRECT), 출발 경우 settings.CASES_DIRECT → A팀/adv_direct/
      python A팀/code/make_direct_set.py --split dev --limit 2 --cases rtn-V-4,rtn-V-3,rand-V-3,bnb-V-4
          코드 확인용. dev 사진만 쓰고 기록은 A팀/data/dev_direct/ 에 남는다(--out으로 바꿀 수 있다)
결과: A팀/adv_direct/<출발 경우>/<coco_id 12자리>.png   success 시도의 PNG만(올린다)
      A팀/adv_direct/SHA256SUMS                          adv_direct 안에서 shasum -a 256 */*.png 로 만든 것과 같은 내용
      A팀/adv_direct/attempts.csv, attempts.json         처리한 모든 (출발 경우, 사진)의 시도 기록과 실행 환경(올린다)
      A팀/data/direct_attempts/<기록 폴더 이름>/<출발 경우>/try<k>/<coco_id>.png   시도마다 만든 PNG(올리지 않는다)

처리 순서: 출발 경우(settings.CASES_DIRECT 순서) → 사진(set_order 순서) → 시도. 건너뛰지 않는다.
출발 경우마다 cases.build(rt, 경우, reference)로 모델을 한 벌 올려, 그 모델 하나로 만들고 판정한다.
(출발 경우, 사진) 하나의 처리는 3단계(make_attack_set.py)와 같다.
1. 그 모델에 정상 PNG를 넣은 답이 종료 토큰 없이 512에 닿으면 excluded(attempt 0)로 적고 끝낸다.
2. 시도 k(1부터 ATTACK_MAX_TRIES까지): attack_smolvlm.attack_one으로 공격 PNG를 만든다. 시작점과 씨앗은 3단계와 같다
   (make_attack_set.init_seed: 1번째는 노이즈 0, 2번째부터 ATTACK_SEED·사진 번호·시도 번호로 정한 ±EPS 무작위).
3. 저장한 PNG를 다시 읽어 판정한다. 정상 PNG와의 픽셀 차이 최댓값이 EPS 이하이고, runtime.answer로 그 모델에 넣은
   답이 종료 토큰 없이 512에 닿으면 success, 아니면 fail. 공격 코드가 출력하는 값은 판정에 쓰지 않는다.
4. success면 그 (출발 경우, 사진)은 끝, fail이면 다음 시도. 성공이 적게 나와도 설정이나 방식을 바꾸지 않고 그대로 적는다.

3단계와 같은 확인
- 처리할 사진의 정상 PNG가 2단계 input-manifest.json의 SHA-256과 같아야 한다. 없거나 다르면 멈춘다.
- 시도마다 저장한 PNG로 pixel_values 차이(프로세서와 직접 정규화)를 보고, 1e-6을 넘으면 멈춘다.
- 끊겨도 같은 명령으로 이어서 한다. 끝난 (출발 경우, 사진)은 건너뛰고, 시도가 남은 것은 다음 시도부터 한다.
- 제작·판정 설정이나 코드, 실행 환경(GPU·CUDA·패키지·결정적 실행 설정)이 기록과 다르면 멈춘다(README 5.1).
  출발 경우 모델의 기록(census·가중치 변화)이 앞서 올린 것과 다를 때도 멈춘다.
- 세트 사진으로 돌릴 때는 코드에 커밋하지 않은 수정이 있으면 멈춘다(README 5.1).

3단계 파일(이미지생성/attack_smolvlm.py, A팀/code/runtime.py, A팀/code/make_attack_set.py)은 고치지 않고 불러 쓴다.
attempts.json은 3단계와 같은 항목이다. environment(rt.setup() 뒤의 rt.environment()), cases(출발 경우마다 cases.build가
돌려준 기록), inputs(정상 PNG의 SHA-256), csv(attempts.csv의 줄 수·SHA-256), runs, config, pixel_gap_max.
"""
import argparse
import csv
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import runtime as rt
from runtime import settings
import cases  # noqa: E402  runtime이 공통 폴더를 import 경로에 넣는다
import make_attack_set as mas  # noqa: E402  3단계 코드. 시작점 씨앗·제작 인자·설정 키·입력 확인을 그대로 쓴다

atk = mas.atk

COLUMNS = ["source_case", "set_order", "coco_id", "attempt", "status", "clean_n_tokens", "clean_hit_cap",
           "adv_n_tokens", "adv_hit_cap", "linf", "png_sha256",
           "init", "init_seed", "iters_run", "stopped_early", "nan_stop", "attack_seconds"]
# 이어서 돌릴 때 같아야 하는 settings 값: 3단계의 제작·판정 값과, 출발 경우 모델을 만드는 값
CONFIG_KEYS = mas.CONFIG_KEYS + ["QUANT_GROUP", "RAND_SEED"]
CODE = mas.CODE + ["공통/cases.py", "공통/quant_rtn.py"]
SELF = "A팀/code/make_direct_set.py"


def source_cases(cli):
    """출발 경우 목록. 세트 사진은 settings.CASES_DIRECT 그대로, dev 시험은 --cases로 고를 수 있다."""
    if cli.split == "set":
        assert hasattr(settings, "CASES_DIRECT") and hasattr(settings, "N_DIRECT"), \
            "settings에 9단계 값(CASES_DIRECT·N_DIRECT)이 없다 → 9단계 공통 PR을 받은 뒤에 돌린다"
        names = list(settings.CASES_DIRECT)
    else:
        names = cli.cases.split(",") if cli.cases else list(getattr(settings, "CASES_DIRECT", []))
    assert names, "출발 경우가 없다(dev 시험이면 --cases로 준다)"
    assert len(set(names)) == len(names), f"출발 경우가 겹친다: {names}"
    for name in names:
        cases.parse(name)                        # settings.CASES_ALL에 없는 이름이면 멈춘다
    return names


def photo_rows(split, limit):
    """처리할 사진 [(order, coco_id)]와, 순서를 맞대 볼 전체 목록.
    set: 공통/attack_set.csv의 set_order 1~N_DIRECT(정상 PNG는 cand). dev: image_list.csv의 dev, order 순서."""
    if split == "dev":
        everything = rt.image_rows("dev")
        return (everything[:limit] if limit else everything), everything
    with open(rt.ROOT / "공통" / "attack_set.csv", newline="", encoding="utf-8") as fp:
        everything = sorted((int(r["set_order"]), f"{int(r['coco_id']):012d}") for r in csv.DictReader(fp))
    assert [o for o, _ in everything] == list(range(1, len(everything) + 1)), \
        "attack_set.csv의 set_order가 1부터 빠짐없이 이어지지 않는다"
    assert 1 <= settings.N_DIRECT <= len(everything), f"N_DIRECT {settings.N_DIRECT}가 세트 {len(everything)}장 밖이다"
    return everything[:settings.N_DIRECT], everything


def config(names):
    """이 기록을 만든 설정. 이어서 돌릴 때 같아야 한다(JSON으로 한 번 바꿔 저장한 값과 바로 비교할 수 있게)."""
    code = {p: rt.sha256(rt.ROOT / p) for p in CODE}
    code[SELF] = rt.sha256(Path(__file__))
    cfg = {"settings": {k: getattr(settings, k) for k in CONFIG_KEYS}, "source_cases": names, "model": rt.M,
           "code_sha256": code}
    return json.loads(json.dumps(cfg))


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert not rows or list(rows[0].keys()) == COLUMNS, f"{path}의 열이 다르다"
    return rows


def save(meta_path, meta, log_path):
    meta["csv"] = ({"rows": len(read_rows(log_path)), "sha256": rt.sha256(log_path)} if log_path.exists()
                   else {"rows": 0, "sha256": None})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def last_rows(rows):
    """(출발 경우, coco_id)마다 마지막 줄."""
    last = {}
    for r in rows:
        last[(r["source_case"], r["coco_id"])] = r
    return last


def is_finished(r):
    return r["status"] in ("excluded", "success") or \
        (r["status"] == "fail" and int(r["attempt"]) >= settings.ATTACK_MAX_TRIES)


def check_rows(rows, names, targets):
    """기록이 처리 규칙대로인지 본다(select_attack_set.py가 3단계 기록에 하는 확인과 같다).
    excluded는 attempt 0 한 줄, 아니면 attempt 1부터 빠짐없이 이어지고 마지막 줄만 success일 수 있으며,
    마지막이 fail이면 attempt가 ATTACK_MAX_TRIES다. 이번에 처리할 (출발 경우, 사진)은 모두 끝나 있어야 한다.
    본 실행을 시작한 뒤에 N_DIRECT를 줄이면 줄인 수 밖의 사진(set_order > N_DIRECT)에 시도가 남은 기록이 있을 수 있다.
    그 기록은 지우거나 이어 하지 않고 그대로 두므로(README 9단계 '순서와 고정'), 시도 순서만 보고 끝났는지는 보지 않는다."""
    wanted = {(c, cid) for c in names for _, cid in targets}
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["source_case"], r["coco_id"]), []).append(r)
    for (case, cid), rs in by_pair.items():
        attempts, status = [int(r["attempt"]) for r in rs], [r["status"] for r in rs]
        if status == ["excluded"]:
            assert attempts == [0], f"{case} {cid}: excluded인데 attempt가 0이 아니다"
            continue
        assert attempts == list(range(1, len(rs) + 1)) and all(s == "fail" for s in status[:-1]) \
            and status[-1] in ("success", "fail"), f"{case} {cid}: 시도 기록이 규칙과 다르다 ({attempts}, {status})"
        assert status[-1] == "success" or attempts[-1] == settings.ATTACK_MAX_TRIES or (case, cid) not in wanted, \
            f"{case} {cid}: 마지막이 fail인데 시도가 {attempts[-1]}번뿐이다(처리가 끝나지 않음)"
    missing = [(c, cid) for c in names for _, cid in targets if (c, cid) not in by_pair]
    assert not missing, f"기록이 없는 (출발 경우, 사진) {len(missing)}개(예: {missing[0]})"


def publish(out_dir, work, rows):
    """success 줄의 PNG를 out_dir/<출발 경우>/<coco_id>.png에 두고 SHA256SUMS를 쓴다.
    이미 있으면 해시가 기록과 같은지 보고, 없으면 시도 폴더의 PNG(해시가 기록과 같아야 한다)를 복사한다.
    success가 아닌 PNG가 out_dir에 있으면 멈춘다. 돌릴 때마다 처음부터 다시 맞추므로 끊겨도 어긋나지 않는다."""
    want = {}
    for r in rows:
        if r["status"] == "success":
            want[f"{r['source_case']}/{r['coco_id']}.png"] = (r["png_sha256"], work / r["source_case"]
                                                              / f"try{r['attempt']}" / f"{r['coco_id']}.png")
    extra = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.glob("*/*.png")
                   if p.relative_to(out_dir).as_posix() not in want)
    assert not extra, f"success 기록이 없는 PNG {len(extra)}장이 {out_dir}에 있다(예: {extra[0]})"
    for rel, (sha, src) in want.items():
        dst = out_dir / rel
        if not dst.exists():
            assert src.exists() and rt.sha256(src) == sha, f"{rel}: 시도 폴더의 PNG가 없거나 기록과 다르다({src})"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        assert rt.sha256(dst) == sha, f"{rel}: PNG의 SHA-256이 attempts.csv의 png_sha256과 다르다"
    lines = [f"{sha}  {rel}\n" for rel, (sha, _) in sorted(want.items())]
    (out_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return len(lines)


def report(rows, names, targets):
    """출발 경우마다 처리 결과와 걸린 시간(시험 결과를 알릴 때 쓴다). 이번에 처리할 사진(targets)만 세고,
    그 밖의 사진(N_DIRECT를 줄이기 전에 처리한 것)은 수만 적는다."""
    wanted = {cid for _, cid in targets}
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["source_case"], r["coco_id"]), []).append(r)
    for case in names:
        pairs = [rs for (c, cid), rs in by_pair.items() if c == case and cid in wanted]
        outside = sum(c == case and cid not in wanted for c, cid in by_pair)
        final = [rs[-1]["status"] for rs in pairs]
        n = {s: final.count(s) for s in ("success", "fail", "excluded")}
        tried = [rs for rs in pairs if rs[-1]["status"] != "excluded"]
        first = sum(rs[-1]["status"] == "success" and rs[-1]["attempt"] == "1" for rs in tried)
        iters = [sum(int(r["iters_run"]) for r in rs) for rs in tried if rs[-1]["status"] == "success"]
        secs = [sum(float(r["attack_seconds"]) for r in rs) for rs in tried]
        line = (f"  {case:9s} 사진 {len(pairs)}장 | success {n['success']}(1번째 시도 {first}) · fail {n['fail']}"
                f" · excluded {n['excluded']}")
        if iters:
            line += f" | 성공까지 반복 중앙 {statistics.median(iters):.0f}회(최대 {max(iters)})"
        if secs:
            line += f" | 한 장 공격 시간 중앙 {statistics.median(secs):.0f}초(최대 {max(secs):.0f}초)"
        if outside:
            line += f" | 범위 밖 사진 {outside}장(세지 않음)"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["set", "dev"], default="set",
                    help="set: 세트 사진 set_order 1~N_DIRECT(기본). dev: 코드 확인용 dev 사진")
    ap.add_argument("--limit", type=int, default=0, help="dev 사진 앞에서부터 몇 장만(0이면 전부)")
    ap.add_argument("--cases", default="", help="dev 시험의 출발 경우(쉼표로). 비우면 settings.CASES_DIRECT")
    ap.add_argument("--out", default=None, help="dev 시험의 기록 폴더(기본 A팀/data/dev_direct)")
    ap.add_argument("--manifest", default=str(rt.TEAM / "results" / "input-manifest.json"), help="2단계 결과")
    cli = ap.parse_args()
    if cli.split == "set":
        assert not (cli.limit or cli.cases or cli.out), \
            "--limit·--cases·--out은 dev 시험에서만 쓴다(세트 사진은 settings 그대로 끝까지 돌린다)"
    assert isinstance(settings.ATTACK_MAX_TRIES, int) and settings.ATTACK_MAX_TRIES >= 1
    assert 1 <= settings.ATTACK_ITERS <= settings.ATTACK_MAX_ITERS
    names = source_cases(cli)
    png_split = "cand" if cli.split == "set" else "dev"       # 세트 사진의 정상 PNG는 cand에 있다
    out_dir = rt.TEAM / "adv_direct" if cli.split == "set" else (Path(cli.out) if cli.out else rt.DATA / "dev_direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = rt.DATA / "direct_attempts" / out_dir.name
    log_path, meta_path = out_dir / "attempts.csv", out_dir / "attempts.json"

    targets, everything = photo_rows(cli.split, cli.limit)
    missing = [cid for _, cid in targets if not rt.png_path(png_split, "clean", cid).exists()]
    assert not missing, f"정상 PNG가 없는 사진 {len(missing)}장(예: {missing[0]}) → 2단계 PNG를 확인한다"
    inputs = mas.input_hashes(png_split, targets, Path(cli.manifest))

    cfg = config(names)
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if old:
        assert old["config"] == cfg, "이 기록은 다른 설정·출발 경우·코드로 만들던 것이다 → 새 폴더에서 처음부터"
    done = read_rows(log_path)
    assert old or not done, "attempts.csv는 있는데 attempts.json이 없다 → 어느 설정으로 만든 기록인지 알 수 없으니 새 폴더에서"
    order_of = {cid: o for o, cid in everything}
    moved = [r["coco_id"] for r in done if order_of.get(r["coco_id"]) != int(r["set_order"])]
    assert not moved, f"기록의 set_order가 지금 사진 목록과 다른 사진 {len(moved)}장(예: {moved[0]})"
    last = last_rows(done)
    todo = {c: [(o, cid) for o, cid in targets if not ((c, cid) in last and is_finished(last[(c, cid)]))]
            for c in names}
    n_todo = sum(len(v) for v in todo.values())
    print(f"출발 경우 {len(names)}개 × 사진 {len(targets)}장 중 남은 것 {n_todo}개 | 시도당 최대 {settings.ATTACK_ITERS}회, "
          f"최대 {settings.ATTACK_MAX_TRIES}번 시도", flush=True)
    for c in names:
        print(f"  {c}: 남은 사진 {len(todo[c])}장", flush=True)
    if not n_todo:
        if old:
            save(meta_path, old, log_path)       # 마지막 줄을 쓰고 기록 전에 끊긴 경우를 맞춘다
        rows = read_rows(log_path)
        check_rows(rows, names, targets)
        print(f"SHA256SUMS {publish(out_dir, work, rows)}줄", flush=True)
        report(rows, names, targets)
        return

    rt.setup()
    processor = rt.load_processor()
    assert atk.eos_token_ids(processor) == sorted(rt.EOS), "종료 토큰이 settings와 다르다"
    env = rt.environment()
    if cli.split == "set":
        assert env["code_commit"] and not env["code_commit"].endswith("-dirty"), \
            "코드에 커밋하지 않은 수정이 있다(README 5.1) → 커밋한 코드로 돌린다"
    if old:
        assert mas.same_machine(old["environment"]) == mas.same_machine(json.loads(json.dumps(env))), \
            "실행 환경(GPU·CUDA·패키지·결정적 실행 설정)이 이 기록과 다르다(README 5.1) → 새 폴더에서 처음부터"
    mean = torch.tensor(processor.image_processor.image_mean, device="cuda").view(3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device="cuda").view(3, 1, 1)
    # bnb 경우의 가중치 변화를 재는 원본 가중치. bnb 출발 경우를 이번에 올릴 때만 만든다
    reference = (cases.reference_weights(rt)
                 if any(todo[c] and cases.parse(c)[0] == "bnb" for c in names) else None)
    runs = old.get("runs", [])
    run = {"run": len(runs) + 1, "first_row": len(done) + 1, "rows_added": 0, "code_commit": env["code_commit"]}
    runs.append(run)
    meta = {"environment": env,
            "cases": old.get("cases", {}),
            "inputs": {"split": png_split, "clean_sha256": old.get("inputs", {}).get("clean_sha256", {})
                       | inputs["clean_sha256"]},
            "csv": None, "runs": runs, "config": cfg,
            "pixel_gap_max": old.get("pixel_gap_max", 0.0),
            "note": "attempts.csv 한 줄 = (출발 경우, 사진)의 시도 한 번. 판정은 출발 경우 모델로 한다. "
                    "runs의 first_row는 그 실행이 처음 쓴 줄 번호(머리줄 제외)"}
    save(meta_path, meta, log_path)              # 줄을 쓰기 전에 이 실행의 설정·환경을 남긴다

    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()

        def write(row):
            w.writerow(row)
            fp.flush()
            run["rows_added"] += 1
            save(meta_path, meta, log_path)      # 줄마다 csv의 줄 수·SHA-256을 맞춘다

        for case in names:
            if not todo[case]:
                continue
            started = time.time()
            model, info = cases.build(rt, case, reference)
            info = json.loads(json.dumps(info))
            if case in meta["cases"]:
                assert meta["cases"][case] == info, f"{case} 모델의 기록(census·가중치 변화)이 앞서 올린 것과 다르다"
            meta["cases"] = {c: info if c == case else meta["cases"][c] for c in names
                             if c == case or c in meta["cases"]}
            assert model.dtype == getattr(torch, rt.M["attack_dtype"]), f"{case}: 공격 정밀도가 settings와 다르다"
            print(f"\n[{case}] 모델을 올림 | 가중치 변화 {info['weight_change_pct']} | 남은 사진 {len(todo[case])}장",
                  flush=True)
            try:
                for k, (order, cid) in enumerate(todo[case], 1):
                    clean_p = rt.png_path(png_split, "clean", cid)
                    clean_img = rt.read_png(clean_p)
                    clean = rt.answer(model, processor, clean_img)
                    base = {"source_case": case, "set_order": order, "coco_id": cid,
                            "clean_n_tokens": clean["n_tokens"], "clean_hit_cap": clean["hit_cap"]}
                    if clean["hit_cap"]:
                        write(base | {"attempt": 0, "status": "excluded"})
                        print(f"[{case} {k}/{len(todo[case])}] {cid}: 정상 사진 답이 512 → excluded", flush=True)
                        continue
                    start = int(last[(case, cid)]["attempt"]) + 1 if (case, cid) in last else 1
                    for attempt in range(start, settings.ATTACK_MAX_TRIES + 1):
                        seed = mas.init_seed(cid, attempt)
                        try_dir = work / case / f"try{attempt}"
                        try_dir.mkdir(parents=True, exist_ok=True)
                        t0 = time.perf_counter()
                        res = atk.attack_one(model, processor, str(clean_p), mas.attack_args(try_dir), "cuda", mean,
                                             std, sorted(rt.EOS), init_seed=seed)
                        seconds = time.perf_counter() - t0
                        model.zero_grad(set_to_none=True)
                        png = Path(res["out_path"])
                        adv_img = rt.read_png(png)            # 저장한 PNG를 다시 읽어 판정한다
                        linf = int(np.abs(np.asarray(adv_img, dtype=np.int16)
                                          - np.asarray(clean_img, dtype=np.int16)).max())
                        gap = rt.pixel_gap(processor, adv_img)
                        assert gap <= 1e-6, f"{case} {cid}: 프로세서와 직접 정규화의 차이 {gap:.2e} (README 5.3)"
                        meta["pixel_gap_max"] = max(meta["pixel_gap_max"], gap)
                        adv = rt.answer(model, processor, adv_img)
                        status = "success" if (linf <= settings.EPS and adv["hit_cap"]) else "fail"
                        sha = rt.sha256(png)
                        write(base | {"attempt": attempt, "status": status, "adv_n_tokens": adv["n_tokens"],
                                      "adv_hit_cap": adv["hit_cap"], "linf": linf, "png_sha256": sha,
                                      "init": "zero" if seed is None else "random",
                                      "init_seed": "" if seed is None else seed, "iters_run": res["iters_run"],
                                      "stopped_early": res["stopped_early"], "nan_stop": res["nan"],
                                      "attack_seconds": round(seconds, 1)})
                        print(f"[{case} {k}/{len(todo[case])}] {cid} 시도 {attempt}: {status} "
                              f"(답 {adv['n_tokens']}토큰, 최대 차이 {linf}, {res['iters_run']}회, {seconds:.0f}초)",
                              flush=True)
                        if status == "success":
                            dst = out_dir / case / f"{cid}.png"
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(png, dst)
                            break
            finally:
                model = None
                cases.free()
            print(f"[{case}] 끝 | {(time.time() - started) / 60:.1f}분", flush=True)
    save(meta_path, meta, log_path)

    rows = read_rows(log_path)
    check_rows(rows, names, targets)
    n_png = publish(out_dir, work, rows)
    shown = out_dir.relative_to(rt.ROOT) if out_dir.is_relative_to(rt.ROOT) else out_dir
    print(f"\n기록 {len(rows)}줄 · success PNG {n_png}장 → {shown}", flush=True)
    report(rows, names, targets)


if __name__ == "__main__":
    main()
