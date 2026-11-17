"""3단계(B팀, Qwen2-VL): 공격 사진 시도(README 3단계, 이미지생성/README.md의 제작 방식). A팀 make_attack_set.py와 같은 규칙.

실행: python B팀/code/make_attack_set.py                        # cand 전부(order 순서) → B팀/adv/attempts.csv
      python B팀/code/make_attack_set.py --split dev --limit 2  # 코드 확인용(기록은 B팀/data/dev_attempts/)
결과: B팀/adv/attempts.csv, attempts.json   처리한 모든 후보·시도의 기록과 실행 환경(올린다)
      B팀/data/attempts/<split>/try<k>/<id>.png   시도마다 만든 공격 PNG(올리지 않는다. 세트에 든 것만 따로 올린다)

후보 한 장의 처리(order 순서, 건너뛰지 않음):
1. ref에 정상 PNG를 넣은 답이 종료 토큰 없이 512에 닿으면 excluded(attempt 0)로 적고 끝낸다.
2. 시도 k(1부터 ATTACK_MAX_TRIES까지): attack_qwen.attack_one으로 공격 사진을 만들어 PNG로 저장한다.
   1번째는 노이즈 0에서, 2번째부터는 ±EPS 안 무작위에서 시작한다(씨앗은 ATTACK_SEED·사진 번호·시도 번호).
3. 저장한 PNG를 다시 읽어 판정한다: 정상 PNG와의 픽셀 차이 최댓값 ≤ EPS이고, runtime.answer로 ref에 넣은 답이
   종료 토큰 없이 512에 닿으면 success, 아니면 fail. 공격 코드가 출력하는 값은 판정에 쓰지 않는다.
4. success면 그 후보는 끝, fail이면 다음 시도.

- 판정은 4단계와 같은 runtime.load_ref()·runtime.answer()로 한다. 공격과 판정은 같은 모델 한 벌을 쓴다.
- 끊겨도 같은 명령으로 이어서 한다. 끝난 후보는 건너뛰고, 시도가 남은 후보는 다음 시도부터 한다.
- 설정·코드가 기록을 만들 때와 다르면 멈춘다(한 기록 = 한 설정).
"""
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch

import attack_qwen as atk
import runtime as rt
from runtime import settings

COLUMNS = ["cand_order", "coco_id", "attempt", "status", "ref_clean_n_tokens", "ref_clean_hit_cap",
           "ref_adv_n_tokens", "ref_adv_hit_cap", "linf", "png_sha256",
           # 이미지생성/README.md가 정한 '시도마다 달라지는 값'과 과정 기록
           "init", "init_seed", "iters_run", "stopped_early", "attack_seconds"]


def init_seed(coco_id, attempt):
    """2번째 시도부터 쓰는 무작위 시작의 씨앗. 1번째 시도는 None(노이즈 0에서 시작). A팀과 같은 식."""
    if attempt == 1:
        return None
    return int(hashlib.sha256(f"{settings.ATTACK_SEED}:{coco_id}:{attempt}".encode()).hexdigest()[:15], 16)


def config():
    """이 기록을 만든 설정. 이어서 돌릴 때 같아야 한다."""
    code = {p: rt.sha256(rt.ROOT / p) for p in ["B팀/code/attack_qwen.py", "B팀/code/runtime.py",
                                                 "B팀/code/make_attack_set.py", "공통/settings.py"]}
    attack = {k: getattr(settings, k) for k in ["ATTACK_ITERS", "EPS", "STEP", "TARGET_LEN", "REFRESH", "W_UNC", "W_DIV"]}
    return {"settings_version": settings.SETTINGS_VERSION, "attack_dtype": rt.M["attack_dtype"], "attack": attack,
            "max_tries": settings.ATTACK_MAX_TRIES, "attack_seed": settings.ATTACK_SEED,
            "model_id": rt.M["model_id"], "revision": rt.M["revision"], "code_sha256": code}


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert not rows or list(rows[0].keys()) == COLUMNS, f"{path}의 열이 다르다"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cand", "dev"], default="cand")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 몇 장만(0이면 전부). 코드 확인용")
    ap.add_argument("--out", default=None, help="기록 폴더(기본: cand는 B팀/adv, dev는 B팀/data/dev_attempts)")
    cli = ap.parse_args()
    assert isinstance(settings.ATTACK_MAX_TRIES, int) and settings.ATTACK_MAX_TRIES >= 1, \
        "settings.ATTACK_MAX_TRIES가 정해지지 않았다(README 1-1단계)"
    assert 1 <= settings.ATTACK_ITERS <= settings.ATTACK_MAX_ITERS
    assert settings.W_DIV == 0, "Qwen 공격 코드는 다양성 손실(W_DIV)을 쓰지 않는다"
    out_dir = rt.Path(cli.out) if cli.out else (rt.TEAM / "adv" if cli.split == "cand" else rt.DATA / "dev_attempts")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = rt.DATA / "attempts" / (cli.split if not cli.out else out_dir.name)
    log_path, meta_path = out_dir / "attempts.csv", out_dir / "attempts.json"

    rows = rt.image_rows(cli.split)
    rows = rows[:cli.limit] if cli.limit else rows
    missing = [cid for _, cid in rows if not rt.png_path(cli.split, "clean", cid).exists()]
    assert not missing, f"정상 PNG가 없는 사진 {len(missing)}장 → prepare_images.py --splits {cli.split}를 먼저 실행"

    cfg = config()
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))["config"]
        assert old == cfg, "이 기록은 다른 설정이나 다른 코드로 만들던 것이다 → 새 폴더(--out)를 쓴다"
    last = {r["coco_id"]: r for r in read_rows(log_path)}
    finished = {cid for cid, r in last.items() if r["status"] in ("excluded", "success")
                or (r["status"] == "fail" and int(r["attempt"]) >= settings.ATTACK_MAX_TRIES)}
    todo = [(o, cid) for o, cid in rows if cid not in finished]
    print(f"{cli.split} {len(rows)}장 중 남은 후보 {len(todo)}장 | 시도당 최대 {settings.ATTACK_ITERS}회, "
          f"최대 {settings.ATTACK_MAX_TRIES}번 시도", flush=True)
    if not todo:
        return

    rt.setup()
    processor = rt.load_processor()
    model = rt.load_ref(attack=True)
    assert model.dtype == getattr(torch, rt.M["attack_dtype"]), "공격 정밀도가 settings와 다르다"
    meta = {"config": cfg, "environment": rt.environment(), "split": cli.split,
            "pixel_gap_max": 0.0, "note": "attempts.csv 한 줄 = 후보 한 장의 시도 한 번"}
    if meta_path.exists():
        meta["pixel_gap_max"] = json.loads(meta_path.read_text(encoding="utf-8")).get("pixel_gap_max", 0.0)

    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for k, (order, cid) in enumerate(todo, 1):
            clean_img = rt.read_png(rt.png_path(cli.split, "clean", cid))
            ref_clean = rt.answer(model, processor, clean_img)
            base = {"cand_order": order, "coco_id": cid, "ref_clean_n_tokens": ref_clean["n_tokens"],
                    "ref_clean_hit_cap": ref_clean["hit_cap"]}
            if ref_clean["hit_cap"]:
                w.writerow(base | {"attempt": 0, "status": "excluded"})
                fp.flush()
                print(f"[{k}/{len(todo)}] {cid}: 정상 사진 답이 512 → excluded", flush=True)
                continue
            start = int(last[cid]["attempt"]) + 1 if cid in last else 1
            for attempt in range(start, settings.ATTACK_MAX_TRIES + 1):
                seed = init_seed(cid, attempt)
                png = work / f"try{attempt}" / f"{cid}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                adv, res = atk.attack_one(model, processor, clean_img, name=cid, init_seed=seed,
                                          log=lambda s: print(s, flush=True))
                seconds = time.perf_counter() - t0
                tmp = png.with_name(png.name + ".tmp")
                adv.save(tmp, format="PNG")
                tmp.replace(png)
                adv_img = rt.read_png(png)            # 저장한 PNG를 다시 읽어 판정한다
                linf = int(np.abs(np.asarray(adv_img, dtype=np.int16) - np.asarray(clean_img, dtype=np.int16)).max())
                gap = rt.pixel_gap(processor, adv_img)
                assert gap <= 1e-6, f"{cid}: 프로세서와 직접 계산의 차이 {gap:.2e} (README 5.3)"
                meta["pixel_gap_max"] = max(meta["pixel_gap_max"], gap)
                ref_adv = rt.answer(model, processor, adv_img)
                status = "success" if (linf <= settings.EPS and ref_adv["hit_cap"]) else "fail"
                w.writerow(base | {"attempt": attempt, "status": status, "ref_adv_n_tokens": ref_adv["n_tokens"],
                                   "ref_adv_hit_cap": ref_adv["hit_cap"], "linf": linf, "png_sha256": rt.sha256(png),
                                   "init": "zero" if seed is None else "random",
                                   "init_seed": "" if seed is None else seed, "iters_run": res["iters_run"],
                                   "stopped_early": res["stopped_early"], "attack_seconds": round(seconds, 1)})
                fp.flush()
                print(f"[{k}/{len(todo)}] {cid} 시도 {attempt}: {status} (ref 답 {ref_adv['n_tokens']}토큰, "
                      f"최대 차이 {linf}, {res['iters_run']}회, {seconds:.0f}초)", flush=True)
                if status == "success":
                    break
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    final = {r["coco_id"]: r["status"] for r in read_rows(log_path)}
    counts = {s: sum(v == s for v in final.values()) for s in ("success", "fail", "excluded")}
    print(f"\n처리한 후보 {len(final)}장 | 성공 {counts['success']} · 실패 {counts['fail']} · 제외 {counts['excluded']}"
          f" → {log_path.relative_to(rt.ROOT) if log_path.is_relative_to(rt.ROOT) else log_path}")


if __name__ == "__main__":
    main()
