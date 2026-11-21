"""3단계 세트 정하기(README 3단계 3번): 두 팀의 attempts.csv에서 공격 사진 세트를 골라 공통/attack_set.csv를 쓴다.

실행: python 공통/select_attack_set.py
읽는 것: 공통/image_list.csv(cand), A팀/adv/attempts.csv·attempts.json, B팀/adv/attempts.csv·attempts.json
쓰는 것: 공통/attack_set.csv   열 set_order(1~N_ATTACK_SET), cand_order(image_list의 order), coco_id(12자리)

고르는 규칙: 두 팀 모두 마지막 시도가 success인 cand 후보를 order 순서로 앞에서 settings.N_ATTACK_SET장.

쓰기 전에 확인하는 것(하나라도 어긋나면 쓰지 않고 멈춘다):
1. 팀마다 attempts.csv의 줄 수와 SHA-256이 attempts.json의 csv 값과 같다(코드가 쓴 그대로의 기록인지).
2. 팀마다 attempts.csv의 후보가 image_list.csv의 cand 전부와 같고, cand_order가 그 order와 같다.
3. 팀마다 모든 후보의 처리가 끝났다: excluded는 attempt 0 한 줄, 아니면 attempt 1부터 빠짐없이 이어지고
   마지막 줄만 success일 수 있으며, 마지막이 fail이면 attempt가 ATTACK_MAX_TRIES다(README 3단계 2번).
4. 두 팀 공통 success가 N_ATTACK_SET장 이상이다. 모자라면 README 3단계 4번(cand 늘리기)을 따른다.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "공통"))
import settings  # noqa: E402

TEAMS = ("A팀", "B팀")
OUT = ROOT / "공통" / "attack_set.csv"


def fail(msg):
    sys.exit(f"멈춤: {msg}")


def cand_list():
    with open(ROOT / "공통" / "image_list.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "cand"]
    order = {r["coco_id"].zfill(12): int(r["order"]) for r in rows}
    if len(order) != len(rows) or sorted(order.values()) != list(range(1, len(rows) + 1)):
        fail("image_list.csv의 cand order가 1부터 빠짐없이 이어지지 않는다")
    if len(rows) != settings.N_CAND:
        fail(f"image_list.csv의 cand {len(rows)}장 ≠ settings.N_CAND {settings.N_CAND}")
    return order


def final_status(team, order):
    """팀의 attempts.csv를 확인하고 후보마다 마지막 시도의 status를 돌려준다."""
    adv = ROOT / team / "adv"
    raw = (adv / "attempts.csv").read_bytes()
    with open(adv / "attempts.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rec = json.loads((adv / "attempts.json").read_text(encoding="utf-8"))["csv"]
    if rec["sha256"] != hashlib.sha256(raw).hexdigest() or rec["rows"] != len(rows):
        fail(f"{team} attempts.csv가 attempts.json의 csv 값(줄 수·SHA-256)과 다르다")

    by_id = {}
    for r in rows:
        cid = r["coco_id"]
        if cid not in order:
            fail(f"{team} attempts.csv에 cand가 아닌 후보 {cid}")
        if int(r["cand_order"]) != order[cid]:
            fail(f"{team} {cid}의 cand_order {r['cand_order']} ≠ image_list order {order[cid]}")
        by_id.setdefault(cid, []).append(r)
    if set(by_id) != set(order):
        fail(f"{team} attempts.csv에 없는 cand 후보 {len(set(order) - set(by_id))}장")

    final = {}
    for cid, rs in by_id.items():
        attempts = [int(r["attempt"]) for r in rs]
        status = [r["status"] for r in rs]
        if status == ["excluded"]:
            if attempts != [0]:
                fail(f"{team} {cid}: excluded인데 attempt가 0이 아니다")
        elif attempts != list(range(1, len(rs) + 1)) or any(s != "fail" for s in status[:-1]) \
                or status[-1] not in ("success", "fail"):
            fail(f"{team} {cid}: 시도 기록이 규칙과 다르다 (attempt {attempts}, status {status})")
        elif status[-1] == "fail" and attempts[-1] != settings.ATTACK_MAX_TRIES:
            fail(f"{team} {cid}: 마지막이 fail인데 시도가 {attempts[-1]}번뿐이다(처리가 끝나지 않음)")
        final[cid] = status[-1]
    return final


def main():
    order = cand_list()
    finals = {t: final_status(t, order) for t in TEAMS}
    for t in TEAMS:
        n = {s: sum(v == s for v in finals[t].values()) for s in ("success", "fail", "excluded")}
        print(f"{t}: 후보 {len(finals[t])}장 | success {n['success']} · fail {n['fail']} · excluded {n['excluded']}")

    common = sorted((order[c], c) for c in order if all(finals[t][c] == "success" for t in TEAMS))
    print(f"두 팀 공통 success {len(common)}장")
    if len(common) < settings.N_ATTACK_SET:
        fail(f"공통 success가 N_ATTACK_SET {settings.N_ATTACK_SET}장보다 적다. README 3단계 4번대로 cand를 늘린다")

    chosen = common[:settings.N_ATTACK_SET]
    last = chosen[-1][0]
    for t in TEAMS:
        out = [f"{o}({finals[t][c]})" for c, o in sorted(order.items(), key=lambda x: x[1])
               if o <= last and finals[t][c] != "success"]
        print(f"cand_order 1~{last} 중 {t}에서 success가 아닌 후보 {len(out)}장: {', '.join(out)}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set_order", "cand_order", "coco_id"])
        for i, (o, c) in enumerate(chosen, 1):
            w.writerow([i, o, c])
    print(f"세트 {len(chosen)}장(cand_order {chosen[0][0]}~{last}) → {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
