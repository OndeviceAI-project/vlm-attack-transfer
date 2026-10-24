"""결과 CSV를 읽어 경우마다 전이율을 낸다 (README '반드시 지킬 것' 8번). 측정은 하지 않고 계산만 한다.

실행: python A팀/code/04_summarize.py A팀/results/smolvlm_phase1.csv

정의 (README 그대로):
- 성공     = 공격 사진(adv)의 답이 상한에 닿고, 같은 모델에서 원본 사진(clean)의 답은 닿지 않음
- 유효 사진 = ref에서 원본 사진이 상한에 닿지 않은 사진
- 전이율   = ref에서 성공한 사진 중 그 경우에서도 성공한 비율
두 팀 결과를 모은 그림은 공통 분석 코드가 그린다. 이 파일은 A팀 결과를 올리기 전에 확인하는 용도다.
"""
import csv
import sys
from collections import defaultdict


def main(path):
    hit, tokens, cases, images = {}, defaultdict(list), [], []
    with open(path, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            key = (r["case"], r["image_id"], r["input"])
            assert key not in hit, f"같은 줄이 두 번 있다: {key}"
            hit[key] = r["hit_cap"] == "True"
            tokens[(r["case"], r["input"])].append(int(r["n_tokens"]))
            if r["case"] not in cases:
                cases.append(r["case"])
            if r["image_id"] not in images:
                images.append(r["image_id"])
    assert "ref" in cases, "ref가 없는 결과 파일이다 (규칙 7)"

    def has(case, image):
        return all((case, image, k) in hit for k in ("clean", "adv"))

    def success(case, image):
        return hit[(case, image, "adv")] and not hit[(case, image, "clean")]

    valid = [i for i in images if has("ref", i) and not hit[("ref", i, "clean")]]
    base = [i for i in valid if success("ref", i)]
    print(f"파일: {path}")
    print(f"사진 {len(images)}장 | 유효 사진 {len(valid)}장 (ref에서 원본이 상한에 닿은 {len(images) - len(valid)}장 제외) | ref에서 성공 {len(base)}장")
    print()
    print("경우 | 전이율 (성공/ref 성공) | 상한에 닿은 수 clean, noise, adv | 평균 n_tokens clean, noise, adv")
    for c in cases:
        pool = [i for i in base if has(c, i)]
        ok = sum(success(c, i) for i in pool)
        rate = f"{100 * ok / len(pool):5.1f}% ({ok}/{len(pool)})" if pool else "계산 불가 (0장)"
        note = "" if len(pool) == len(base) else f"  ※ 이 경우에 결과가 없는 사진 {len(base) - len(pool)}장"
        hits = ", ".join(str(sum(hit.get((c, i, k), False) for i in images)) for k in ("clean", "noise", "adv"))
        means = ", ".join(f"{sum(tokens[(c, k)]) / len(tokens[(c, k)]):.1f}" if tokens[(c, k)] else "-" for k in ("clean", "noise", "adv"))
        print(f"{c} | {rate} | {hits} | {means}{note}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "사용법: python A팀/code/04_summarize.py <결과 CSV>"
    main(sys.argv[1])
