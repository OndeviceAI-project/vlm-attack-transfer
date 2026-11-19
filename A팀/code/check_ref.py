"""1-2단계: ref 점검(README 1-2단계). 결과: A팀/results/check_ref.json

실행: python A팀/code/prepare_images.py --splits dev   # dev 5장의 PNG가 없을 때만 먼저
      python A팀/code/check_ref.py

2·3단계와 같은 runtime.load_ref()·answer()·pixel_gap()으로, ref(bf16 원본)와 dev 사진 5장을 점검한다.
1. 모델: model_id, revision이 settings 값과 같다. 불러온 모델의 이름과, 모델 파일이 들어 있는 스냅샷(커밋)을 본다.
2. 종료 토큰: 모델 generation_config의 eos_token_id 목록이 settings의 eos_token_ids와 같다.
3. 전처리: dev 정상·노이즈 PNG 각각에서, 프로세서가 만든 pixel_values와 PNG를 직접 정규화한 값의 차이 최댓값이
   1e-6 이하다(float32와 모델에 넣는 bf16 모두). pixel_values 모양이 (1, 1, 3, 512, 512)라서 프로세서가 사진 크기를
   다시 바꾸지 않는다.
4. 재현성: ref에 dev 정상 사진 5장을 두 번씩 넣어 두 답이 토큰까지 같다.

하나라도 통과하지 못하면 결과를 적은 뒤 오류로 끝난다. 그때는 2단계로 가지 않는다.
load_ref()가 census·Linear 수·종료 토큰 확인에서 멈추면 결과 파일 없이 끝난다(그것도 통과하지 못한 것이다).
결과의 code_commit에 '-dirty'가 붙지 않도록, 코드를 커밋한 뒤에 돌린다.
"""
import argparse
import hashlib
import json
import sys

from huggingface_hub import try_to_load_from_cache

import runtime as rt
from runtime import settings

GAP_MAX = 1e-6
SHAPE = [1, 1, 3, rt.M["img_size"], rt.M["img_size"]]
# from_pretrained가 읽는 파일. 스냅샷 폴더 이름이 그 파일을 받은 커밋이다
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
               "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json"]


def snapshot_commits():
    """모델 파일마다 들어 있는 캐시 스냅샷의 커밋. 경로는 적지 않는다. 받은 적 없는 파일은 None."""
    out = {}
    for name in MODEL_FILES:
        path = try_to_load_from_cache(rt.M["model_id"], name, revision=rt.M["revision"])
        out[name] = rt.Path(path).parent.name if isinstance(path, str) else None
    return out


def check_model(model):
    commits = snapshot_commits()
    found = {k: v for k, v in commits.items() if v is not None}
    ok = (model.config._name_or_path == rt.M["model_id"]
          and "config.json" in found and "model.safetensors" in found
          and all(v == rt.M["revision"] for v in found.values()))
    return {"pass": ok, "settings_model_id": rt.M["model_id"], "settings_revision": rt.M["revision"],
            "loaded_name_or_path": model.config._name_or_path, "snapshot_commit": commits}


def check_eos(model):
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    return {"pass": sorted(ids) == sorted(rt.EOS), "generation_config": ids, "settings": rt.EOS}


def check_preprocess(processor, paths):
    gaps, shapes = {}, {}
    for kind, by_id in paths.items():
        gaps[kind], shapes[kind] = {}, {}
        for cid, path in by_id.items():
            img = rt.read_png(path)
            enc = processor(text=rt.prompt_text(processor), images=[img], return_tensors="pt")
            shapes[kind][cid] = list(enc["pixel_values"].shape)
            gaps[kind][cid] = rt.pixel_gap(processor, img)
    ok = (all(g <= GAP_MAX for k in gaps.values() for g in k.values())
          and all(s == SHAPE for k in shapes.values() for s in k.values()))
    return {"pass": ok, "gap_max_allowed": GAP_MAX, "expected_shape": SHAPE,
            "gap": gaps, "shape": shapes, "dtypes": ["float32", settings.DTYPE]}


def check_repro(model, processor, clean_paths):
    answers, same = {}, {}
    for cid, path in clean_paths.items():
        img = rt.read_png(path)
        runs = [rt.answer(model, processor, img) for _ in range(2)]
        same[cid] = runs[0]["token_ids"] == runs[1]["token_ids"]
        answers[cid] = [{k: r[k] for k in ("n_tokens", "ended_eos", "hit_cap", "latency_s", "text")}
                        | {"token_ids_sha256": hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()}
                        for r in runs]
    return {"pass": all(same.values()), "same_tokens": same, "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rt.TEAM / "results" / "check_ref.json"))
    cli = ap.parse_args()
    rows = rt.image_rows("dev")
    paths = {kind: {cid: rt.png_path("dev", kind, cid) for _, cid in rows} for kind in ("clean", "noise")}
    missing = [p.name for by_id in paths.values() for p in by_id.values() if not p.exists()]
    assert not missing, f"dev PNG가 없다({len(missing)}장) → prepare_images.py --splits dev를 먼저 실행"

    rt.setup()
    processor = rt.load_processor()
    model = rt.load_ref()   # census·Linear 수·종료 토큰을 확인한다. 통과하지 못하면 여기서 멈춘다
    checks = {"model": check_model(model), "eos": check_eos(model),
              "preprocess": check_preprocess(processor, paths),
              "reproducibility": check_repro(model, processor, paths["clean"])}
    passed = all(c["pass"] for c in checks.values())
    meta = {"environment": rt.environment(),
            # ref는 가중치를 바꾸지 않는다: 바뀐 Linear 0개, 가중치 변화 0(README 5.4의 식에서 대상 부품이 없다)
            "cases": {"ref": {"census": rt.census_record(model), "weight_change_pct": 0.0}},
            "inputs": {"split": "dev", **{f"{kind}_sha256": {cid: rt.sha256(p) for cid, p in by_id.items()}
                                          for kind, by_id in paths.items()}},
            "checks": checks, "passed": passed,
            "note": "README 1-2단계 점검 1~4. checks의 각 pass가 모두 true면 통과"}
    out = rt.Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, c in checks.items():
        print(f"{name}: {'통과' if c['pass'] else '통과하지 못함'}")
    print(f"저장: {out} | {'모두 통과' if passed else '통과하지 못한 점검이 있다 → 2단계로 가지 않는다'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
