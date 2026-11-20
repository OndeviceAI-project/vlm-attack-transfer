"""1-2단계: B팀 Qwen2-VL ref 점검. 결과: B팀/results/check_ref.json

실행: python B팀/code/prepare_images.py --splits dev   # dev 5장의 PNG가 없을 때만 먼저
      python B팀/code/check_ref.py

2·3단계와 같은 runtime.load_ref()·generate()·pixel_gap()으로 다음을 점검한다.
1. 모델 id와 캐시 스냅샷 revision이 settings 값과 같다.
2. 모델 generation_config의 종료 토큰이 settings와 같다.
3. dev 정상·노이즈 PNG의 전처리 값이 직접 계산한 값과 같고 모양이 (1024, 1176)이다.
4. ref에 dev 정상 사진 5장을 두 번씩 넣은 답이 토큰까지 같다.

하나라도 통과하지 못하면 결과를 적은 뒤 오류로 끝난다. 결과의 code_commit에
'-dirty'가 붙지 않도록 이 파일을 커밋한 뒤 실행한다.
"""
import argparse
import hashlib
import json
import sys

from huggingface_hub import try_to_load_from_cache

import runtime as rt
from runtime import settings

GAP_MAX = 1e-6
SHAPE = [1024, 1176]
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
]
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}


def snapshot_commits():
    """모델 파일별 캐시 스냅샷 커밋. 절대 경로는 결과에 적지 않는다."""
    out = {}
    for name in MODEL_FILES:
        path = try_to_load_from_cache(rt.M["model_id"], name, revision=rt.M["revision"])
        out[name] = rt.Path(path).parent.name if isinstance(path, str) else None
    return out


def check_model(model):
    commits = snapshot_commits()
    found = {k: v for k, v in commits.items() if v is not None}
    ok = (
        model.config._name_or_path == rt.M["model_id"]
        and REQUIRED_MODEL_FILES.issubset(found)
        and all(v == rt.M["revision"] for v in found.values())
    )
    return {
        "pass": ok,
        "settings_model_id": rt.M["model_id"],
        "settings_revision": rt.M["revision"],
        "loaded_name_or_path": model.config._name_or_path,
        "required_files": sorted(REQUIRED_MODEL_FILES),
        "snapshot_commit": commits,
    }


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
            enc = rt.encode(processor, img)
            shapes[kind][cid] = list(enc["pixel_values"].shape)
            gaps[kind][cid] = rt.pixel_gap(processor, img)
    ok = (
        all(g <= GAP_MAX for by_id in gaps.values() for g in by_id.values())
        and all(s == SHAPE for by_id in shapes.values() for s in by_id.values())
    )
    return {
        "pass": ok,
        "gap_max_allowed": GAP_MAX,
        "expected_shape": SHAPE,
        "gap": gaps,
        "shape": shapes,
        "dtypes": ["float32", settings.DTYPE],
    }


def check_repro(model, processor, clean_paths):
    answers, same = {}, {}
    for cid, path in clean_paths.items():
        img = rt.read_png(path)
        runs = []
        token_runs = []
        for _ in range(2):
            result, token_ids = rt.generate(model, processor, img)
            token_runs.append(token_ids)
            runs.append(
                result
                | {"token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()}
            )
        same[cid] = token_runs[0] == token_runs[1]
        answers[cid] = runs
    return {"pass": all(same.values()), "same_tokens": same, "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rt.TEAM / "results" / "check_ref.json"))
    cli = ap.parse_args()
    rows = rt.image_rows("dev")
    paths = {
        kind: {cid: rt.png_path("dev", kind, cid) for _, cid in rows}
        for kind in ("clean", "noise")
    }
    missing = [p.name for by_id in paths.values() for p in by_id.values() if not p.exists()]
    assert not missing, f"dev PNG가 없다({len(missing)}장) → prepare_images.py --splits dev를 먼저 실행"

    rt.setup()
    processor = rt.load_processor()
    model = rt.load_ref()
    checks = {
        "model": check_model(model),
        "eos": check_eos(model),
        "preprocess": check_preprocess(processor, paths),
        "reproducibility": check_repro(model, processor, paths["clean"]),
    }
    passed = all(c["pass"] for c in checks.values())
    meta = {
        "environment": rt.environment(),
        "cases": {"ref": {"census": rt.census_record(model), "weight_change_pct": 0.0}},
        "inputs": {
            "split": "dev",
            **{
                f"{kind}_sha256": {cid: rt.sha256(path) for cid, path in by_id.items()}
                for kind, by_id in paths.items()
            },
        },
        "checks": checks,
        "passed": passed,
        "note": "README 1-2단계 점검 1~4. checks의 각 pass가 모두 true면 통과",
    }
    out = rt.Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, check in checks.items():
        print(f"{name}: {'통과' if check['pass'] else '통과하지 못함'}")
    print(f"저장: {out} | {'모두 통과' if passed else '통과하지 못한 점검이 있다 → 2단계로 가지 않는다'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
