"""3단계: 경우마다 모델을 올려 같은 사진 × 입력 3종(clean, noise, adv)의 답을 잰다 (README 파이프라인 3, 규칙 1·6·7·9).

실행: python A팀/code/03_transfer.py --name smolvlm_phase1     # 1차 경우 8개 × eval 30장 × 입력 3종 = 720줄
      python A팀/code/03_transfer.py --name check --split dev --cases ref rtn-V-4 bnb-V-4   # 코드 확인용 (dev 5장)
결과: A팀/results/<name>.csv   답 하나가 한 줄. 열은 README '결과 형식' 그대로
      A팀/results/<name>.json  settings 버전, GPU, 패키지 버전, 경우별 census와 가중치 변화

- 경우 이름: ref(bf16 원본) / rtn-<부품>-<비트>(공통/quant_rtn.apply_rtn) / bnb-<부품>-<비트>(bitsandbytes 8=LLM.int8, 4=nf4).
  부품은 settings.MODELS의 parts 글자(V, C, L, H)를 이어 쓴 것이다. 예: rtn-V-4, rtn-VCLH-3.
- 경우마다 양자화 뒤 quant_rtn.census로 의도한 부품만 바뀌었는지 확인하고, 아니면 멈춘다(규칙 6).
- ref는 같은 실행에 반드시 들어간다(규칙 7). 질문·생성 설정은 settings.PROMPT와 settings.GEN_KWARGS 그대로(규칙 1).
- 실행 하나에 CSV 하나. 도중에 끊기면 같은 명령을 다시 실행한다: 이미 쓴 줄은 건너뛰고 이어서 쓴다.
  처음부터 다시 재려면 --name을 바꾸거나 CSV와 JSON을 지운다. CSV는 손으로 고치지 않는다(규칙 9).
"""
import argparse
import csv
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "공통"))
import quant_rtn  # noqa: E402
import settings  # noqa: E402

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)   # bnb 8비트가 층마다 찍는 dtype 안내를 숨긴다

MODEL_KEY = "smolvlm"
M = settings.MODELS[MODEL_KEY]
TEAM = ROOT / "A팀"
# split → (clean 폴더, noise 폴더, adv 폴더)
SPLITS = {"eval": (TEAM / "data" / "clean", TEAM / "data" / "noise", TEAM / "adv"),
          "dev": (TEAM / "data" / "dev_clean", TEAM / "data" / "dev_noise", TEAM / "adv_dev")}
INPUTS = ["clean", "noise", "adv"]
COLUMNS = ["model", "case", "image_id", "input", "n_tokens", "hit_cap", "latency_s", "text"]


def read_ids(split):
    with open(ROOT / "공통" / "image_list.csv", newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp) if r["split"] == split]
    return [f"{int(r['coco_id']):012d}" for r in sorted(rows, key=lambda r: int(r["order"]))]


def parse_case(case):
    """'ref' → ('ref', '', None), 'rtn-V-4' → ('rtn', 'V', 4), 'bnb-V-8' → ('bnb', 'V', 8)"""
    if case == "ref":
        return "ref", "", None
    method, targets, bits = case.split("-")
    assert method in ("rtn", "bnb"), f"모르는 경우 이름: {case}"
    assert targets and all(p in M["parts"] for p in targets), f"모르는 부품: {case}"
    assert method == "rtn" or int(bits) in (8, 4), f"bitsandbytes는 8과 4만 된다: {case}"
    return method, targets, int(bits)


def load_case(case, device):
    """경우의 모델을 올리고 (모델, census, 가중치 변화 %)를 돌려준다. 의도한 부품만 양자화되지 않았으면 멈춘다."""
    method, targets, bits = parse_case(case)
    parts, change = M["parts"], None
    if method == "bnb":
        assert device == "cuda", "bitsandbytes 경우는 CUDA에서만 된다"
        skip = [prefix for p, prefix in parts.items() if p not in targets]   # 대상이 아닌 부품은 건너뛴다
        if bits == 8:
            cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        else:
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip)
        model = AutoModelForImageTextToText.from_pretrained(
            M["model_id"], quantization_config=cfg, dtype=torch.bfloat16, device_map=device).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(M["model_id"], dtype=torch.bfloat16).to(device).eval()
        if method == "rtn":
            change = quant_rtn.apply_rtn(model, parts, targets, bits, group=settings.QUANT_GROUP)
    count = quant_rtn.census(model, parts)
    quant_rtn.check_census(count, targets, settings.PART_NAMES)
    return model, count, change


@torch.no_grad()
def answer(model, processor, prompt, path, eos_ids, device):
    """(n_tokens, hit_cap, latency_s, text). hit_cap = 종료 토큰 없이 상한에 닿았는가."""
    img = Image.open(path).convert("RGB")
    assert img.size == (M["img_size"], M["img_size"]), f"{path}: 크기 {img.size}"
    enc = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**enc, **settings.GEN_KWARGS)
    if device == "cuda":
        torch.cuda.synchronize()
    latency = time.time() - t0
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    hit_cap = len(new) >= settings.MAX_NEW_TOKENS and not any(t in eos_ids for t in new)
    text = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
    return len(new), hit_cap, round(latency, 3), text


def check_adv_sums(adv_dir):
    """adv 폴더에 SHA256SUMS가 있으면 지금 파일과 같은지 확인한다(공격 사진은 고정, 규칙 4)."""
    path = adv_dir / "SHA256SUMS"
    if not path.exists():
        return "SHA256SUMS 없음"
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        actual = hashlib.sha256((adv_dir / name).read_bytes()).hexdigest()
        assert actual == digest, f"{adv_dir / name}: SHA256SUMS와 다르다 → 공격 사진이 바뀌었다"
    return "SHA256SUMS와 일치"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="결과 파일 이름 (A팀/results/<name>.csv)")
    ap.add_argument("--split", choices=list(SPLITS), default="eval")
    ap.add_argument("--cases", nargs="+", default=settings.CASES_PHASE1)
    ap.add_argument("--adv", default=None, help="공격 사진 폴더 (기본: split에 따라 A팀/adv 또는 A팀/adv_dev)")
    ap.add_argument("--outdir", default=str(TEAM / "results"))
    ap.add_argument("--limit", type=int, default=0, help="목록 앞에서부터 몇 장만(0이면 전부). 코드 확인용")
    cli = ap.parse_args()
    assert "ref" in cli.cases, "ref는 같은 실행에 반드시 포함한다 (규칙 7)"
    for c in cli.cases:
        parse_case(c)

    clean_dir, noise_dir, adv_dir = SPLITS[cli.split]
    adv_dir = Path(cli.adv) if cli.adv else adv_dir
    dirs = {"clean": clean_dir, "noise": noise_dir, "adv": adv_dir}
    ids = read_ids(cli.split)
    ids = ids[:cli.limit] if cli.limit else ids
    for kind, d in dirs.items():
        missing = [i for i in ids if not (d / f"{i}.png").exists()]
        assert not missing, f"{d}에 없는 {kind} 사진 {len(missing)}장: {missing[:3]}"
    adv_check = check_adv_sums(adv_dir)
    assert not (cli.split == "eval" and not cli.limit and adv_check == "SHA256SUMS 없음"), (
        f"{adv_dir}에 SHA256SUMS가 없다 → 공격 사진 30장을 고정한 뒤 실행한다 "
        f"(02_make_adv.py가 30장을 마치면 써 준다. 직접 만들었으면 그 폴더에서 shasum -a 256 *.png > SHA256SUMS)")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(M["model_id"], **M["processor_kwargs"])
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": settings.PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    gen_eos = GenerationConfig.from_pretrained(M["model_id"]).eos_token_id
    eos_ids = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])   # generation_config의 종료 토큰 전부

    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = outdir / f"{cli.name}.csv", outdir / f"{cli.name}.json"
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fp:
            done = {(r["case"], r["image_id"], r["input"]) for r in csv.DictReader(fp)}
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(COLUMNS)
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {"cases": {}}
    adv_label = str(adv_dir.relative_to(ROOT)) if adv_dir.is_relative_to(ROOT) else str(adv_dir)
    now = {"settings_version": settings.SETTINGS_VERSION, "split": cli.split, "n_images": len(ids), "adv_dir": adv_label}
    if "split" in meta:
        was = {k: meta.get(k) for k in now}
        assert was == now, f"{csv_path.name}은 다른 조건의 실행이다: {was} → 지금 {now}. --name을 바꿔 새 파일로 실행한다"
    assert not (done and "split" not in meta), f"{csv_path.name}은 있는데 {json_path.name}이 없다 → --name을 바꿔 새 파일로 실행한다"
    meta.update({
        "settings_version": settings.SETTINGS_VERSION, "model": MODEL_KEY, "model_id": M["model_id"],
        "split": cli.split, "n_images": len(ids), "inputs": INPUTS, "prompt": settings.PROMPT,
        "gen_kwargs": settings.GEN_KWARGS, "eos_token_ids": sorted(eos_ids), "processor_kwargs": M["processor_kwargs"],
        "quant_group": settings.QUANT_GROUP, "adv_dir": adv_label,
        "adv_check": adv_check, "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
        "packages": {"torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__,
                     "numpy": np.__version__, "bitsandbytes": __import__("bitsandbytes").__version__ if device == "cuda" else None},
    })
    print(f"{cli.split} {len(ids)}장 × 입력 {len(INPUTS)}종 × 경우 {len(cli.cases)}개 = {len(ids) * len(INPUTS) * len(cli.cases)}줄 → {csv_path}")
    print(f"공격 사진: {adv_dir} ({adv_check}) | 장치: {device} | settings {settings.SETTINGS_VERSION}")

    for case in cli.cases:
        todo = [(i, k) for i in ids for k in INPUTS if (case, i, k) not in done]
        if not todo:
            print(f"=== {case} | 이미 모두 측정됨 → 건너뜀")
            continue
        t_case = time.time()
        model, count, change = load_case(case, device)
        print(f"=== {case} | 양자화된 Linear: " + ", ".join(f"{settings.PART_NAMES[p]} {q}/{t}" for p, (q, t) in count.items())
              + (" | 가중치 변화 " + ", ".join(f"{p} {v:.2f}%" for p, v in change.items()) if change else "")
              + f" | 새로 잴 답 {len(todo)}개")
        with open(csv_path, "a", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            for n, (image_id, kind) in enumerate(todo, 1):
                n_tokens, hit_cap, latency, text = answer(model, processor, prompt, dirs[kind] / f"{image_id}.png", eos_ids, device)
                w.writerow([MODEL_KEY, case, image_id, kind, n_tokens, hit_cap, latency, text])
                fp.flush()
                if n % 30 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}")
        meta["cases"][case] = {"census": {p: list(v) for p, v in count.items()}, "weight_change_pct": change,
                               "seconds": round(time.time() - t_case, 1)}
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(csv_path, newline="", encoding="utf-8") as fp:
        n_rows = sum(1 for _ in csv.DictReader(fp))
    meta["n_rows"] = n_rows
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {n_rows}줄 → {csv_path}\n메모: {json_path}")


if __name__ == "__main__":
    main()
