"""2단계: 원본 모델로 공격 사진(adv)을 만든다 (README '반드시 지킬 것' 4·5번, 이미지생성/README).

실행: python A팀/code/02_make_adv.py              # eval 30장 → A팀/adv/
      python A팀/code/02_make_adv.py --split dev  # dev 5장 → A팀/adv_dev/ (코드 확인·공격 설정 조정용)
결과: <out>/<id>.png          공격 사진 (512×512, 무손실 PNG)
      <out>/attack_log.csv    사진마다 한 줄: image_id, iters_run, seconds, start_len, best_len, saved_len, linf
      <out>/attack_meta.json  settings 버전, 인자, 공격 코드 해시, GPU, 패키지 버전
      <out>/SHA256SUMS        목록의 사진이 모두 끝나면 쓴다 (shasum -a 256 *.png 와 같은 형식)

- 공격은 이미지생성/attack_smolvlm.py의 attack_one을 그대로 부른다. 반복문·저장 방식 모두 그 파일 그대로다:
  답 길이를 잰 시점(원본 포함) 중 (종료 없이 512, 답 길이)가 가장 큰 사진을 저장하고, 종료 없이 512가 두 번 연속이면
  조기 종료, 원본 답이 이미 512면 공격하지 않는다(이미지생성/README '예비 실험 코드에서 바뀐 것' 2).
  이 파일이 더하는 것은 그 스크립트에 없는 것뿐이다: 사진마다 로그, 저장본 다시 읽어 확인, SHA256SUMS.
- 고정값은 공통/settings.py에서 읽는다. 프로세서는 settings의 processor_kwargs(size longest_edge 512)로 만들어 넘긴다.
  그래서 답 길이 확인과 조기 종료도 평가가 보는 것과 같은 사진으로 한다(규칙 5).
- 이미 PNG가 있는 사진은 건너뛴다. 덮어쓰지 않는다. 다시 만들어야 하면 --out으로 새 폴더를 준다(규칙 4).
- 답이 안 길어진 사진도 저장하고 로그에 적는다(이미지생성/README 4번).
- 로그의 best_len은 attack_one이 출력하는 '고른 사진'의 답 길이, saved_len은 '저장본 실제 답 길이'다(이미지생성/README 2번).
  attack_one이 '주의'(저장본이 고를 때와 다름)를 출력했거나, 저장한 PNG를 다시 읽어 잰 길이가 saved_len과 다르면 화면에 알린다.
- seconds는 attack_one 한 번(시작 답 길이 재기부터 저장본 답 길이 재기까지)에 걸린 시간이다.
- 폴더에 이어서 만들 때는 settings 버전·인자·공격 코드(attack_smolvlm.py 해시)가 그 폴더를 만들 때와 같아야 한다.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "공통"))
sys.path.insert(0, str(ROOT / "이미지생성"))
import attack_smolvlm as atk  # noqa: E402
import settings  # noqa: E402

M = settings.MODELS["smolvlm"]
TEAM = ROOT / "A팀"
SPLITS = {"eval": (TEAM / "data" / "clean", TEAM / "adv"), "dev": (TEAM / "data" / "dev_clean", TEAM / "adv_dev")}
LOG_COLUMNS = ["image_id", "iters_run", "seconds", "start_len", "best_len", "saved_len", "linf"]
ATTACK_CODE = ROOT / "이미지생성" / "attack_smolvlm.py"


def read_ids(split):
    with open(ROOT / "공통" / "image_list.csv", newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp) if r["split"] == split]
    return [f"{int(r['coco_id']):012d}" for r in sorted(rows, key=lambda r: int(r["order"]))]


def check_same_pixels(processor, prompt, pil, direct):
    """규칙 5: 공격 코드가 모델에 넣는 값과 프로세서가 같은 사진으로 만드는 값의 차이."""
    enc = processor(text=prompt, images=[pil], return_tensors="pt")
    assert tuple(enc["pixel_values"].shape) == tuple(direct.shape), (enc["pixel_values"].shape, direct.shape)
    return (enc["pixel_values"] - direct.cpu()).abs().max().item()


@torch.no_grad()
def answer_len(model, processor, prompt, pil, device):
    enc = processor(text=prompt, images=[pil], return_tensors="pt").to(device)
    out = model.generate(**enc, **settings.GEN_KWARGS)
    return out.shape[1] - enc["input_ids"].shape[1]


# attack_one이 화면에 찍는 줄에서 로그에 쓸 값을 읽는다 (그 함수는 값을 돌려주지 않는다)
RE_START = re.compile(r"시작 자연 답 길이: (\d+) 토큰")
RE_CHECK = re.compile(r"it\s+(\d+) \| eos_logit \S+ \| 자연 답 길이 (\d+) 토큰")
RE_NAN = re.compile(r"it (\d+): nan 발생")
RE_BEST = re.compile(r"고른 사진: (?:원본 그대로|it\s+\d+) \| 답 길이 (\d+) 토큰")
RE_SAVED = re.compile(r"저장본 실제 답 길이: (\d+) 토큰")


class Tee(io.TextIOBase):
    """화면에 그대로 내보내면서 내용을 모아 둔다."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()


def attack(model, processor, clean_path, args, device, mean, std, eos_ids):
    """공유 스크립트의 attack_one으로 사진 한 장을 공격하고 저장한다.
    (시작 길이, 고른 사진의 길이, 저장본 길이, 돈 반복 수, 걸린 초, '주의' 출력 여부)를 돌려준다."""
    tee = Tee(sys.stdout)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        atk.attack_one(model, processor, str(clean_path), args, device, mean, std, eos_ids)
    seconds = time.perf_counter() - t0
    text = "".join(tee.parts)
    start, best, saved = RE_START.search(text), RE_BEST.search(text), RE_SAVED.search(text)
    checks = RE_CHECK.findall(text)
    assert start and best and saved, "attack_one의 출력 형식이 바뀌었다 → 02_make_adv.py의 정규식을 고칠 것"
    nan = RE_NAN.search(text)
    if "공격하지 않고" in text:     # 원본 답이 이미 512라 한 걸음도 움직이지 않았다
        iters_run = 0
    elif nan:                     # it번째에서 멈췄으므로 그 앞까지 it번 움직였다
        iters_run = int(nan.group(1))
    elif "조기 종료" in text:      # 확인한 걸음에서 멈췄다
        iters_run = int(checks[-1][0]) + 1
    else:                         # 끝까지 돌았다. 마지막 걸음도 확인하므로 마지막 확인 줄이 it = iter-1 이어야 한다
        assert checks and int(checks[-1][0]) == args.iter - 1, f"마지막 확인이 it {checks[-1][0] if checks else '-'}"
        iters_run = args.iter
    warned = "주의: 저장본 답 길이가 고를 때와 다르다" in text
    return int(start.group(1)), int(best.group(1)), int(saved.group(1)), iters_run, seconds, warned


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(out_dir):
    """attack_log.csv를 사진 번호 → 줄로 읽는다. 없으면 빈 dict."""
    path = out_dir / "attack_log.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fp:
        return {r["image_id"]: r for r in csv.DictReader(fp)}


def write_sums(out_dir, ids):
    """SHA256SUMS를 쓴다. 이미 있으면 고치지 않고 지금 파일과 같은지만 본다."""
    names = [f"{i}.png" for i in ids]
    lines = [f"{sha256(out_dir / n)}  {n}\n" for n in sorted(names)]
    path = out_dir / "SHA256SUMS"
    if path.exists():
        same = path.read_text(encoding="utf-8").splitlines(keepends=True) == lines
        print("SHA256SUMS: 이미 있음 →", "지금 파일과 일치" if same else "지금 파일과 다르다! 공격 사진이 바뀌었는지 확인할 것")
        return
    path.write_text("".join(lines), encoding="utf-8")
    print(f"SHA256SUMS: {len(lines)}장 기록 → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="eval")
    ap.add_argument("--out", default=None, help="출력 폴더. 다시 만들 때는 새 폴더를 준다")
    ap.add_argument("--iter", type=int, default=settings.ATTACK_ITERS)
    ap.add_argument("--limit", type=int, default=0, help="목록 앞에서부터 몇 장만(0이면 전부). 코드 확인용")
    cli = ap.parse_args()
    assert 1 <= cli.iter <= settings.ATTACK_MAX_ITERS, f"반복은 {settings.ATTACK_MAX_ITERS} 이하 (settings.ATTACK_MAX_ITERS)"
    clean_dir, default_out = SPLITS[cli.split]
    out_dir = Path(cli.out) if cli.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(iter=cli.iter, eps=settings.EPS, step=settings.STEP, target_len=settings.TARGET_LEN,
                           refresh=settings.REFRESH, w_unc=settings.W_UNC, w_div=settings.W_DIV, out=str(out_dir))

    ids = read_ids(cli.split)
    ids = ids[:cli.limit] if cli.limit else ids
    missing = [i for i in ids if not (clean_dir / f"{i}.png").exists()]
    assert not missing, f"{clean_dir}에 없는 사진 {len(missing)}장 → 01_prepare_images.py를 먼저 실행할 것"
    todo = [i for i in ids if not (out_dir / f"{i}.png").exists()]
    print(f"{cli.split} {len(ids)}장 중 새로 만들 사진 {len(todo)}장 → {out_dir}")

    # 이 폴더가 지금 설정으로, 이 스크립트로 만든 것인지 확인한다 (한 폴더 = 한 세트, 규칙 4)
    logged = read_log(out_dir)
    orphans = [i for i in ids if (out_dir / f"{i}.png").exists() and i not in logged]
    assert not orphans, (f"로그에 없는 공격 사진 {len(orphans)}장: {orphans[:3]} → 중간에 끊긴 실행의 흔적이면 그 PNG를 지우고 "
                         f"다시 실행한다. 다른 방법으로 만든 사진이면 새 폴더(--out)를 쓴다")
    want = {"settings_version": settings.SETTINGS_VERSION,
            "args": {k: v for k, v in vars(args).items() if k != "out"},
            "attack_code_sha256": sha256(ATTACK_CODE)}
    meta_path = out_dir / "attack_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        had = {"settings_version": old.get("settings_version"),
               "args": {k: v for k, v in old.get("args", {}).items() if k != "out"},
               "attack_code_sha256": old.get("attack_code_sha256")}
        assert had == want, (f"이 폴더는 다른 설정이나 다른 공격 코드로 만들던 것이다: {had} → 지금 {want}. "
                             f"새 폴더(--out)를 쓴다")
    log_path = out_dir / "attack_log.csv"
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as fp:
            header = next(csv.reader(fp), None)
        assert header == LOG_COLUMNS, f"{log_path}의 열이 지금과 다르다: {header} → 새 폴더(--out)를 쓴다"

    # 공유 스크립트의 상수가 settings와 같은지 (다르면 두 파일 중 하나가 바뀐 것)
    assert atk.MODEL_ID == M["model_id"] and atk.QUESTION == settings.PROMPT and atk.IMG_SIZE == M["img_size"]

    if todo:
        device = atk.pick_device()
        processor = AutoProcessor.from_pretrained(M["model_id"], **M["processor_kwargs"])
        model = AutoModelForImageTextToText.from_pretrained(
            M["model_id"], dtype=getattr(torch, M["attack_dtype"])).to(device).eval()
        mean = torch.tensor(processor.image_processor.image_mean, device=device).view(3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std, device=device).view(3, 1, 1)
        gen_eos = GenerationConfig.from_pretrained(M["model_id"]).eos_token_id
        eos_ids = sorted(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        assert eos_ids == atk.eos_token_ids(processor), (eos_ids, atk.eos_token_ids(processor))
        prompt = atk.build_prompt(processor)
        print("device:", device, "| eos ids:", eos_ids, "| 반복:", args.iter)

        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "settings_version": settings.SETTINGS_VERSION, "model": "smolvlm", "model_id": M["model_id"],
                "split": cli.split, "attack_dtype": M["attack_dtype"], "processor_kwargs": M["processor_kwargs"],
                "args": vars(args),
                "attack_code": "이미지생성/attack_smolvlm.py의 attack_one (잰 시점 중 가장 높은 사진 저장)",
                "attack_code_sha256": want["attack_code_sha256"],
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device),
                "packages": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "pillow": PIL.__version__, "numpy": np.__version__},
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        new_log = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            if new_log:
                w.writerow(LOG_COLUMNS)
            for k, image_id in enumerate(todo, 1):
                clean_path, out_path = clean_dir / f"{image_id}.png", out_dir / f"{image_id}.png"
                start_len, best_len, saved_len, iters_run, seconds, warned = attack(
                    model, processor, clean_path, args, device, mean, std, eos_ids)

                # 저장본을 다시 읽어 확인한다 (규칙 4·5)
                saved = Image.open(out_path).convert("RGB")
                clean = Image.open(clean_path).convert("RGB")
                assert saved.size == clean.size == (M["img_size"], M["img_size"]), (saved.size, clean.size)
                linf = int(np.abs(np.asarray(saved, dtype=np.int16) - np.asarray(clean, dtype=np.int16)).max())
                x = torch.from_numpy(np.asarray(saved, dtype=np.float32) / 255).permute(2, 0, 1).to(device)
                diff = check_same_pixels(processor, prompt, saved, atk.make_pixel_values(x, torch.zeros_like(x), mean, std, 1))
                assert diff < 1e-5, f"{image_id}: 공격 코드가 넣는 값과 프로세서가 만든 값의 차이 {diff:.2e} (규칙 5)"
                reread_len = answer_len(model, processor, prompt, saved, device)
                w.writerow([image_id, iters_run, round(seconds, 1), start_len, best_len, saved_len, linf])
                fp.flush()
                flag = "" if linf <= settings.EPS else f"  ※ 최대 차이가 {settings.EPS}를 넘는다 → 이 세트는 쓰지 말 것"
                if warned:
                    flag += f"  ※ 저장본({saved_len})이 고를 때({best_len})와 다르다 → 이 사진은 따로 확인"
                if reread_len != saved_len:
                    flag += f"  ※ PNG를 다시 읽어 잰 길이는 {reread_len} 토큰으로 다르다"
                print(f"[{image_id}] 저장본 답 길이 {saved_len} 토큰, 원본과의 최대 차이 {linf}{flag}  ({k}/{len(todo)}장)")

    log = read_log(out_dir)
    done = [i for i in ids if i in log and (out_dir / f"{i}.png").exists()]
    if done:
        hit = sum(int(log[i]["saved_len"]) >= settings.MAX_NEW_TOKENS for i in done)
        worst = max(int(log[i]["linf"]) for i in done)
        differ = [i for i in done if log[i]["best_len"] != log[i]["saved_len"]]
        print(f"\n완료 {len(done)}/{len(ids)}장 | 저장본이 {settings.MAX_NEW_TOKENS}토큰에 닿은 사진 {hit}장 | 원본과의 최대 차이 {worst} (상한 {settings.EPS})"
              + (f" | best_len과 saved_len이 다른 사진 {len(differ)}장: {differ}" if differ else ""))
    if cli.limit:
        print("SHA256SUMS: --limit 실행이라 쓰지 않음")
    elif len(done) < len(ids):
        print(f"SHA256SUMS: 아직 {len(done)}/{len(ids)}장이라 쓰지 않음")
    else:
        write_sums(out_dir, ids)


if __name__ == "__main__":
    main()
