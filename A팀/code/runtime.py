"""A팀 실험 공통 코드(README 5장). 2~7단계 코드가 모두 이 파일의 함수로 모델을 올리고 답을 만든다.

- 5.1 실행 환경: setup()이 seed와 결정적 실행 설정을 정한다. CUBLAS_WORKSPACE_CONFIG는 CUDA를 쓰기 전에 정해야
  하므로 이 파일을 import할 때 정한다.
- 5.2 답 생성: answer()
- 5.3 사진: image_rows(), png_path(), read_png()
- 3단계 성공 판정과 4단계 측정이 같은 load_ref()·answer()를 쓴다(README 3단계 '4단계와 같은 코드·환경의 ref').
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv  # noqa: E402
import hashlib  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "A팀"
DATA = TEAM / "data"            # .gitignore의 data/ — 올리지 않는다
sys.path.insert(0, str(ROOT / "공통"))
import quant_rtn  # noqa: E402
import settings  # noqa: E402

MODEL_KEY = "smolvlm"
M = settings.MODELS[MODEL_KEY]
PARTS = M["parts"]
EOS = list(M["eos_token_ids"])
DTYPE = getattr(torch, settings.DTYPE)


def setup():
    """5.1의 seed와 결정적 실행 설정."""
    torch.manual_seed(settings.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "README 5.1: bf16을 지원하는 CUDA GPU에서 실행한다"


def determinism():
    """5.1의 결정적 실행 설정(README 7장 결과 JSON). warn_only가 False면 결정적으로 돌지 않는 연산은 오류로 멈추므로,
    기록이 끝까지 만들어졌다면 '돌지 않은 설정'은 없다."""
    return {"seed": settings.SEED, "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


def image_rows(split):
    """image_list.csv의 split 줄을 order 순서로. [(order, 'coco_id 12자리'), ...]"""
    with open(ROOT / "공통" / "image_list.csv", newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp) if r["split"] == split]
    rows.sort(key=lambda r: int(r["order"]))
    return [(int(r["order"]), f"{int(r['coco_id']):012d}") for r in rows]


def png_path(split, kind, coco_id):
    """2단계가 만드는 정상(clean)·노이즈(noise) PNG의 자리."""
    return DATA / split / kind / f"{coco_id}.png"


def read_png(path):
    img = Image.open(path).convert("RGB")
    assert img.size == (M["img_size"], M["img_size"]), f"{path}: 크기 {img.size}"
    return img


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_processor():
    return AutoProcessor.from_pretrained(M["model_id"], revision=M["revision"], **M["processor_kwargs"])


def census_record(model, targets=""):
    """README 5.4의 경우 확인과, 결과 JSON에 남길 census(부품별 [바뀐 Linear 수, 전체 Linear 수]). ref는 targets=""."""
    count = quant_rtn.census(model, PARTS)
    quant_rtn.check_census(count, targets, settings.PART_NAMES)
    assert {p: c[1] for p, c in count.items()} == M["linear_count"], count
    return {p: list(c) for p, c in count.items()}


def load_ref():
    """ref: bf16 원본(5.4). census로 아무 층도 바뀌지 않았는지와 Linear 수를 확인하고,
    모델 generation_config의 종료 토큰이 settings와 같은지 확인한다(README 1-2단계 점검 2)."""
    model = AutoModelForImageTextToText.from_pretrained(
        M["model_id"], revision=M["revision"], dtype=DTYPE,
        attn_implementation=settings.ATTN_IMPLEMENTATION).to("cuda").eval()
    census_record(model)
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    assert sorted(ids) == sorted(EOS), f"모델의 종료 토큰 {ids}가 settings {EOS}와 다르다(README 1-2단계 점검 2)"
    return model


def prompt_text(processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": settings.PROMPT}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


@torch.no_grad()
def answer(model, processor, img):
    """5.2: 사진 하나와 질문으로 답을 만든다. n_tokens·ended_eos·hit_cap·latency_s·text와,
    새로 만든 토큰 id 목록 token_ids(1-2단계 재현성 점검용. 결과 CSV에는 적지 않는다)."""
    enc = processor(text=prompt_text(processor), images=[img], return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, **settings.GEN_KWARGS, eos_token_id=EOS)
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    ended = bool(new) and new[-1] in EOS
    return {"n_tokens": len(new), "ended_eos": ended,
            "hit_cap": len(new) == settings.MAX_NEW_TOKENS and not ended,
            "latency_s": round(latency, 3),
            "text": processor.tokenizer.decode(new, skip_special_tokens=True), "token_ids": new}


def pixel_gap(processor, img):
    """같은 PNG를 직접 정규화한 값과 프로세서가 만든 pixel_values의 차이 최댓값(README 5.3, 1-2단계 점검 3).
    CPU float32와, 공격 코드처럼 GPU에서 계산해 모델에 넣는 bf16 두 가지를 모두 본다."""
    enc = processor(text=prompt_text(processor), images=[img], return_tensors="pt")
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
    gaps = []
    for device, dtype in (("cpu", torch.float32), ("cuda", DTYPE)):
        mean = torch.tensor(processor.image_processor.image_mean, device=device).view(3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std, device=device).view(3, 1, 1)
        direct = ((x.to(device) - mean) / std).unsqueeze(0).unsqueeze(0).to(dtype)
        reference = enc["pixel_values"].to(device, dtype)
        assert tuple(reference.shape) == tuple(direct.shape), (reference.shape, direct.shape)
        gaps.append((reference.float() - direct.float()).abs().max().item())
    return max(gaps)


def git_commit():
    """결과를 만든 코드의 커밋. 커밋하지 않은 수정이 있으면 '-dirty'를 붙인다."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "공통", "이미지생성", "A팀/code"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    """결과 JSON에 남길 실행 환경(README 7장). 절대 경로·기계 이름은 넣지 않는다."""
    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except ImportError:
        bnb_version = None
    return {"settings_version": settings.SETTINGS_VERSION, "code_commit": git_commit(),
            "model_id": M["model_id"], "revision": M["revision"], "eos_token_ids": EOS,
            "dtype": settings.DTYPE, "attn_implementation": settings.ATTN_IMPLEMENTATION,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "packages": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "bitsandbytes": bnb_version,
                         "pillow": PIL.__version__, "numpy": np.__version__},
            "determinism": determinism()}
