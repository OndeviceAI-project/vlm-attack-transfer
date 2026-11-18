"""B팀 실험 공통 코드(README 5장). Qwen2-VL-2B. 2~7단계 코드가 모두 이 파일의 함수로 모델을 올리고 답을 만든다.

v1 B팀 코드(태그 v1-midterm의 B팀/src/runtime.py)의 Qwen 전처리·생성을 v2 규칙에 맞게 옮겼다.
- 사진 텐서는 0~255 RGB 단위로 다룬다(raw_pixels, to_image). 노이즈 상한 EPS도 같은 단위다.
- 5.1 실행 환경: setup(). CUBLAS_WORKSPACE_CONFIG는 CUDA를 쓰기 전에 정해야 하므로 import할 때 정한다.
- 5.2 답 생성: generate()·answer()
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
TEAM = ROOT / "B팀"
DATA = TEAM / "data"            # .gitignore의 data/ — 올리지 않는다
sys.path.insert(0, str(ROOT / "공통"))
import quant_rtn  # noqa: E402
import settings  # noqa: E402

csv.field_size_limit(10_000_000)   # 긴 답이 CSV 기본 필드 한도를 넘을 수 있다
MODEL_KEY = "qwen2vl"
M = settings.MODELS[MODEL_KEY]
PARTS = M["parts"]
EOS = list(M["eos_token_ids"])
DTYPE = getattr(torch, settings.DTYPE)
GRID = [[1, 32, 32]]               # 448×448 사진의 image_grid_thw


def setup():
    """5.1의 seed와 결정적 실행 설정."""
    torch.manual_seed(settings.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "README 5.1: bf16을 지원하는 CUDA GPU"


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
    with Image.open(path) as img:
        img = img.convert("RGB")
        assert img.size == (M["img_size"], M["img_size"]), f"{path}: 크기 {img.size}"
        return img.copy()


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


def load_ref(attack=False):
    """ref: bf16 원본(5.4). census로 아무 층도 바뀌지 않았는지와 Linear 수를 확인하고,
    모델 generation_config의 종료 토큰이 settings와 같은지 확인한다(README 1-2단계 점검 2).
    attack=True면 가중치의 기울기를 끈다(공격은 사진 노이즈의 기울기만 쓴다)."""
    model = AutoModelForImageTextToText.from_pretrained(
        M["model_id"], revision=M["revision"], dtype=DTYPE,
        attn_implementation=settings.ATTN_IMPLEMENTATION).to("cuda").eval()
    census_record(model)
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    assert sorted(ids) == sorted(EOS), f"모델의 종료 토큰 {ids}가 settings {EOS}와 다르다(README 1-2단계 점검 2)"
    if attack:
        model.requires_grad_(False)
    return model


def prompt_text(processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": settings.PROMPT}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def encode(processor, img):
    enc = processor(text=prompt_text(processor), images=[img], return_tensors="pt")
    assert enc["image_grid_thw"].tolist() == GRID, f"예상과 다른 사진 격자: {enc['image_grid_thw'].tolist()}"
    return enc


@torch.no_grad()
def generate(model, processor, img, overrides=None):
    """5.2: (n_tokens·ended_eos·hit_cap·latency_s·text, 새 토큰 목록). overrides는 공격의 목표 문장 생성에만 쓴다."""
    enc = encode(processor, img).to("cuda")
    enc["pixel_values"] = enc["pixel_values"].to(DTYPE)
    kwargs = {**settings.GEN_KWARGS, "eos_token_id": EOS, **(overrides or {})}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, **kwargs)
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    ended = bool(new) and new[-1] in EOS
    result = {"n_tokens": len(new), "ended_eos": ended,
              "hit_cap": len(new) == kwargs["max_new_tokens"] and not ended,
              "latency_s": round(latency, 3), "text": processor.tokenizer.decode(new, skip_special_tokens=True)}
    return result, new


def answer(model, processor, img):
    """5.2의 답 한 줄."""
    return generate(model, processor, img)[0]


def raw_pixels(img, device="cpu"):
    """PIL 사진 → (3, H, W) float 텐서, 0~255 단위."""
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().to(device)


def to_image(pixels):
    """0~255 텐서 → 반올림한 uint8 PIL 사진(PNG로 저장할 픽셀)."""
    arr = pixels.detach().clamp(0, 255).round().to("cpu").permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def pixel_values(processor, pixels):
    """프로세서와 같은 계산(정규화 → 조각 배열)을 미분 가능한 텐서로. 448×448 → (1024, 1176)."""
    ip = processor.image_processor
    normalized = ip.rescale_and_normalize(pixels.unsqueeze(0), do_rescale=ip.do_rescale,
                                          rescale_factor=ip.rescale_factor, do_normalize=ip.do_normalize,
                                          image_mean=ip.image_mean, image_std=ip.image_std)
    return ip.patchify(normalized, patch_size=ip.patch_size, merge_size=ip.merge_size,
                       temporal_patch_size=ip.temporal_patch_size)[0].squeeze(0)


def pixel_gap(processor, img):
    """같은 PNG를 프로세서가 만든 값과 직접 계산한 값의 차이 최댓값(README 5.3, 1-2단계 점검 3).
    CPU float32와, 모델에 넣는 bf16 두 가지를 모두 본다."""
    reference = encode(processor, img)["pixel_values"]
    cpu = pixel_values(processor, raw_pixels(img))
    assert tuple(cpu.shape) == (1024, 1176), cpu.shape
    gap_cpu = (cpu - reference).abs().max().item()
    gap_bf16 = (pixel_values(processor, raw_pixels(img, "cuda")).to(DTYPE)
                - reference.to("cuda", DTYPE)).abs().max().item()
    return max(gap_cpu, gap_bf16)


def git_commit():
    """결과를 만든 코드의 커밋. 커밋하지 않은 수정이 있으면 '-dirty'를 붙인다."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "공통", "B팀/code"],
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
