"""1단계: 공통 사진 목록에서 원본(clean)과 노이즈 대조군(noise)을 만든다.

실행: python A팀/code/01_prepare_images.py
결과: A팀/data/clean/<id>.png  (eval 30장, 512×512)
      A팀/data/noise/<id>.png  (같은 30장, 픽셀마다 ±8)
      A팀/data/dev_clean/, dev_noise/  (dev 5장, 코드 확인용)
COCO 원본은 A팀/data/coco_cache/ 에 받아 두고 커밋하지 않는다.
"""
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "공통"))
import settings  # noqa: E402

SIZE = settings.MODELS["smolvlm"]["img_size"]
DATA = ROOT / "A팀" / "data"
CACHE = DATA / "coco_cache"


def read_list(split):
    with open(ROOT / "공통" / "image_list.csv", newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp) if r["split"] == split]
    return sorted(rows, key=lambda r: int(r["order"]))


def download(row):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{int(row['coco_id']):012d}.jpg"
    if not path.exists():
        urllib.request.urlretrieve(row["url"], path)
    return path


def make_noise(clean, coco_id):
    # 규칙 3: 픽셀마다 ±EPS, 부호는 default_rng([NOISE_SEED, coco_id])
    rng = np.random.default_rng([settings.NOISE_SEED, int(coco_id)])
    arr = np.asarray(clean, dtype=np.int16)
    sign = rng.choice(np.array([-1, 1], dtype=np.int16), size=arr.shape)
    return Image.fromarray(np.clip(arr + settings.EPS * sign, 0, 255).astype(np.uint8))


def prepare(split, clean_dir, noise_dir):
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    for row in read_list(split):
        name = f"{int(row['coco_id']):012d}.png"
        if (clean_dir / name).exists() and (noise_dir / name).exists():
            continue
        img = Image.open(download(row)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        noise = make_noise(img, row["coco_id"])
        diff = np.abs(np.asarray(img, dtype=np.int16) - np.asarray(noise, dtype=np.int16)).max()
        assert diff <= settings.EPS, f"{name}: 노이즈 차이 {diff}"
        img.save(clean_dir / name, format="PNG")
        noise.save(noise_dir / name, format="PNG")
    print(f"{split}: {len(list(clean_dir.glob('*.png')))}장 → {clean_dir}, {noise_dir}")


if __name__ == "__main__":
    prepare("eval", DATA / "clean", DATA / "noise")
    prepare("dev", DATA / "dev_clean", DATA / "dev_noise")
