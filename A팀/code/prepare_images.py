"""2단계: 사진 준비(README 2단계, 5.3절). COCO JPEG를 받아 정상(clean)·노이즈(noise) PNG를 만들고 해시를 남긴다.

실행: python A팀/code/prepare_images.py                  # dev·cand·acc 전부
      python A팀/code/prepare_images.py --splits dev cand  # 일부만
결과: A팀/data/coco/<id>.jpg, A팀/data/<split>/clean|noise/<id>.png   (data/라 올리지 않는다)
      A팀/results/input-manifest.json                      사진마다 JPEG·clean·noise의 SHA-256
                                                           (PNG는 파일 해시와 픽셀 해시를 함께. PNG 압축 결과는
                                                           Pillow·zlib 버전에 따라 달라질 수 있어 픽셀 해시로 같은 사진인지 본다)

- 이미 있는 파일은 다시 만들지 않는다. 끊기면 같은 명령으로 이어서 한다.
- 정상: JPEG를 RGB로 열어 img_size×img_size로 바로 BICUBIC 리사이즈(자르지 않음).
- 노이즈: 정상 배열에 EPS × 부호(default_rng([NOISE_SEED, coco_id]))를 더하고 0~255로 자른다.
"""
import argparse
import hashlib
import json
import time
import urllib.request

import numpy as np
from PIL import Image

import runtime as rt
from runtime import settings

SPLITS = ("dev", "cand", "acc")
S = rt.M["img_size"]


def download(coco_id, dest, tries=3):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://images.cocodataset.org/val2017/{coco_id}.jpg"
    for k in range(tries):
        try:
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r:
                tmp.write_bytes(r.read())
            Image.open(tmp).verify()
            tmp.replace(dest)
            return
        except Exception as e:            # 네트워크가 잠깐 끊긴 경우 다시 받는다
            if k == tries - 1:
                raise RuntimeError(f"{url} 내려받기 실패: {e}") from e
            time.sleep(3)


def pixel_sha256(path):
    """PNG를 RGB 배열로 읽은 바이트의 SHA-256. 압축 방식과 상관없이 픽셀이 같으면 같다."""
    return hashlib.sha256(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def make_pngs(split, coco_id, jpeg):
    clean_p, noise_p = rt.png_path(split, "clean", coco_id), rt.png_path(split, "noise", coco_id)
    if not clean_p.exists():
        clean_p.parent.mkdir(parents=True, exist_ok=True)
        Image.open(jpeg).convert("RGB").resize((S, S), Image.BICUBIC).save(clean_p, format="PNG")
    if not noise_p.exists():
        noise_p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(Image.open(clean_p).convert("RGB")).astype(np.int16)
        sign = np.random.default_rng([settings.NOISE_SEED, int(coco_id)]).choice(
            np.array([-1, 1], dtype=np.int16), size=arr.shape)
        Image.fromarray(np.clip(arr + settings.EPS * sign, 0, 255).astype(np.uint8)).save(noise_p, format="PNG")
    return clean_p, noise_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--manifest", default=str(rt.TEAM / "results" / "input-manifest.json"))
    cli = ap.parse_args()

    entries = {}
    for split in cli.splits:
        rows = rt.image_rows(split)
        entries[split] = []
        for k, (order, cid) in enumerate(rows, 1):
            jpeg = rt.DATA / "coco" / f"{cid}.jpg"
            download(cid, jpeg)
            clean_p, noise_p = make_pngs(split, cid, jpeg)
            rt.read_png(clean_p)
            entries[split].append({"order": order, "coco_id": cid, "jpeg_sha256": rt.sha256(jpeg),
                                   "clean_sha256": rt.sha256(clean_p), "noise_sha256": rt.sha256(noise_p),
                                   "clean_pixels_sha256": pixel_sha256(clean_p),
                                   "noise_pixels_sha256": pixel_sha256(noise_p)})
            if k % 50 == 0 or k == len(rows):
                print(f"{split}: {k}/{len(rows)}장", flush=True)

    manifest = {"settings_version": settings.SETTINGS_VERSION, "code_commit": rt.git_commit(),
                "model": rt.MODEL_KEY, "img_size": S, "noise": {"eps": settings.EPS, "seed": settings.NOISE_SEED},
                "packages": {"pillow": rt.PIL.__version__, "numpy": np.__version__},
                "counts": {s: len(v) for s, v in entries.items()}, "images": entries}
    out = rt.Path(cli.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out.relative_to(rt.ROOT) if out.is_relative_to(rt.ROOT) else out)


if __name__ == "__main__":
    main()
