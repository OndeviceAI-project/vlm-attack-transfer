"""0단계: 공식 실험 전 확인 세 가지 (README '반드시 지킬 것' 5·6번과 재현성).

실행: python A팀/code/00_check.py
1) 전처리 일치: 사진을 직접 정규화한 값과 프로세서가 만든 pixel_values의 차이 = 0
2) census: 모든 Linear가 V/C/L/H 중 하나에 속하고, ref는 양자화된 층이 0개
3) 재현성: 같은 사진을 두 번 넣으면 같은 답
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "공통"))
import quant_rtn  # noqa: E402
import settings  # noqa: E402

M = settings.MODELS["smolvlm"]
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoProcessor.from_pretrained(M["model_id"], **M["processor_kwargs"])
img = Image.open(next((ROOT / "A팀" / "data" / "dev_clean").glob("*.png"))).convert("RGB")

# 1) 전처리 일치
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": settings.PROMPT}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
enc = proc(text=prompt, images=[img], return_tensors="pt")
x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
mean = torch.tensor(proc.image_processor.image_mean).view(3, 1, 1)
std = torch.tensor(proc.image_processor.image_std).view(3, 1, 1)
direct = ((x - mean) / std)[None, None]
diff = (enc["pixel_values"] - direct).abs().max().item()
print(f"1) 전처리 일치: 차이 최댓값 {diff:.2e}, pixel_values {tuple(enc['pixel_values'].shape)}, 입력 {enc['input_ids'].shape[1]}토큰 →", "통과" if diff < 1e-5 else "실패")

# 2) census
model = AutoModelForImageTextToText.from_pretrained(M["model_id"], dtype=torch.bfloat16).to(dev).eval()
count = quant_rtn.census(model, M["parts"])
quant_rtn.check_census(count, "", settings.PART_NAMES)
print("2) census:", {settings.PART_NAMES[p]: f"{q}/{t}" for p, (q, t) in count.items()}, "→ 통과")
eos = GenerationConfig.from_pretrained(M["model_id"]).eos_token_id
print("   종료 토큰:", eos, "| 장치:", dev)

# 3) 재현성
@torch.no_grad()
def answer(image):
    e = proc(text=prompt, images=[image], return_tensors="pt").to(dev)
    out = model.generate(**e, **settings.GEN_KWARGS)
    new = out[0, e["input_ids"].shape[1]:]
    return len(new), proc.tokenizer.decode(new, skip_special_tokens=True).strip()

a, b = answer(img), answer(img)
print(f"3) 재현성: {a[0]}토큰 '{a[1][:60]}' / {b[0]}토큰 →", "통과" if a == b else "실패")
