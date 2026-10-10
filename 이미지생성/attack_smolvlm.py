# 실행: python attack_smolvlm.py --images ./imgs --out ./adv --iter 200 --eps 8
#   진행 상황은 터미널(노트북이면 셀 아래)에 출력되고, 공격 사진은 --out 폴더에 PNG로 저장된다.
#   이미 있는 공격 사진은 건너뛴다. 다시 만들려면 --overwrite 를 붙인다.
"""SmolVLM verbose-image attack (ported from Verbose Images, ICLR 2024)."""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
QUESTION = "Describe this image in one sentence."
IMG_SIZE = 512


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_prompt(processor):
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def eos_token_ids(processor):
    ids = set()
    tok = processor.tokenizer
    tid = tok.convert_tokens_to_ids("<end_of_utterance>")
    if tid is not None and tid >= 0:
        ids.add(tid)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def _to_float_array(pil_img):
    import numpy as np
    return np.asarray(pil_img, dtype="float32") / 255.0


@torch.no_grad()
def load_clean_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return torch.from_numpy(_to_float_array(img)).permute(2, 0, 1).contiguous()


def _pil_from(x, delta):
    import numpy as np
    adv = torch.clamp(x + delta, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((adv * 255).round().astype("uint8"))


def _save_png(adv_chw, path):
    import numpy as np
    arr = (adv_chw.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path, format="PNG")   # 반드시 무손실 PNG (JPEG면 공격이 지워진다)


def make_pixel_values(x, delta, mean, std, n_tiles):
    adv = torch.clamp(x + delta, 0.0, 1.0)
    pv = ((adv - mean) / std).unsqueeze(0).unsqueeze(0)
    if n_tiles > 1:
        pv = pv.repeat(1, n_tiles, 1, 1, 1)
    return pv


@torch.no_grad()
def natural_len_of(model, processor, x, delta, prompt, device):
    enc = processor(text=prompt, images=[_pil_from(x, delta)], return_tensors="pt").to(device)
    gen = model.generate(**enc, do_sample=False, max_new_tokens=512)
    return gen.shape[1] - enc["input_ids"].shape[1]


def losses(model, base, pixel_values, answer_start, eos_ids, use_div):
    out = model(
        input_ids=base["input_ids"],
        attention_mask=base["attention_mask"],
        pixel_values=pixel_values,
        pixel_attention_mask=base.get("pixel_attention_mask"),
        output_hidden_states=use_div,
        use_cache=False,
    )
    logits = out.logits[0, answer_start - 1:-1, :].float()
    eos_logit = logits[:, eos_ids].max(dim=-1).values
    loss_eos = eos_logit.mean()
    logp = F.log_softmax(logits, dim=-1)
    uni = torch.full_like(logp, 1.0 / logp.shape[-1])
    loss_unc = F.kl_div(logp, uni, reduction="batchmean")
    if use_div:
        hs = out.hidden_states[-1][0, answer_start - 1:-1, :].float()
        loss_div = -torch.linalg.matrix_norm(hs.cpu(), ord="nuc").to(logits.device)
    else:
        loss_div = torch.zeros((), device=logits.device)
    return loss_eos, loss_unc, loss_div


def attack_one(model, processor, path, args, device, mean, std, eos_ids):
    x = load_clean_image(path).to(device)
    prompt = build_prompt(processor)
    delta = torch.zeros_like(x, requires_grad=True)
    eps, step = args.eps / 255.0, args.step / 255.0
    bad_words = [[e] for e in eos_ids]
    name = os.path.basename(path)

    def long_target():
        with torch.no_grad():
            enc = processor(text=prompt, images=[_pil_from(x, delta)], return_tensors="pt").to(device)
            gen = model.generate(**enc, do_sample=False, bad_words_ids=bad_words,
                                  min_new_tokens=args.target_len, max_new_tokens=args.target_len)
            new = gen[0, enc["input_ids"].shape[1]:]
            return processor.tokenizer.decode(new, skip_special_tokens=True).strip() or "This"

    target = long_target()
    print(f"[{name}] 시작 자연 답 길이: {natural_len_of(model, processor, x, delta, prompt, device)} 토큰")

    best_delta = delta.detach().clone()
    hits = 0
    for it in range(args.iter):
        if it > 0 and it % args.refresh == 0:
            target = long_target()
        base = processor(text=prompt + target, images=[_pil_from(x, delta.detach())],
                         return_tensors="pt").to(device)
        prompt_len = processor(text=prompt, images=[_pil_from(x, delta.detach())],
                               return_tensors="pt")["input_ids"].shape[1]
        n_tiles = base["pixel_values"].shape[1]
        pv = make_pixel_values(x, delta, mean, std, n_tiles)
        l_eos, l_unc, l_div = losses(model, base, pv, prompt_len, eos_ids, args.w_div > 0)
        loss = l_eos + args.w_unc * l_unc + args.w_div * l_div
        model.zero_grad(set_to_none=True)
        delta.grad = None
        loss.backward()
        if torch.isnan(loss) or torch.isnan(delta.grad).any():
            print(f"[{name}] it {it}: nan 발생 → 직전 정상 상태로 저장하고 중단")
            break
        with torch.no_grad():
            delta -= step * delta.grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)
            best_delta = delta.detach().clone()
        if it % args.refresh == 0:
            nl = natural_len_of(model, processor, x, delta, prompt, device)
            print(f"[{name}] it {it:4d} | eos_logit {l_eos.item():+.3f} | 자연 답 길이 {nl} 토큰")
            hits = hits + 1 if nl >= 500 else 0
            if hits >= 2:
                print(f"[{name}] 512 도달 → 조기 종료")
                break

    out_path = os.path.join(args.out, os.path.splitext(name)[0] + ".png")   # 무손실 PNG로 저장
    _save_png(torch.clamp(x + best_delta, 0, 1), out_path)
    saved = natural_len_of(model, processor, x, best_delta, prompt, device)
    print(f"[{name}] 저장본 실제 답 길이: {saved} 토큰  →  {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="./imgs")
    ap.add_argument("--out", default="./adv")
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--target_len", type=int, default=128)
    ap.add_argument("--refresh", type=int, default=20)
    ap.add_argument("--w_unc", type=float, default=0.05)
    ap.add_argument("--w_div", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = [os.path.join(args.images, f) for f in sorted(os.listdir(args.images))
             if f.lower().endswith(exts)]
    todo = []   # 이미 만든 공격 사진은 다시 만들지 않는다(같은 사진을 모든 양자화 경우에 재사용)
    for p in paths:
        out_path = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".png")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{os.path.basename(p)}] 이미 있음 → 건너뜀 ({out_path})")
        else:
            todo.append(p)
    if not todo:
        print("새로 만들 사진 없음 (다시 만들려면 --overwrite)")
        return

    device = pick_device()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    eos_ids = eos_token_ids(processor)
    print("device:", device, "| eos ids:", eos_ids)

    print(f"{len(todo)}장 공격 시작")
    for p in todo:
        attack_one(model, processor, p, args, device, mean, std, eos_ids)
    print("완료 →", args.out)

if __name__ == "__main__":
    main()
