# 실행: python attack_smolvlm.py --images ./imgs --out ./adv
#   진행 상황은 터미널(노트북이면 셀 아래)에 출력되고, 공격 사진은 --out 폴더에 PNG로 저장된다.
#   이미 있는 공격 사진은 건너뛴다. 다시 만들려면 --overwrite 를 붙인다.
#   공식 공격 사진 세트(README 3단계)는 A팀/code/make_attack_set.py가 attack_one을 불러 만든다.
"""SmolVLM verbose-image attack (ported from Verbose Images, ICLR 2024).
모델은 평가 모델과 같은 bf16(16비트)으로 계산한다. 사진과 노이즈는 픽셀 값(0~1 실수)으로 다룬다.
모델·질문·정밀도·공격 기본값은 공통/settings.py에서 읽는다."""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "공통"))
import settings  # noqa: E402

_M = settings.MODELS["smolvlm"]
MODEL_ID = _M["model_id"]
MODEL_REVISION = _M["revision"]
QUESTION = settings.PROMPT
IMG_SIZE = _M["img_size"]
MODEL_DTYPE = getattr(torch, _M["attack_dtype"])   # 공격을 계산하는 모델 정밀도(bf16)


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
def natural_answer(model, processor, x, delta, prompt, device, eos_ids):
    """(답 토큰 수, 종료 토큰 없이 512에 닿았는가). 뒤의 것이 README 결과 표의 hit_cap이다."""
    enc = processor(text=prompt, images=[_pil_from(x, delta)], return_tensors="pt").to(device)
    gen = model.generate(**enc, do_sample=False, max_new_tokens=512)
    new = gen[0, enc["input_ids"].shape[1]:].tolist()
    return len(new), len(new) == 512 and not any(t in eos_ids for t in new)


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


def attack_one(model, processor, path, args, device, mean, std, eos_ids, init_seed=None):
    """사진 한 장을 공격해 args.out/<이름>.png로 저장하고, 과정의 값을 dict로 돌려준다.
    init_seed가 None이면 노이즈 0에서 시작한다(1번째 시도). 정수면 노이즈를 ±eps 안에서 무작위로 시작한다
    (2번째 시도부터). 난수는 CPU에서 만들어 기계와 상관없이 같다."""
    x = load_clean_image(path).to(device)
    prompt = build_prompt(processor)
    delta = torch.zeros_like(x, requires_grad=True)
    eps, step = args.eps / 255.0, args.step / 255.0
    if init_seed is not None:
        start = (torch.rand(tuple(x.shape), generator=torch.Generator().manual_seed(init_seed)) * 2 - 1) * eps
        with torch.no_grad():
            delta.copy_(torch.clamp(x + start.to(device), 0, 1) - x)
    bad_words = [[e] for e in eos_ids]
    name = os.path.basename(path)

    def long_target():
        with torch.no_grad():
            enc = processor(text=prompt, images=[_pil_from(x, delta)], return_tensors="pt").to(device)
            gen = model.generate(**enc, do_sample=False, bad_words_ids=bad_words,
                                  min_new_tokens=args.target_len, max_new_tokens=args.target_len)
            new = gen[0, enc["input_ids"].shape[1]:]
            return processor.tokenizer.decode(new, skip_special_tokens=True).strip() or "This"

    def tag(cap):
        return " (종료 없이 512)" if cap else ""

    start_len, start_cap = natural_answer(model, processor, x, delta, prompt, device, eos_ids)
    print(f"[{name}] 시작 자연 답 길이: {start_len} 토큰{tag(start_cap)}")

    # 저장할 사진: 조기 종료하면 멈춘 시점(두 번째 연속 512)의 사진. 끝까지 돌면 확인한 시점(시작 포함) 중
    # (종료 없이 512, 답 길이)가 가장 큰 것, 같으면 앞의 것
    best, best_delta, best_it = (start_cap, start_len), delta.detach().clone(), None
    hits = 0
    iters_run, stopped_early, got_nan = 0, False, False
    n_iter = args.iter
    if start_len >= 512:   # 이미 512면 어느 걸음도 원본보다 높지 않으므로 공격하지 않는다(B팀 코드와 같음)
        print(f"[{name}] 시작부터 512 → 공격하지 않고 원본 그대로 저장")
        n_iter = 0
    for it in range(n_iter):
        if it % args.refresh == 0:
            target = long_target()
        base = processor(text=prompt + target, images=[_pil_from(x, delta.detach())],
                         return_tensors="pt").to(device)
        prompt_len = processor(text=prompt, images=[_pil_from(x, delta.detach())],
                               return_tensors="pt")["input_ids"].shape[1]
        n_tiles = base["pixel_values"].shape[1]
        pv = make_pixel_values(x, delta, mean, std, n_tiles).to(model.dtype)   # 모델과 같은 bf16으로 넣는다
        l_eos, l_unc, l_div = losses(model, base, pv, prompt_len, eos_ids, args.w_div > 0)
        loss = l_eos + args.w_unc * l_unc + args.w_div * l_div
        model.zero_grad(set_to_none=True)
        delta.grad = None
        loss.backward()
        if torch.isnan(loss) or torch.isnan(delta.grad).any():
            print(f"[{name}] it {it}: nan 발생 → 그때까지 고른 사진을 저장하고 중단")
            got_nan = True
            break
        with torch.no_grad():
            delta -= step * delta.grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)
        iters_run = it + 1
        if it % args.refresh == 0 or it + 1 == n_iter:   # 마지막 걸음도 확인한다
            nl, cap = natural_answer(model, processor, x, delta, prompt, device, eos_ids)
            print(f"[{name}] it {it:4d} | eos_logit {l_eos.item():+.3f} | 자연 답 길이 {nl} 토큰{tag(cap)}")
            if (cap, nl) > best:
                best, best_delta, best_it = (cap, nl), delta.detach().clone(), it
            hits = hits + 1 if cap else 0
            if hits >= 2:   # 멈춘 시점의 사진을 저장한다. 처음 닿은 사진보다 refresh걸음 더 다듬어졌고 512도 확인했다
                best, best_delta, best_it, stopped_early = (cap, nl), delta.detach().clone(), it, True
                print(f"[{name}] 512 도달 → 조기 종료")
                break

    print(f"[{name}] 고른 사진: " + ("원본 그대로" if best_it is None else f"it {best_it:4d}")
          + f" | 답 길이 {best[1]} 토큰{tag(best[0])}")
    out_path = os.path.join(args.out, os.path.splitext(name)[0] + ".png")   # 무손실 PNG로 저장
    _save_png(torch.clamp(x + best_delta, 0, 1), out_path)
    saved, saved_cap = natural_answer(model, processor, x, best_delta, prompt, device, eos_ids)
    print(f"[{name}] 저장본 실제 답 길이: {saved} 토큰{tag(saved_cap)}  →  {out_path}")
    if (saved_cap, saved) != best:
        print(f"[{name}] 주의: 저장본 답 길이가 고를 때와 다르다 → 이 사진은 확인 필요")
    return {"out_path": out_path, "start_len": start_len, "start_cap": start_cap, "iters_run": iters_run,
            "best_it": best_it, "best_len": best[1], "best_cap": best[0], "saved_len": saved, "saved_cap": saved_cap,
            "stopped_early": stopped_early, "nan": got_nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="./imgs")
    ap.add_argument("--out", default="./adv")
    ap.add_argument("--iter", type=int, default=settings.ATTACK_ITERS)
    ap.add_argument("--eps", type=float, default=float(settings.EPS))
    ap.add_argument("--step", type=float, default=float(settings.STEP))
    ap.add_argument("--target_len", type=int, default=settings.TARGET_LEN)
    ap.add_argument("--refresh", type=int, default=settings.REFRESH)
    ap.add_argument("--w_unc", type=float, default=settings.W_UNC)
    ap.add_argument("--w_div", type=float, default=settings.W_DIV)
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
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, **_M["processor_kwargs"])
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=MODEL_DTYPE,
        attn_implementation=settings.ATTN_IMPLEMENTATION).to(device).eval()
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    eos_ids = eos_token_ids(processor)
    assert eos_ids == sorted(_M["eos_token_ids"]), f"종료 토큰이 settings와 다르다: {eos_ids}"
    print("device:", device, "| eos ids:", eos_ids)

    print(f"{len(todo)}장 공격 시작")
    for p in todo:
        attack_one(model, processor, p, args, device, mean, std, eos_ids)
    print("완료 →", args.out)

if __name__ == "__main__":
    main()
