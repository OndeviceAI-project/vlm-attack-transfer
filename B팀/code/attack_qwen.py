"""Qwen2-VL 공격 사진 한 장 만들기(이미지생성/README.md의 제작 방식).

v1 B팀 코드(태그 v1-midterm의 B팀/src/attack.py)의 손실·노이즈 갱신을 그대로 옮기고, 두 팀이 합의한 방식으로 바꿨다.
- 손실: 긴 목표 문장(종료 토큰 금지, TARGET_LEN토큰, REFRESH걸음마다 갱신)을 따라 읽힐 때 자리마다 종료 토큰
  점수의 최댓값 평균 + W_UNC × 불확실성. 노이즈는 기울기 부호로 STEP씩, ±EPS 안(0~255 단위).
- 시작: init_seed가 None이면 노이즈 0(1번째 시도), 정수면 ±EPS 안 무작위(2번째 시도부터, CPU 난수).
- 확인: REFRESH걸음마다와 마지막 걸음에 답 길이를 잰다. 종료 없이 512가 두 번 연속이면 멈춘다.
- 저장할 사진: 멈추면 멈춘 시점(두 번째 연속 512)의 사진. 끝까지 돌면 확인한 시점(시작 포함) 중
  (종료 없이 512, 답 길이)가 가장 큰 것, 같으면 앞의 것.
- 숫자가 깨지면(nan·inf) 그때까지 고른 사진으로 멈춘다.
"""
import inspect
import math

import torch
import torch.nn.functional as F

import runtime as rt
from runtime import settings


def extend_answer(base, target):
    """질문 입력 뒤에 목표 문장 토큰을 붙인다."""
    out = dict(base)
    suffix = target.unsqueeze(0)
    out["input_ids"] = torch.cat([base["input_ids"], suffix], dim=1)
    out["attention_mask"] = torch.cat([base["attention_mask"], torch.ones_like(suffix)], dim=1)
    for key in ["mm_token_type_ids", "token_type_ids"]:
        if key in base:
            out[key] = torch.cat([base[key], torch.zeros_like(suffix)], dim=1)
    return out


def objective(model, processor, base, pixels, target):
    extended = extend_answer(base, target)
    extended["pixel_values"] = rt.pixel_values(processor, pixels).to(rt.DTYPE)
    extra = {"logits_to_keep": target.numel() + 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    out = model(**extended, use_cache=False, **extra)
    logits = out.logits[0, -target.numel() - 1:-1].float()
    assert logits.shape[0] == target.numel(), "목표 문장과 로짓의 자리가 맞지 않는다"
    loss_eos = logits[:, rt.EOS].max(dim=-1).values.mean()
    logp = F.log_softmax(logits, dim=-1)
    loss_unc = -logp.mean() - math.log(logp.shape[-1])
    return loss_eos + settings.W_UNC * loss_unc, loss_eos


@torch.no_grad()
def target_tokens(model, processor, img):
    """종료 토큰을 금지하고 TARGET_LEN토큰을 생성한 긴 목표 문장."""
    _, ids = rt.generate(model, processor, img, {"max_new_tokens": settings.TARGET_LEN,
                                                  "min_new_tokens": settings.TARGET_LEN,
                                                  "bad_words_ids": [[e] for e in rt.EOS]})
    assert len(ids) == settings.TARGET_LEN and not any(i in rt.EOS for i in ids), "목표 문장이 128토큰이 아니다"
    return torch.tensor(ids, device="cuda", dtype=torch.long)


def attack_one(model, processor, clean_img, name="", iters=None, init_seed=None, log=print):
    """공격 PNG로 저장할 PIL 사진과 과정의 값을 돌려준다."""
    iters = settings.ATTACK_ITERS if iters is None else iters
    x = rt.raw_pixels(clean_img, "cuda")
    delta = torch.zeros_like(x, requires_grad=True)
    if init_seed is not None:
        start = (torch.rand(tuple(x.shape), generator=torch.Generator().manual_seed(init_seed)) * 2 - 1) * settings.EPS
        with torch.no_grad():
            delta.copy_((x + start.to("cuda")).clamp(0, 255) - x)
    base = rt.encode(processor, clean_img).to("cuda")

    start_res, _ = rt.generate(model, processor, rt.to_image(x + delta))
    log(f"[{name}] 시작 답 길이: {start_res['n_tokens']} 토큰" + (" (종료 없이 512)" if start_res["hit_cap"] else ""))
    best, best_delta, best_it = (start_res["hit_cap"], start_res["n_tokens"]), delta.detach().clone(), None
    hits, iters_run, stopped_early, got_nan, target = 0, 0, False, False, None
    for it in range(iters):
        if it % settings.REFRESH == 0:
            target = target_tokens(model, processor, rt.to_image(x + delta))
        delta.grad = None
        loss, loss_eos = objective(model, processor, base, (x + delta).clamp(0, 255), target)
        loss.backward()
        if not torch.isfinite(loss) or delta.grad is None or not torch.isfinite(delta.grad).all():
            log(f"[{name}] it {it}: 숫자가 깨짐(nan·inf) → 그때까지 고른 사진으로 멈춤")
            got_nan = True
            break
        with torch.no_grad():
            delta.sub_(settings.STEP * delta.grad.sign())
            delta.clamp_(-settings.EPS, settings.EPS)
            delta.copy_((x + delta).clamp(0, 255) - x)
        iters_run = it + 1
        if it % settings.REFRESH == 0 or iters_run == iters:
            res, _ = rt.generate(model, processor, rt.to_image(x + delta))
            score = (res["hit_cap"], res["n_tokens"])
            log(f"[{name}] it {it:4d} | eos_logit {loss_eos.item():+.3f} | 답 길이 {res['n_tokens']} 토큰"
                + (" (종료 없이 512)" if res["hit_cap"] else ""))
            if score > best:
                best, best_delta, best_it = score, delta.detach().clone(), it
            hits = hits + 1 if res["hit_cap"] else 0
            if hits >= 2:   # 멈춘 시점의 사진을 저장한다. 처음 닿은 사진보다 REFRESH걸음 더 다듬어졌고 512도 확인했다
                best, best_delta, best_it, stopped_early = score, delta.detach().clone(), it, True
                log(f"[{name}] 512 도달 → 조기 종료")
                break
    log(f"[{name}] 고른 사진: " + ("시작 그대로" if best_it is None else f"it {best_it:4d}")
        + f" | 답 길이 {best[1]} 토큰")
    return rt.to_image(x + best_delta), {"start_len": start_res["n_tokens"], "iters_run": iters_run,
                                         "best_it": best_it, "best_len": best[1], "best_cap": best[0],
                                         "stopped_early": stopped_early, "nan": got_nan}
