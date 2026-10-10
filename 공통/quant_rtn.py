"""가짜 양자화(rtn)와 확인(census). 두 팀이 이 함수를 그대로 쓴다.

- Linear 층의 weight만 바꾼다. bias·LayerNorm·Conv·임베딩은 원본 그대로.
- 가중치 행렬을 입력 방향으로 group개씩 묶고, 묶음마다 최솟값~최댓값을 2^bits − 1 칸으로 나눈 눈금에
  반올림한 뒤 다시 실수로 풀어 둔다. 계산 결과는 압축 모델과 같지만 속도·메모리 이득은 없다.

쓰는 법:
    model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
    change = apply_rtn(model, settings.MODELS["smolvlm"]["parts"], targets="V", bits=4)   # {'V': 9.38}
    check_census(census(model, PARTS), targets="V", names=settings.PART_NAMES)
확인된 값: SmolVLM 비전 인코더 8비트 가중치 변화 0.57%, 4비트 9.38% (예비 실험의 bnb nf4 9.42%와 같은 크기).
"""
import torch

BNB_TYPES = ("Linear8bitLt", "Linear4bit")   # bitsandbytes가 바꿔 끼운 층의 클래스 이름


def rtn_weight(w, bits, group=64):
    """가중치 행렬 w(출력 × 입력)를 그룹별 균등 눈금으로 반올림한 값. 모양과 dtype은 그대로."""
    if bits < 2 or bits > 16:
        raise ValueError(f"비트는 2~16 사이여야 한다: {bits}")
    out_f, in_f = w.shape
    if in_f % group != 0:
        raise ValueError(f"입력 크기 {in_f}가 그룹 {group}으로 나누어떨어지지 않는다")
    g = w.detach().float().reshape(out_f, in_f // group, group)
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-8) / levels
    q = ((g - lo) / scale).round().clamp(0, levels)
    return (q * scale + lo).reshape(out_f, in_f).to(w.dtype)


def part_of(name, parts):
    """모듈 이름이 어느 부품에 속하는지(글자). 어디에도 안 속하면 None."""
    owners = [p for p, prefix in parts.items() if name == prefix or name.startswith(prefix + ".")]
    if len(owners) > 1:
        raise ValueError(f"두 부품에 동시에 속하는 모듈: {name}")
    return owners[0] if owners else None


def apply_rtn(model, parts, targets, bits, group=64):
    """targets(예: "V", "VCLH")에 속한 Linear weight를 제자리에서 가짜 양자화한다.
    부품별 가중치 변화(%) = 바뀐 크기 ÷ 원래 크기를 돌려준다.
    weight를 새 Parameter로 갈아 끼우므로 가중치를 공유(tie)하던 다른 층은 바뀌지 않는다."""
    sums = {p: [0.0, 0.0] for p in targets}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        p = part_of(name, parts)
        if p is None or p not in targets:
            continue
        w = module.weight.data
        w_q = rtn_weight(w, bits, group)
        sums[p][0] += (w_q.float() - w.float()).pow(2).sum().item()
        sums[p][1] += w.float().pow(2).sum().item()
        module.weight = torch.nn.Parameter(w_q, requires_grad=False)
        module.rtn_bits = bits
    return {p: 100 * (num / den) ** 0.5 for p, (num, den) in sums.items() if den > 0}


def census(model, parts):
    """부품별 [양자화된 Linear 수, 전체 Linear 수]. 어느 부품에도 안 속하는 Linear가 있으면 멈춘다."""
    count = {p: [0, 0] for p in parts}
    for name, module in model.named_modules():
        kind = type(module).__name__
        if not isinstance(module, torch.nn.Linear) and kind not in BNB_TYPES:
            continue
        p = part_of(name, parts)
        if p is None:
            raise ValueError(f"어느 부품에도 속하지 않는 Linear: {name} → settings의 parts를 고칠 것")
        count[p][1] += 1
        count[p][0] += int(kind in BNB_TYPES or hasattr(module, "rtn_bits"))
    return count


def check_census(count, targets, names):
    """의도한 부품만 '전부' 양자화됐는지 확인한다. targets가 ""이면 기준(ref)."""
    for p, (q, total) in count.items():
        want = total if p in targets else 0
        if q != want:
            raise RuntimeError(f"{names[p]}: 양자화된 Linear {q}개, 기대 {want}개")
