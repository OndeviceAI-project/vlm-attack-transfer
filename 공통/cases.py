"""경우 만들기(README 5.4). 두 팀이 이 코드를 그대로 쓴다. 4~7단계 코드는 경우를 직접 만들지 않고 build()를 부른다.

쓰는 법(<팀>/code/ 의 코드에서):
    import runtime as rt                        # 팀의 runtime.py. 불러오면 공통 폴더가 import 경로에 들어간다
    import cases
    reference = cases.reference_weights(rt)     # 원본 가중치. bnb 경우의 가중치 변화를 잴 때 쓴다. 한 번만 만든다
    model, info = cases.build(rt, "rtn-V-4", reference)
    ...                                         # 답 생성(rt.answer)
    del model
    cases.free()                                # 다음 경우를 올리기 전에 GPU 메모리를 비운다

build()는 경우마다 모델을 새로 올리고, 아래를 확인한다. 하나라도 다르면 멈춘다.
- 대상 부품의 Linear는 전부 바뀌고 나머지 부품은 하나도 바뀌지 않았다(census), 부품별 Linear 수가 settings와 같다
- bnb 경우도 모델의 종료 토큰 목록이 settings와 같다(ref·rtn·rand는 rt.load_ref()가 확인한다)
"""
import gc
import logging

import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

import quant_rtn
import settings

DEVICE_MAP = {"": 0}      # bnb 모델을 올리는 GPU. ref·rtn·rand는 rt.load_ref()가 올린다

# bitsandbytes 8비트 층은 bf16 입력을 float16으로 바꿔 계산하고, 그때마다 같은 안내를 한 줄씩 찍는다.
# 답 하나에 수천 줄이 되어 진행 상황이 묻히므로 이 안내만 끈다. 계산은 바뀌지 않는다
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)


def parse(name):
    """'ref' → ('ref', '', None), 'rtn-VL-4' → ('rtn', 'VL', 4). settings.CASES_ALL에 없는 이름이면 멈춘다."""
    if name not in settings.CASES_ALL:
        raise ValueError(f"settings.CASES_ALL에 없는 경우: {name}")
    if name == "ref":
        return "ref", "", None
    method, targets, bits = name.split("-")
    return method, targets, int(bits)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reference_weights(rt):
    """원본(ref)에서 V·L 부품의 Linear weight를 CPU로 복사해 둔다. {모듈 이름: 텐서}"""
    model = rt.load_ref()
    weights = {name: module.weight.detach().to("cpu")
               for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and quant_rtn.part_of(name, rt.PARTS) in ("V", "L")}
    del model
    free()
    return weights


def _bnb_config(bits, skip):
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip)


def _load_bnb(rt, targets, bits):
    """대상이 아닌 부품은 건너뛰게 하고 bitsandbytes로 올린다. 그 밖의 인자는 rt.load_ref()와 같다."""
    skip = [prefix for part, prefix in rt.PARTS.items() if part not in targets]
    model = AutoModelForImageTextToText.from_pretrained(
        rt.M["model_id"], revision=rt.M["revision"], dtype=rt.DTYPE,
        attn_implementation=settings.ATTN_IMPLEMENTATION,
        quantization_config=_bnb_config(bits, skip), device_map=DEVICE_MAP).eval()
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if sorted(ids) != sorted(rt.EOS):
        raise RuntimeError(f"모델의 종료 토큰 {ids}가 settings {rt.EOS}와 다르다")
    return model


def dequantized(module):
    """bnb가 바꿔 끼운 층의 weight를 실수(float32, CPU)로 되돌린다(README 5.4)."""
    kind = type(module).__name__
    if kind == "Linear4bit":
        import bitsandbytes
        weight = bitsandbytes.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
        return weight.float().cpu()
    if kind == "Linear8bitLt":
        # 행마다의 스케일(SCB)은 처음에는 weight에, 한 번 계산(forward)한 뒤에는 state에 있다
        scale = module.weight.SCB if module.weight.SCB is not None else module.state.SCB
        if scale is None or module.weight.data.dtype != torch.int8:
            raise RuntimeError("8비트 층에서 int8 weight와 행 스케일(SCB)을 찾지 못했다")
        return (module.weight.data.float() * scale.float().unsqueeze(1) / 127).cpu()
    raise TypeError(f"bitsandbytes 층이 아니다: {kind}")


def _bnb_change(model, parts, targets, reference):
    """부품별 가중치 변화(%). quant_rtn.apply_rtn이 돌려주는 값과 같은 식이다."""
    sums = {p: [0.0, 0.0] for p in targets}
    for name, module in model.named_modules():
        if type(module).__name__ not in quant_rtn.BNB_TYPES:
            continue
        part = quant_rtn.part_of(name, parts)
        original = reference[name].float()
        sums[part][0] += (dequantized(module) - original).pow(2).sum().item()
        sums[part][1] += original.pow(2).sum().item()
    return {p: 100 * (num / den) ** 0.5 for p, (num, den) in sums.items() if den > 0}


def build(rt, name, reference=None):
    """경우 하나의 모델을 올린다. (모델, 기록)을 돌려준다.
    기록: case·method·targets·bits, census(부품별 [바뀐 Linear 수, 전체 Linear 수]), weight_change_pct(부품별 %)."""
    method, targets, bits = parse(name)
    if method == "bnb":
        if reference is None:
            raise ValueError("bnb 경우는 reference_weights(rt)의 결과를 함께 준다")
        model = _load_bnb(rt, targets, bits)
        change = _bnb_change(model, rt.PARTS, targets, reference)
    else:
        model = rt.load_ref()
        change = {}
        if method == "rtn":
            change = quant_rtn.apply_rtn(model, rt.PARTS, targets, bits, group=settings.QUANT_GROUP)
        elif method == "rand":
            change = quant_rtn.apply_rand(model, rt.PARTS, targets, bits, group=settings.QUANT_GROUP,
                                          seed=settings.RAND_SEED)
    census = rt.census_record(model, targets)
    info = {"case": name, "method": method, "targets": targets, "bits": bits, "census": census,
            "weight_change_pct": {p: round(v, 4) for p, v in change.items()}}
    return model, info
