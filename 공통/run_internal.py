"""6단계 모델 내부 측정(README 6단계). 두 팀이 같은 코드를 자기 실험 기계에서 돌린다. 답을 채점하지 않고 모델 안의 값을 잰다.

실행: python 공통/run_internal.py --team A                (B팀은 --team B)
      python 공통/run_internal.py --team A --limit 2 --cases rtn-V-4,rtn-L-4
                                   짧은 시험. 사진 2장, ref와 적은 경우만 돌리고 결과 폴더에는 쓰지 않는다
결과: <팀>/results/internal.csv, internal.json      모든 경우가 끝났을 때만 생긴다

하는 일
1. 시작 조건을 확인한다(run_answers.py와 같다). 실행 환경, 코드, 1-3단계 통과, 넣을 사진의 SHA-256.
2. 첫 경우인 ref에서 사진마다 토큰 열 두 개를 만든다.
   - a: ref가 공격 사진에 생성한 답의 앞 64토큰. 그 안에 종료 토큰이 있으면 멈춘다.
   - c: ref가 정상 사진에 생성한 답에서 마지막 종료 토큰을 뺀 것. 종료 토큰으로 끝나지 않으면 멈춘다.
   정상 사진 답의 토큰 수가 adv/attempts.csv의 ref_clean_n_tokens와 같은지 본다(3단계 기록과 맞는가). 다르면 멈춘다.
3. 경우(settings.CASES_ALL 순서)마다 모델을 새로 올려, 사진마다 두 번 계산(forward)한다.
   - 공격 사진 + 질문 + a → 그림 정보 f(adv)와, a의 토큰을 고르는 64자리의 로짓
   - 정상 사진 + 질문 + c → 그림 정보 f(clean)와, c 다음 토큰을 고르는 자리의 로짓
   그림 정보는 연결부 모듈의 출력이다. 사진만으로 정해지므로 로짓을 구하는 계산에서 함께 얻는다.
4. ref에서는 다시 계산한 로짓이 생성한 답(a, c, 종료)을 고르는지 본다. 입력과 자리 계산이 5.2절의 생성과 같다는
   확인이다. 고른 비율이 REPLAY_MIN보다 낮으면 멈춘다.
5. 경우 하나가 끝날 때마다 <팀>/data/runs/internal/ 에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는
   건너뛴다). 환경·코드·입력이 처음과 다르면 이어서 하지 않고 멈춘다. 처음부터 다시 하려면 --restart 를 붙인다.
"""
import argparse
import csv
import json
import os
import time

import torch

import cases
import check_cases
import run_answers as common
import settings

ROOT = check_cases.ROOT
STEP = "internal"
N_A = 64                 # a의 길이(README 6단계)
# ref에서 다시 계산한 로짓이 생성한 토큰을 고른 비율의 하한. 입력이나 자리 계산이 틀리면 이 비율이 크게 떨어진다.
# bf16에서는 로짓 값의 간격이 넓어 1·2등이 동점인 자리가 생기고, 그런 자리는 생성 때와 다른 토큰이 뽑힐 수 있어
# 100%가 되지 않는다(float32로 올리면 100%다). 그래서 하한을 넉넉히 둔다
REPLAY_MIN = 0.8
COLUMNS = ["model", "case", "image_id", "feat_change_clean", "attack_shift", "attack_dir_kept", "shake_ratio",
           "eos_gap_mean", "eos_gap_min", "clean_end_margin"]
# 프로세서가 토큰마다 하나씩 주는 입력과, 뒤에 붙인 글 토큰에 줄 값. attention_mask는 1(보는 토큰),
# mm_token_type_ids는 0(글 토큰. Qwen2-VL 프로세서가 사진 토큰과 글 토큰을 가르려고 준다)
TOKEN_FILL = {"attention_mask": 1, "mm_token_type_ids": 0}
stop = common.stop


class Feature:
    """연결부 모듈의 출력(그림 정보)을 잡아 둔다. 한 번 계산할 때 연결부는 한 번만 불려야 한다."""

    def __init__(self, model, prefix):
        self.value, self.calls = None, 0
        self.handle = model.get_submodule(prefix).register_forward_hook(self._keep)

    def _keep(self, module, args, output):
        self.value, self.calls = output, self.calls + 1

    def reset(self):
        self.value, self.calls = None, 0

    def take(self):
        """잡아 둔 출력을 1차원으로 펴서 CPU로 옮긴다. 값은 모델이 낸 그대로다(dtype을 바꾸지 않는다)."""
        if self.calls != 1 or not torch.is_tensor(self.value):
            raise RuntimeError(f"연결부 출력이 텐서 하나가 아니다(불린 횟수 {self.calls})")
        value = self.value.detach().flatten().cpu()
        self.reset()
        return value

    def close(self):
        self.handle.remove()


def encode(rt, processor, model, img):
    """rt.answer가 모델에 넣는 것과 같은 입력(README 5.2·5.3). 팀 runtime.py가 입력을 만드는 줄을 그대로 따른다."""
    if hasattr(rt, "encode"):          # B팀(Qwen2-VL): 사진 격자를 확인하고 pixel_values를 모델 dtype으로 바꾼다
        enc = rt.encode(processor, img).to(model.device)
        enc["pixel_values"] = enc["pixel_values"].to(rt.DTYPE)
        return enc
    return processor(text=rt.prompt_text(processor), images=[img], return_tensors="pt").to(model.device)


@torch.no_grad()
def generate(rt, model, enc, max_new_tokens):
    """README 5.2의 생성에서 max_new_tokens만 바꾼다. 새로 만든 토큰 id 목록을 돌려준다."""
    kwargs = {**settings.GEN_KWARGS, "max_new_tokens": max_new_tokens}
    out = model.generate(**enc, **kwargs, eos_token_id=rt.EOS)
    return out[0, enc["input_ids"].shape[1]:].tolist()


@torch.no_grad()
def forward(model, feature, enc, tokens):
    """사진과 질문 뒤에 tokens를 붙여 한 번 계산한다. (그림 정보, 로짓 float32 [len(tokens) + 1, 낱말 수])
    로짓의 j번째 줄은 tokens[j]를 고르는 자리이고, 마지막 줄은 tokens 다음 토큰을 고르는 자리다."""
    ids = enc["input_ids"]
    extra = torch.tensor([tokens], dtype=ids.dtype, device=ids.device).reshape(1, len(tokens))
    inputs = dict(enc)
    inputs["input_ids"] = torch.cat([ids, extra], dim=1)
    for key, value in enc.items():          # 토큰마다 값이 하나씩 있는 다른 입력도 붙인 토큰만큼 늘린다
        if key == "input_ids" or not torch.is_tensor(value) or value.shape != ids.shape:
            continue
        if key not in TOKEN_FILL:
            raise RuntimeError(f"토큰마다 값이 있는 입력 {key}을(를) 붙인 토큰만큼 늘리는 방법이 정해져 있지 않다")
        inputs[key] = torch.cat([value, torch.full_like(extra, TOKEN_FILL[key]).to(value.dtype)], dim=1)
    feature.reset()
    logits = model(**inputs).logits[0, ids.shape[1] - 1:].float().cpu()
    return feature.take(), logits


def make_tokens(rt, processor, model, plan):
    """ref에서 사진마다 a와 c를 만든다. ({사진: {"a": [...], "c": [...]}}, 3단계 기록과 맞는지의 기록)"""
    recorded = {}
    for row in common.read_csv(rt.TEAM / "adv" / "attempts.csv", "adv/attempts.csv"):
        recorded.setdefault(row["coco_id"], int(row["ref_clean_n_tokens"]))
    tokens, differs = {}, []
    for photo in plan:
        image_id = photo["image_id"]
        a = generate(rt, model, encode(rt, processor, model, rt.read_png(photo["paths"]["adv"])), N_A)
        if len(a) != N_A or any(token in rt.EOS for token in a):
            stop(f"ref의 공격 사진 답이 {N_A}토큰 안에 끝났다({image_id}) → 3단계의 공격 사진, 실행 환경과 같은지 확인한다")
        full = generate(rt, model, encode(rt, processor, model, rt.read_png(photo["paths"]["clean"])),
                        settings.MAX_NEW_TOKENS)
        if not full or full[-1] not in rt.EOS:
            stop(f"ref의 정상 사진 답이 종료 토큰으로 끝나지 않았다({image_id}) → 3단계의 세트, 실행 환경과 같은지 확인한다")
        if len(full) != recorded.get(image_id):
            differs.append(image_id)
        tokens[image_id] = {"a": a, "c": full[:-1]}
    return tokens, {"pass": not differs, "photos": len(plan), "clean_tokens_differ": differs}


def measure(rt, processor, model, feature, photo, tokens):
    """사진 한 장의 두 계산. 그림 정보 둘, eos_gap 평균·최솟값, clean_end_margin, 다시 고른 토큰이 맞은 수(a, c)."""
    eos = torch.tensor(rt.EOS)
    enc = encode(rt, processor, model, rt.read_png(photo["paths"]["adv"]))
    f_adv, logits = forward(model, feature, enc, tokens["a"])
    steps = logits[:N_A]
    gap = steps.max(dim=-1).values - steps[:, eos].max(dim=-1).values
    a_hits = int((steps.argmax(dim=-1) == torch.tensor(tokens["a"])).sum())
    enc = encode(rt, processor, model, rt.read_png(photo["paths"]["clean"]))
    f_clean, logits = forward(model, feature, enc, tokens["c"])
    last = logits[-1]
    others = last.clone()
    others[eos] = float("-inf")
    margin = float(last[eos].max() - others.max())
    c_hits = int((logits[:-1].argmax(dim=-1) == torch.tensor(tokens["c"], dtype=torch.long)).sum())
    return {"f_clean": f_clean, "f_adv": f_adv, "gap_mean": float(gap.mean()), "gap_min": float(gap.min()),
            "margin": margin, "a_hits": a_hits, "c_hits": c_hits}


def ratio(top, bottom):
    top, bottom = float(top), float(bottom)
    return top / bottom if bottom != 0 else float("nan")


def feature_values(f_clean, f_adv, ref_clean, ref_adv):
    """README 6단계의 앞 네 값: feat_change_clean, attack_shift, attack_dir_kept, shake_ratio. float64로 계산한다."""
    fc, fa, rc, ra = (t.double() for t in (f_clean, f_adv, ref_clean, ref_adv))
    attack, attack_ref = fa - fc, ra - rc
    return [100 * ratio((fc - rc).norm(), rc.norm()),
            100 * ratio(attack.norm(), fc.norm()),
            ratio(torch.dot(attack, attack_ref), attack.norm() * attack_ref.norm()),
            ratio((fa - ra).norm(), attack_ref.norm())]


def run_case(rt, processor, name, reference, plan, ref):
    """경우 하나의 모델을 올려 모든 사진을 잰다. ref 경우는 먼저 a·c를 만들고, 잰 그림 정보를 ref에 채운다.
    (경우 기록, CSV 줄들)"""
    started = time.time()
    model, info = cases.build(rt, name, reference)
    feature = Feature(model, rt.PARTS["C"])
    rows, same, hits, ended = [], 0, [0, 0], 0
    try:
        if name == "ref":
            ref["tokens"], ref["check"] = make_tokens(rt, processor, model, plan)
        for photo in plan:
            image_id = photo["image_id"]
            m = measure(rt, processor, model, feature, photo, ref["tokens"][image_id])
            if name == "ref":
                ref["features"][image_id] = (m["f_clean"], m["f_adv"])
            ref_clean, ref_adv = ref["features"][image_id]
            same += int(torch.equal(m["f_clean"], ref_clean) and torch.equal(m["f_adv"], ref_adv))
            hits = [hits[0] + m["a_hits"], hits[1] + m["c_hits"]]
            ended += int(m["margin"] > 0)
            values = feature_values(m["f_clean"], m["f_adv"], ref_clean, ref_adv)
            rows.append([rt.MODEL_KEY, name, image_id] + [round(v, 6) for v in values]
                        + [round(m["gap_mean"], 4), round(m["gap_min"], 4), round(m["margin"], 4)])
    finally:
        feature.close()
        model = None
        cases.free()
    info.update({"rows": len(rows), "seconds": round(time.time() - started, 1), "same_features": same})
    if name == "ref":
        c_total = sum(len(ref["tokens"][photo["image_id"]]["c"]) for photo in plan)
        replay = {"a": round(hits[0] / (N_A * len(plan)), 4), "c": round(ratio(hits[1], c_total), 4),
                  "end": round(ended / len(plan), 4)}
        ref["check"]["replay"] = replay
        ref["check"]["replay_pass"] = all(value >= REPLAY_MIN for value in replay.values())
    return info, rows


def append_rows(path, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)
        fp.flush()
        os.fsync(fp.fileno())


def save_ref(work, ref):
    """ref의 그림 정보를 파일로 남긴다(이어 할 때 쓴다). 파일의 SHA-256을 돌려준다."""
    path, temp = work / "ref_features.pt", work / "ref_features.tmp"
    torch.save(ref["features"], temp)
    os.replace(temp, path)
    return common.sha256(path.read_bytes())


def load_ref(work, state):
    """이어 할 때: 저장해 둔 a·c와 ref의 그림 정보를 읽는다. 파일이 없거나 바뀌었으면 멈춘다."""
    path = work / "ref_features.pt"
    if not path.exists() or common.sha256(path.read_bytes()) != state["ref"]["features_sha256"]:
        stop("저장해 둔 ref의 그림 정보(ref_features.pt)가 없거나 기록과 다르다 → --restart 로 처음부터 다시 한다")
    return {"tokens": state["ref"]["tokens"], "features": torch.load(path, map_location="cpu"), "check": None}


def check_ref(args, state, work, check):
    """ref에서 만든 a·c가 3단계 기록과 맞는지, 다시 계산한 로짓이 생성한 답을 고르는지. 본 실행에서는 다르면 멈춘다."""
    problems = []
    if not check["pass"]:
        problems.append("ref의 정상 사진 답 토큰 수가 3단계 기록과 다르다 → 환경이 달라졌는지 확인한다")
    if not check["replay_pass"]:
        problems.append("다시 계산한 로짓이 ref가 생성한 답을 고르지 않는다 → 입력이나 자리 계산이 5.2절의 생성과 다르다")
    if not problems:
        return
    if args.limit is None:
        common.save_state(work / "ref_check_failed.json", check)
        stop(f"{' / '.join(problems)}. 설정을 바꾸지 말고 그대로 알린다. {check}")
    for problem in problems:
        print(f"알림(시험이라 계속한다): {problem}. {check}")


def run_all(rt, args, names, plan, state, work):
    """남은 경우를 차례로 돌린다. 경우 하나가 끝날 때마다 줄과 기록을 저장한다."""
    todo = [name for name in names if name not in state["cases"]]
    if not todo:
        return
    ref = load_ref(work, state) if "ref" in state["cases"] else {"tokens": {}, "features": {}, "check": None}
    first_row = sum(c["rows"] for c in state["cases"].values()) + 1
    run = {"run": len(state["runs"]) + 1, "first_row": first_row, "rows_added": 0,
           "code_commit": state["identity"]["environment"]["code_commit"]}
    state["runs"].append(run)
    processor, reference = rt.load_processor(), cases.reference_weights(rt)
    for count, name in enumerate(todo, 1):
        info, rows = run_case(rt, processor, name, reference, plan, ref)
        if name == "ref":
            check_ref(args, state, work, ref["check"])
            state["ref_check"] = ref["check"]
            state["ref"] = {"tokens": ref["tokens"], "features_sha256": save_ref(work, ref)}
        append_rows(work / "rows.csv", rows)
        state["cases"][name] = info
        run["rows_added"] += len(rows)
        common.save_state(work / "state.json", state)
        print(f"[{count}/{len(todo)}] {name:10s} {len(rows)}줄, {info['seconds'] / 60:.1f}분 | "
              f"그림 정보가 ref와 같은 사진 {info['same_features']}/{len(plan)} | 가중치 변화 {info['weight_change_pct']}")


def finalize(rt, args, state, work, checks, inputs, plan):
    """모든 경우가 끝났으면 결과 파일을 쓴다. 시험이면 작업 폴더에, 아니면 <팀>/results/ 에 쓴다."""
    data = (work / "rows.csv").read_bytes()
    # 언어모델만 압축한 경우는 비전 인코더와 연결부가 ref와 같으므로 그림 정보가 사진마다 ref와 같아야 한다(README 6단계)
    language = [name for name in state["cases"] if cases.parse(name)[1] == "L"]
    differs = [name for name in language if state["cases"][name]["same_features"] != len(plan)]
    checks = {**checks, "language_features": {"pass": not differs, "differs": differs}}
    if differs:
        print(f"알림: 언어모델만 압축했는데 그림 정보가 ref와 다른 경우가 있다 → 원인을 확인한다. {differs}")
    meta = {"environment": state["identity"]["environment"], "step": STEP, "cases": state["cases"],
            "inputs": inputs, "ref_check": state["ref_check"], "tokens": state["ref"]["tokens"], "checks": checks,
            "csv": {"rows": sum(c["rows"] for c in state["cases"].values()), "sha256": common.sha256(data)},
            "runs": state["runs"],
            "note": "README 6단계. csv 한 줄 = 경우 하나의 사진 하나. tokens는 사진마다 ref가 만든 a(공격 사진 답의 앞 "
                    f"{N_A}토큰)와 c(정상 사진 답에서 마지막 종료 토큰을 뺀 것). same_features는 그림 정보가 ref와 "
                    "값까지 같은 사진 수. runs의 first_row는 그 실행이 처음 쓴 줄 번호(머리줄 제외)"}
    out = work if args.limit is not None else rt.TEAM / "results"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{STEP}.csv").write_bytes(data)
        (out / f"{STEP}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except OSError as err:
        stop(f"결과 파일을 쓰지 못했다({type(err).__name__})")
    print(f"저장: {out.relative_to(ROOT).as_posix()}/{STEP}.csv, {STEP}.json | {meta['csv']['rows']}줄")


def parse_args():
    parser = argparse.ArgumentParser(description="6단계 모델 내부 측정")
    parser.add_argument("--team", required=True, choices=sorted(check_cases.TEAMS))
    parser.add_argument("--limit", type=int, help="시험: 앞에서 이 수만큼의 사진만 돌린다. 결과 폴더에 쓰지 않는다")
    parser.add_argument("--cases", help="시험(--limit과 함께): ref와, 쉼표로 나눈 경우 이름만 돌린다")
    parser.add_argument("--restart", action="store_true", help="이어 하지 않고 처음부터 다시 한다")
    args = parser.parse_args()
    if args.cases and args.limit is None:
        parser.error("--cases 는 시험(--limit)에서만 쓴다")
    unknown = [name for name in (args.cases or "").split(",") if name and name not in settings.CASES_ALL]
    if unknown:
        parser.error(f"settings.CASES_ALL에 없는 경우: {unknown}")
    return args


def main():
    args = parse_args()
    rt = check_cases.load_runtime(args.team)
    names = list(settings.CASES_ALL)
    if args.cases:                   # ref는 a·c와 기준 그림 정보를 만들므로 언제나 돌린다
        names = [name for name in names if name == "ref" or name in args.cases.split(",")]
    plan, _, split = common.photo_plan(rt, True, args.limit)
    for photo in plan:               # 6단계는 정상 사진과 공격 사진만 쓴다
        photo["paths"].pop("noise")
    rt.setup()
    env = rt.environment()
    checks = common.preconditions(rt, env, trial=args.limit is not None)
    inputs = common.verify_inputs(rt, plan, split, True)
    identity = {"environment": env, "step": STEP, "cases": names, "images": [p["image_id"] for p in plan],
                "inputs": inputs, "prompt": settings.PROMPT, "gen_kwargs": settings.GEN_KWARGS, "tokens_a": N_A}
    identity = json.loads(json.dumps(identity))      # 저장했다 읽은 기록과 같은 꼴로 맞춘다(비교용)
    work = rt.DATA / ("trial" if args.limit is not None else "runs") / STEP
    work.mkdir(parents=True, exist_ok=True)
    state = common.load_state(work, identity, args.restart)
    if not state["cases"]:           # 새로 시작한다: 이전 실행이 남긴 ref의 그림 정보를 지운다
        (work / "ref_features.pt").unlink(missing_ok=True)
    run_all(rt, args, names, plan, state, work)
    finalize(rt, args, state, work, checks, inputs, plan)


if __name__ == "__main__":
    main()
