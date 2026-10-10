"""두 팀(A: SmolVLM, B: Qwen2-VL)이 반드시 같은 값을 쓰는 실험 조건.

규칙(docs/02_공통규칙.md):
- 이 파일의 값은 혼자 바꾸지 않는다. 바꿔야 하면 두 팀이 합의하고, SETTINGS_VERSION을 올리고, 문서를 같이 고친다.
- 결과 CSV마다 SETTINGS_VERSION을 기록한다. 버전이 다른 결과는 섞어서 분석하지 않는다.
- 모델마다 달라도 되는 값(사진 크기, 부품 이름)은 MODEL_SPECIFIC에 있다.
"""

SETTINGS_VERSION = "v1"   # 2026-10-10 초안. 두 팀 확인 후 v1 확정

# ── 질문과 답 생성 ──────────────────────────────────────────────
PROMPT = "Describe this image in one sentence."
MAX_NEW_TOKENS = 512          # 답 길이 상한. 512에 닿으면 '스스로 멈추지 못함'으로 본다

# generate()에 매번 그대로 넘긴다. 모델의 generation_config 기본값(샘플링, 반복 벌점 등)에 기대지 않는다
GEN_KWARGS = {
    "do_sample": False,       # greedy: 같은 입력이면 항상 같은 답
    "num_beams": 1,
    "max_new_tokens": MAX_NEW_TOKENS,
    "repetition_penalty": 1.0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}
# 종료 토큰: 모델 generation_config의 eos_token_id 목록 전부를 종료로 취급한다 (Qwen은 2개)

# ── 공격 사진 (위협 모델의 일부라 바꾸지 않는다) ──────────────────
EPS = 8                   # 픽셀 변화 상한(L∞), 0~255 눈금. 저장된 PNG를 다시 읽어 확인한다
STEP = 1                  # 한 걸음 크기, 0~255 눈금
ATTACK_ITERS = 300        # 기본 반복 수 (A팀 1번 노트북의 --iter)
ATTACK_MAX_ITERS = 500    # 상한. 어떤 사진도 이보다 많이 돌리지 않는다
ATTACK_MAX_MINUTES = {"smolvlm": 5, "qwen2vl": 20}   # 사진 한 장 시간 상한. 넘기면 그때까지의 최선을 저장
CHECK_EVERY = 20          # 이 걸음마다 실제 답 길이를 재고 기록한다 (--refresh)
TARGET_LEN = 128          # 1번 노트북의 --target_len
W_UNC = 0.05              # 1번 노트북의 --w_unc (--w_div는 0으로 둔다)
STOP_AFTER_HITS = 2       # 답이 상한에 연속으로 이 횟수만큼 닿으면 조기 종료

# ── 노이즈 대조군 ───────────────────────────────────────────────
NOISE_SEED = 2026         # 사진마다 (NOISE_SEED, COCO id)로 난수를 만든다. 부호만 무작위인 ±EPS 노이즈

# ── 양자화 ─────────────────────────────────────────────────────
REF_DTYPE = "bfloat16"    # 기준(원본) 모델과 양자화하지 않은 부품의 정밀도
QUANT_GROUP = 64          # 가짜 양자화(rtn)의 그룹 크기. MLX 기본값과 같다
QUANT_TARGET = "Linear 층의 weight만. bias·LayerNorm·Conv·임베딩은 원본 그대로"

# 경우 이름: ref = 양자화 없음 / <방법>-<부품>-<비트>
#   방법  rtn = 그룹 64 균등(min/max) 반올림 가짜 양자화 (비트 자유, CUDA·MPS 모두 가능)
#         bnb = bitsandbytes 실제 양자화 (8 = LLM.int8, 4 = nf4. CUDA 전용)
#   부품  V 비전 인코더, C 연결부, L 언어모델, H 출력층. 여러 개면 붙여 쓴다 (VCLH = 전체)
PRESETS = {
    # 1차: 비전 인코더 고정, 비트만 바꾼다. bnb 두 개는 예비 실험과 이어지는 기준점
    "phase1": ["ref", "rtn-V-8", "rtn-V-6", "rtn-V-5", "rtn-V-4", "rtn-V-3", "bnb-V-8", "bnb-V-4"],
    # 1차 여유분: 평가만 더 돌리면 되므로 싸다. 실제 배포 형식(전체 양자화)과 '망가진 모델' 끝점
    "phase1-extra": ["ref", "rtn-VCLH-8", "rtn-VCLH-4", "rtn-V-2"],
    # 2차: 부품별 × 비트. LH = 비전은 원본(llama.cpp 방식), VCLH = 전체(MLX 방식)
    "phase2": ["ref"] + [f"rtn-{p}-{b}" for p in ("C", "L", "H", "LH", "VCLH") for b in (8, 4, 3)],
}

# ── 사진 수 ─────────────────────────────────────────────────────
N_DEV = 5                 # 개발용. 코드 확인·디버깅에만 쓴다
N_EVAL = 30               # 1차 공식 실험. 유효성 검사(docs/03 1단계)를 통과한 앞 30장
N_EVAL_MAX = 48           # 후보 사진 전체 수. 이보다 늘리려면 사진 목록을 새 버전으로 만든다

# ── 모델마다 다른 값 ─────────────────────────────────────────────
MODEL_SPECIFIC = {
    "smolvlm": {
        "team": "A",
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "img_size": 512,
        "processor": "do_image_splitting=False, size={'longest_edge': 512}",
        "attack_dtype": "float32",
        "parts": {"V": "model.vision_model", "C": "model.connector",
                  "L": "model.text_model", "H": "lm_head"},
    },
    "qwen2vl": {
        "team": "B",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "img_size": 448,
        "processor": "min_pixels=max_pixels=448*448",
        "attack_dtype": "bfloat16",
        "parts": {"V": "model.visual.blocks", "C": "model.visual.merger",
                  "L": "model.language_model", "H": "lm_head"},
    },
}
