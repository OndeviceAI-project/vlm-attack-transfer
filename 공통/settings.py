"""두 팀이 똑같이 쓰는 값. 바꿔야 하면 두 팀이 합의하고 SETTINGS_VERSION을 올린다.
공격 사진이나 결과를 하나라도 만든 뒤에 값을 바꾸면, 버전을 올리고 그 값의 영향을 받는 단계를 다시 한다.
단, cand를 늘릴 때(README 3단계 4번)는 N_CAND와 image_list.csv만 바꾸고 버전은 올리지 않는다. 이미 만든 사진과 기록은 그대로 이어서 쓴다.
세트(공통/attack_set.csv)를 정하기 전에 N_ATTACK_SET을 바꿀 때도 버전은 올리지 않는다(README 3단계 3번).
4~7단계 결과가 하나도 없을 때 경우 목록에 경우를 더하는 것도 버전을 올리지 않는다(README 5-2단계). 공격 사진과 시도 기록은 경우 목록과 관계없다.
9단계용 값(CASES_DIRECT, N_DIRECT, CASES_DIRECT_TARGETS)을 더하거나, 9-1단계 본 실행 전에 바꾸는 것도 버전을 올리지 않는다(README 9단계). 1~7단계의 값과 결과는 그대로다.
9-1단계 본 실행을 시작한 뒤에 N_DIRECT를 줄일 때도 버전은 올리지 않는다. 이미 만든 사진과 기록은 그대로 두고 set_order 1~N_DIRECT만 쓴다(README 9단계 '순서와 고정').

쓰는 법:  import sys; sys.path.insert(0, "<저장소>/공통");  import settings, quant_rtn
"""

SETTINGS_VERSION = "v2"

# 질문과 답 생성 — generate()에 GEN_KWARGS와 eos_token_id=MODELS[모델]["eos_token_ids"]를 그대로 넘긴다.
# 모델 기본값(샘플링 등)에 기대지 않는다
PROMPT = "Describe this image in one sentence."
MAX_NEW_TOKENS = 512
GEN_KWARGS = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
              "repetition_penalty": 1.0, "temperature": None, "top_p": None, "top_k": None}
DTYPE = "bfloat16"               # ref·rtn·rand 모델의 dtype이자 bnb 4비트의 계산 dtype
ATTN_IMPLEMENTATION = "sdpa"
SEED = 2026                      # torch.manual_seed

# 공격 사진 (attack_smolvlm.py 인자와 같은 이름). 만드는 방법은 이미지생성/README.md
EPS = 8                    # 노이즈 상한, 0~255 눈금
STEP = 1
ATTACK_ITERS = 500         # --iter. 시도 한 번의 최대 반복
ATTACK_MAX_ITERS = 500     # 이보다 많이 돌리지 않는다
TARGET_LEN = 128           # --target_len
REFRESH = 20               # --refresh
W_UNC = 0.05               # --w_unc
W_DIV = 0.0                # --w_div
ATTACK_MAX_TRIES = 2       # 사진 한 장당 최대 시도 횟수. 1번째는 노이즈 0에서, 2번째는 무작위로 시작(이미지생성/README.md)
ATTACK_SEED = 2026         # 무작위 시작의 씨앗. 사진 번호·시도 번호와 섞어 쓴다
N_ATTACK_SET = 200         # 두 팀이 함께 쓰는 공격 사진 세트의 사진 수. 두 팀 공통 성공 후보를 order 순서로 앞에서부터

# 노이즈 대조군 사진: 부호는 numpy.random.default_rng([NOISE_SEED, coco_id]).choice([-1, 1], size=arr.shape)
NOISE_SEED = 2026

# 압축 경우. 이름은 <방식>-<부품>-<비트>.
# rtn = quant_rtn.apply_rtn, rand = quant_rtn.apply_rand(같은 크기의 무작위 흔들기 대조군), bnb = bitsandbytes 8(LLM.int8)·4(nf4)
QUANT_GROUP = 64
RAND_SEED = 2026
RTN_BITS = (8, 6, 5, 4, 3)
BNB_BITS = (8, 4)
CASES_VISION = (["ref"] + [f"rtn-V-{b}" for b in RTN_BITS] + [f"bnb-V-{b}" for b in BNB_BITS]
                + [f"rand-V-{b}" for b in RTN_BITS])                  # 4단계 비전 인코더 실험, 13경우
CASES_LANGUAGE = (["ref"] + [f"rtn-L-{b}" for b in RTN_BITS] + [f"bnb-L-{b}" for b in BNB_BITS]
                  + [f"rand-L-{b}" for b in RTN_BITS])                # 5단계 언어모델 실험, 13경우
CASES_BOTH = (["ref"] + [f"rtn-VL-{b}" for b in RTN_BITS]
              + [f"bnb-VL-{b}" for b in BNB_BITS])                    # 5-2단계 함께 압축 실험, 8경우(rand 없음)
CASES_ALL = CASES_VISION + CASES_LANGUAGE[1:] + CASES_BOTH[1:]       # 1-3·6·7단계, 32경우(ref는 한 번)

# 9단계: 압축 모델에서 만든 공격 사진(README 9단계). 만드는 값은 위 '공격 사진' 값을 그대로 쓴다(9단계용 값을 따로 두지 않는다)
CASES_DIRECT = ["rtn-V-4", "rtn-V-3", "rand-V-3", "bnb-V-4"]         # 출발 경우: 공격 사진을 만드는 모델. 두 팀이 같다
N_DIRECT = 30                                                        # 쓰는 사진: attack_set.csv의 set_order 1~N_DIRECT
CASES_DIRECT_TARGETS = CASES_VISION                                  # 도착 경우: 만든 사진을 넣어 보는 모델, 13경우

# rtn 가중치 변화(%) 확인값(README 6장과 같은 값). 1-3단계에서 잰 값이 CHANGE_TOL 안에서 같아야 한다.
# rand는 같은 부품·비트의 rtn을 잰 값과 CHANGE_TOL 안에서 같아야 한다
RTN_CHANGE_PCT = {
    "smolvlm": {"V": {8: 0.57, 6: 2.24, 5: 4.54, 4: 9.38, 3: 20.10},
                "L": {8: 0.57, 6: 2.21, 5: 4.48, 4: 9.26, 3: 19.86}},
    "qwen2vl": {"V": {8: 0.57, 6: 2.21, 5: 4.49, 4: 9.27, 3: 19.88},
                "L": {8: 0.57, 6: 2.21, 5: 4.49, 4: 9.27, 3: 19.87}},
}
CHANGE_TOL = 0.01          # %p

# 사진 (image_list.csv의 split)
N_DEV = 5                  # dev: 코드 점검용. 공격 사진 설정을 조정할 때도 이것만 쓴다
N_CAND = 300               # cand: 공격 사진 세트 후보. 두 팀이 각자 전부 order 순서대로 처리한다
N_ACC = 300                # acc: 정확도 세트(공격 사진 없음)

# 모델마다 다른 값. revision·eos_token_ids·linear_count는 확인값이다 — 실제와 다르면 멈춘다
MODELS = {
    "smolvlm": {
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
        "img_size": 512,
        "processor_kwargs": {"do_image_splitting": False, "size": {"longest_edge": 512}},
        "attack_dtype": "bfloat16",
        "eos_token_ids": [49279],
        "parts": {"V": "model.vision_model", "C": "model.connector",
                  "L": "model.text_model", "H": "lm_head"},
        "linear_count": {"V": 72, "C": 1, "L": 210, "H": 1},
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": "895c3a49bc3fa70a340399125c650a463535e71c",
        "img_size": 448,
        "processor_kwargs": {"min_pixels": 448 * 448, "max_pixels": 448 * 448},
        "attack_dtype": "bfloat16",
        "eos_token_ids": [151645, 151643],
        "parts": {"V": "model.visual.blocks", "C": "model.visual.merger",
                  "L": "model.language_model", "H": "lm_head"},
        "linear_count": {"V": 128, "C": 2, "L": 196, "H": 1},
    },
}
PART_NAMES = {"V": "비전 인코더", "C": "연결부", "L": "언어모델", "H": "출력층"}
