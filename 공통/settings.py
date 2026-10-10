"""두 팀이 똑같이 쓰는 값. 바꿔야 하면 두 팀이 합의하고 SETTINGS_VERSION을 올린다.

쓰는 법:  import sys; sys.path.insert(0, "<저장소>/공통");  import settings, quant_rtn
"""

SETTINGS_VERSION = "v1"

# 질문과 답 생성 — generate()에 GEN_KWARGS를 그대로 넘긴다. 모델 기본값(샘플링 등)에 기대지 않는다
PROMPT = "Describe this image in one sentence."
MAX_NEW_TOKENS = 512
GEN_KWARGS = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
              "repetition_penalty": 1.0, "temperature": None, "top_p": None, "top_k": None}
# 종료 토큰은 모델 generation_config의 eos_token_id 전부 (Qwen은 2개)

# 공격 사진 (attack_smolvlm.py 인자와 같은 이름)
EPS = 8                    # 노이즈 상한, 0~255 눈금
STEP = 1
ATTACK_ITERS = 300         # --iter
ATTACK_MAX_ITERS = 500     # 이보다 많이 돌리지 않는다
TARGET_LEN = 128           # --target_len
REFRESH = 20               # --refresh
W_UNC = 0.05               # --w_unc
W_DIV = 0.0                # --w_div

# 노이즈 대조군: 부호는 numpy.random.default_rng([NOISE_SEED, coco_id]).choice([-1, 1], size=arr.shape)
NOISE_SEED = 2026

# 양자화: 비전 인코더의 Linear weight만. rtn은 quant_rtn.apply_rtn (그룹 64), bnb는 bitsandbytes 8·4
QUANT_GROUP = 64
CASES_PHASE1 = ["ref", "rtn-V-8", "rtn-V-6", "rtn-V-5", "rtn-V-4", "rtn-V-3", "bnb-V-8", "bnb-V-4"]
CASES_PHASE2 = ["ref"] + [f"rtn-{p}-{b}" for p in ("C", "L", "H", "VCLH") for b in (8, 4, 3)]   # 2차, 부품별

# 사진
N_EVAL = 30                # image_list.csv의 split=eval
N_DEV = 5                  # split=dev. 코드 확인·조정용

# 모델마다 다른 값
MODELS = {
    "smolvlm": {
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "img_size": 512,
        "processor_kwargs": {"do_image_splitting": False, "size": {"longest_edge": 512}},
        "attack_dtype": "float32",
        "parts": {"V": "model.vision_model", "C": "model.connector",
                  "L": "model.text_model", "H": "lm_head"},
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "img_size": 448,
        "processor_kwargs": {"min_pixels": 448 * 448, "max_pixels": 448 * 448},
        "attack_dtype": "bfloat16",
        "parts": {"V": "model.visual.blocks", "C": "model.visual.merger",
                  "L": "model.language_model", "H": "lm_head"},
    },
}
PART_NAMES = {"V": "비전 인코더", "C": "연결부", "L": "언어모델", "H": "출력층"}
