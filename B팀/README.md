# B팀 — Qwen2-VL-2B-Instruct

## 목표

10/30까지 **Qwen2-VL-2B에서 "비트 수 → 전이율" 선 하나**를 만든다. 비전 인코더를 8·6·5·4·3비트(+ bitsandbytes 8·4)로 양자화한 모델에, 원본 모델로 만든 공격 사진 30장이 몇 % 살아남는지.

돌릴 경우는 여덟 개(`settings.PRESETS["phase1"]`): `ref`(bf16 원본), `rtn-V-8`, `rtn-V-6`, `rtn-V-5`, `rtn-V-4`, `rtn-V-3`, `bnb-V-8`, `bnb-V-4`. 경우마다 사진 30장 × 입력 3종(clean, noise, adv) = 답 90개.

## 내야 하는 것

| 산출물 | 어디에 | 형식 |
|---|---|---|
| 공격 사진 30장 + `attack_log.csv` + `SHA256SUMS` | `B팀/data/adv/` | `이미지생성/README.md` |
| 결과 CSV + json (`ref` 포함) | `B팀/results/` | `공통규칙/결과형식.md` 1절 (team=B, model=qwen2vl) |
| 실험기록 이슈 | GitHub 이슈 | 템플릿 "실험기록" |

맞춰야 하는 값은 `공통규칙/README.md`의 표 하나뿐이고, 코드에서 `settings`를 import하면 된다. 그 외 코드 형태·순서·기계는 자유다.

## 할 일 순서

1. **GPU.** bf16이 되는 GPU(RTX 30/40/50, A10, L4, A100…), 메모리 16GB 이상 권장. Colab 무료 T4는 bf16이 안 돼 공식 실험에 못 쓴다. 안 구해지면 바로 이슈 → A팀 기계 밤 시간 공유.
2. **환경.** `pip install torch transformers accelerate bitsandbytes pillow numpy pandas`. A팀 기계에서 확인된 조합은 torch 2.11(cu128) / transformers 5.19 / bitsandbytes 0.50 / pillow 11.2. 주 버전(transformers 5.x)만 맞추고, 정확한 버전은 결과 json에 기록만 한다.
3. **코드 가져오기.** `이미지생성/attack_smolvlm.py`를 `B팀/code/`로 복사해 아래 "참고"대로 Qwen에 맞게 고친다. 평가(3단계) 코드는 `A팀/참고_예비실험/2_비전인코더_양자화_전이확인.ipynb`를 출발점으로 삼아도 되고 직접 짜도 된다. 노트북 맨 위에 `sys.path.insert(0, "<저장소>/공통규칙")` 후 `import settings, quant_rtn`.
4. **사진.** `공통규칙/image_list.csv`의 dev 5장과 eval 후보 43장을 url로 받아 448×448로 리사이즈해 `data/clean/`, 노이즈 대조군은 `data/noise/`(만드는 법: `공통규칙/진행단계.md` 1단계). 평가에 쓸 30장은 A팀이 10/14에 `공통규칙/eval_v1.txt`로 확정한다. 그 전에는 dev 5장으로만 작업한다.
5. **0단계 확인 세 가지**(공식 실행 전에 한 번): 전처리 일치 차이 0, census, 같은 사진 두 번 → 같은 답. 결과를 아래 "바꾼 것 기록"에 적는다.
6. **go/no-go (10/15까지).** dev 5장으로 공격 사진을 만들어(지금 있는 `attack_smolvlm.py`를 Qwen에 맞게 고친 것으로 충분) 원본 bf16에서 **3장 이상**이 512에 닿으면 go. 2장 이하면 10/16 회의: 반복 수 500까지 → 모델 교체 검토 → "2B에서는 같은 예산으로 성립하지 않았다"도 결과로 쓸 수 있다.
7. **공격 사진 30장.** `eval_v1.txt`가 나오고 A팀이 `이미지생성/README.md`의 "손볼 것"을 반영한 뒤, 그 파일을 다시 복사해 30장을 만든다(사진당 20분 상한, 밤새). 저장본 확인 → `SHA256SUMS` → PR. 이후 `data/adv/`는 손대지 않는다.
8. **전이 실험 phase1.** 경우마다 모델을 올리고 → `rtn-*`은 `quant_rtn.apply_rtn(model, settings.MODEL_SPECIFIC["qwen2vl"]["parts"], "V", bits)`, `bnb-*`는 `BitsAndBytesConfig(..., llm_int8_skip_modules=["model.visual.merger", "model.language_model", "lm_head"])` → `quant_rtn.census`로 V만 양자화됐는지 확인 → 90개를 돌려 답마다 CSV 한 줄(`settings.GEN_KWARGS` 그대로) → json. 실행 ID는 `20261022-210000_qwen2vl_phase1` 꼴.
9. **올리기.** 브랜치 `b/<내용>` → PR(템플릿 체크리스트) → A팀 1명이 확인 → 병합. 결과마다 실험기록 이슈 번호를 커밋 메시지에.

## 진행 현황 (끝날 때마다 고친다)

| 단계 | 상태 | 담당 | 날짜 | 위치 / 이슈 |
|---|---|---|---|---|
| 1 GPU 확보 | 안 함 | | | |
| 2~3 환경·코드 | 안 함 | | | `code/` |
| 4 사진 준비 | 안 함 | | | `data/clean`, `data/noise` |
| 5 0단계 확인 | 안 함 | | | 아래 표 |
| 6 go/no-go (10/15) | 안 함 | | | |
| 7 공격 사진 30장 고정 | 안 함 | | | `data/adv/` |
| 8 전이 실험 phase1 | 안 함 | | | `results/` |

상태는 `안 함 / 진행 중 / 끝 / 막힘`. 막히면 이슈 번호.

## 참고 — Qwen2-VL에서 미리 확인해 둔 것

규칙이 아니라 시간을 아끼라고 적어 둔 것이다. 다른 방법으로 해도 0단계 확인만 통과하면 된다.

**사진과 토큰.** 프로세서를 `min_pixels=max_pixels=448*448`로 만들면 448×448 사진이 14px 조각 1024개 → 이미지 토큰 256개, 입력 284토큰, `image_grid_thw = [[1, 32, 32]]`. 크기를 고정하지 않으면 사진마다 토큰 수가 달라진다.

**pixel_values 모양.** `(1024, 1176)`. 사진을 조각으로 잘라 한 줄로 편 것이고 같은 사진을 2프레임처럼 복사한다(1176 = 3 × 2 × 14 × 14). 직접 만들 때 transformers 5.19 기준: 정규화(CLIP 평균·표준편차, 프로세서의 `image_mean`·`image_std`) → `reshape(1, 3, 16, 2, 14, 16, 2, 14)` → `permute(0, 2, 5, 3, 6, 1, 4, 7)` → 2프레임 복사 → `reshape(1024, 1176)`. 이렇게 하면 프로세서 결과와 차이 0.

**프로세서 출력.** `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`에 더해 `mm_token_type_ids`가 나온다. 모델에 넣는 텐서들의 길이를 `input_ids`와 맞춘다(SmolVLM의 `pixel_attention_mask` 자리에 `image_grid_thw`).

**종료 토큰 2개.** `<|im_end|>`, `<|endoftext|>`. generation_config의 목록을 그대로 쓴다. 하나만 보면 끝난 답을 안 끝난 것으로 센다.

**채팅 양식.** "You are a helpful assistant." 시스템 문장이 자동으로 붙는다. 그대로 둔다(공격과 평가가 같은 양식이면 된다).

**부품 이름** (`settings.MODEL_SPECIFIC["qwen2vl"]["parts"]`): V `model.visual.blocks`(Linear 약 128), C `model.visual.merger`(2), L `model.language_model`(약 196), H `lm_head`(1). 숫자는 census로 확인해 아래에 적는다. 2B는 입력 임베딩과 출력층이 가중치를 공유할 수 있는데(`tie_word_embeddings`), `apply_rtn`은 새 Parameter로 갈아 끼우므로 1차(V만)에서는 상관없다.

**정밀도·메모리·시간.** 공격 생성은 bf16(2B를 fp32로 역전파하면 메모리가 모자람). 모델 bf16 4.4GB, 16GB GPU에서 될 것으로 보이고 안 되면 사진 336으로 줄이는 것을 이슈로 제안. 공격 사진 1장 수 분~20분, 30장은 밤새. 평가는 경우당 15분 안팎(공격 사진이 512토큰까지 가므로), phase1 8경우 약 2시간. bnb 경우는 CUDA에서만 된다.

## 바꾼 것 기록 (검수자가 본다)

| 항목 | A팀 값 | B팀 값 |
|---|---|---|
| 모델 / 사진 크기 | SmolVLM-256M / 512 | Qwen2-VL-2B / 448 |
| 공격 계산 정밀도 | fp32 | bf16 |
| (추가) | | |

0단계 확인 결과: GPU · 패키지 버전 · 전처리 일치 차이 · census(V/C/L/H) · 종료 토큰 · `tie_word_embeddings` · 재현성 → 여기에.
