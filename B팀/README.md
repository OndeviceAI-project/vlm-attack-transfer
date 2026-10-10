# B팀 — Qwen2-VL-2B-Instruct

## 진행 현황 (단계가 끝날 때마다 고친다)

| 단계 | 상태 | 담당 | 날짜 | 결과 위치 / 이슈 |
|---|---|---|---|---|
| GPU 확보 (bf16, 16GB 이상) | 안 함 | | | |
| 0 환경 점검 | 안 함 | | | |
| 1 사진 준비 | 안 함 | | | `data/clean`, `data/noise` |
| go/no-go (dev 5장) — 10/15까지 | 안 함 | | | |
| 2 공격 사진 30장 | 안 함 | | | `data/adv` + 로그 + 해시 |
| 3 전이 실험 phase1 | 안 함 | | | `results/` |
| 4 분석 (공동) | 안 함 | | | |

상태는 `안 함 / 진행 중 / 끝 / 막힘` 중 하나.

## 폴더

```
code/       이 팀의 코드. A팀 코드와 이미지생성/attack_smolvlm.py를 복사해 아래 항목만 고친 것
data/       clean / noise / adv
results/    실행마다 CSV + JSON
reports/    이 팀 그림·메모
```

## 1. SmolVLM과 다른 점

| 항목 | SmolVLM (A) | Qwen2-VL-2B (B) |
|---|---|---|
| 모델 | HuggingFaceTB/SmolVLM-256M-Instruct | Qwen/Qwen2-VL-2B-Instruct |
| 사진 크기 | 512, 조각내기 끔 | **448×448** (28의 배수) |
| 프로세서 설정 | `do_image_splitting=False, size={"longest_edge": 512}` | `min_pixels = max_pixels = 448*448` |
| 이미지 토큰 | 64 | 256 (14px 조각 1024개를 2×2로 합침) |
| 입력 길이 | 76 토큰 | 284 토큰 (시스템 문장 포함) |
| 정규화 | 평균 0.5, 표준편차 0.5 | CLIP 값 (프로세서의 `image_mean`, `image_std`) |
| 공격 계산 정밀도 | fp32 | **bf16** |
| 종료 토큰 | 1개 (49279) | **2개** (`<|im_end|>`, `<|endoftext|>`) — generation_config 목록 전부 |
| 채팅 양식 | 질문만 | "You are a helpful assistant." 시스템 문장이 자동으로 붙음. **그대로 둔다** |

448을 고른 이유: Qwen2-VL은 사진 크기에 따라 토큰 수가 달라진다. 크기를 못 박지 않으면 사진마다 입력 길이가 달라져 비교가 흐려진다. 448은 28의 배수 중 512와 비슷하고 메모리가 감당되는 값이다.

## 2. pixel_values 모양

SmolVLM은 `(1, 1, 3, 512, 512)`로 사진이 그대로 들어간다. Qwen2-VL은 `(1024, 1176)` — 사진을 14×14 조각으로 잘라 한 줄로 편 것이고, 같은 사진을 **2프레임 영상처럼 복사**해서 넣는다(1176 = 3채널 × 2프레임 × 14 × 14). 모델 입력에 `image_grid_thw = [[1, 32, 32]]`가 같이 들어간다.

공격 코드에서 pixel_values를 직접 만들 때는 transformers의 `Qwen2VLImageProcessor.patchify` 순서를 그대로 따른다. transformers 5.19에서 확인된 순서: 정규화 → `reshape(1, 3, 16, 2, 14, 16, 2, 14)` → `permute(0, 2, 5, 3, 6, 1, 4, 7)` → 2프레임 복사 → `reshape(1024, 1176)`. 이렇게 하면 프로세서와 차이가 0이다. 차이 0을 확인하기 전에는 공식 실행을 시작하지 않는다.

## 3. 부품 이름 (census의 PARTS)

| 글자 | Qwen2-VL-2B 모듈 | Linear 수 (대략, 0단계에서 실제 값으로) |
|---|---|---|
| V 비전 인코더 | `model.visual.blocks` | 128 (블록 32 × 4) |
| C 연결부 | `model.visual.merger` | 2 |
| L 언어모델 | `model.language_model` | 196 (층 28 × 7) |
| H 출력층 | `lm_head` | 1 |

`model.visual.patch_embed`는 Conv라 대상이 아니다. census에서 "어디에도 안 속하는 Linear"가 나오면 `settings.MODEL_SPECIFIC["qwen2vl"]["parts"]`를 고친다.

**주의 — 출력층(H).** Qwen2-VL-2B는 입력 임베딩과 출력층이 가중치를 공유(tie)하는 것으로 알려져 있다(`config.tie_word_embeddings` 확인). `quant_rtn.apply_rtn`은 weight를 새 Parameter로 갈아 끼우므로 입력 임베딩은 건드리지 않지만, 다른 방법을 쓰면 같이 바뀐다. 1차(V만)에서는 상관없다.

## 4. go/no-go (10/15까지)

dev 5장으로 `이미지생성/README.md` 규칙 그대로 공격 사진을 만들어 원본 bf16 Qwen2-VL-2B에서 **5장 중 3장 이상**이 상한에 닿으면 go. 2장 이하면 10/16 회의에서: 반복 수 500까지만 올려 다시 → 그래도 안 되면 모델 교체 검토(같은 계열의 큰 SmolVLM이면 A팀 코드를 거의 그대로) → "2B에서는 같은 예산으로 성립하지 않았다"도 결과로 보고 가능.

## 5. 메모리·시간·GPU

- 모델 bf16 약 4.4GB. 공격(역전파) 때 16GB에서 돌아갈 것으로 예상. 안 되면 `use_cache=False` 확인 → 사진 336(토큰 144) 축소를 설정 변경 절차로 제안.
- 공격 사진 1장 수 분~20분(상한). 30장은 밤새. 평가는 경우당 15분 안팎, phase1 8경우 약 2시간.
- bnb 경우는 CUDA에서만. `llm_int8_skip_modules`에 V가 아닌 세 부품 이름.
- bf16이 되는 GPU 필요(RTX 30/40/50, A10, L4, A100). Colab 무료 T4는 안 됨.

## 6. 0단계 체크리스트 (B팀 추가분)

- [ ] 448 사진이 `image_grid_thw = [[1, 32, 32]]`, 이미지 토큰 256개로 들어가는가
- [ ] 전처리 일치 차이 0
- [ ] 종료 토큰 2개 모두 종료로 처리되는가 (한 문장 답이 `ended_by_eos=True`)
- [ ] 시스템 문장이 붙은 템플릿을 공격과 평가가 똑같이 쓰는가
- [ ] `tie_word_embeddings` 값 확인
- [ ] census: V 128 / C 2 / L 196 / H 1 (실제 값으로 아래 갱신)
- [ ] 같은 사진 두 번 → 같은 답

## 7. 바꾼 항목 기록 (검수자가 본다)

| 항목 | A팀 값 | B팀 값 | 이유 |
|---|---|---|---|
| 모델 | SmolVLM-256M-Instruct | Qwen2-VL-2B-Instruct | |
| 사진 크기 | 512 | 448 | 1절 |
| (추가) | | | |

0단계 결과: GPU / 버전 · 전처리 일치 차이 · census · 종료 토큰 · tie_word_embeddings · 재현성 → 여기에 적는다.
