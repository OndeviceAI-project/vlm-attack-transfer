# A팀 — SmolVLM-256M 코드

예비 실험 코드를 그대로 옮긴 것이다(결과 셀은 비움). 공격 사진을 만드는 **유일한 구현은 `attack_smolvlm.py`**이고, B팀은 이 파일을 복사해 아래 "B팀이 바꿀 곳"만 고친다.

## 파일과 파이프라인 단계(docs/03)의 대응

| 단계 | 파일 | 공식 실행 전에 고칠 것 |
|---|---|---|
| 0 환경 점검 | (새로 만들 것) | docs/02 E 항목. 아래 TODO 1 반영 뒤 전처리 일치 차이 0 확인 |
| 1 사진 준비 | `1_…ipynb` 앞부분 | 사진 3장 → `data/image_list.csv` 48장, 유효성 검사, noise 세트 추가 |
| 2 공격 사진 | **`attack_smolvlm.py`** (+ `1_…ipynb`의 저장본 확인 셀) | 아래 TODO 1~7 |
| 3 전이 실험 | `2_…ipynb` | CASES를 `settings.PRESETS["phase1"]`로. rtn 경우는 `common/quant_rtn.py`의 `apply_rtn`으로 추가하고 노트북 안의 census 함수는 같은 파일의 것으로 교체. 결과 CSV 열을 docs/05 §1로. 각 답마다 시간·종료 여부 기록 |
| 4 분석 | (새로) | docs/05 지표, 그림 2·3·4 |
| 5 원인 분석 | `6_…ipynb` | V 비트별 경우 × 30장 |
| 2차 참고 | `3_`, `4_`, `5_…ipynb` | 부품별 실험에서 재사용 |

0단계 기준값: Linear 수 비전 인코더 72 / 연결부 1 / 언어모델 210 / 출력층 1. 종료 토큰 49279. 입력 76토큰(이미지 토큰 64).

## attack_smolvlm.py 사용법

```
python attack_smolvlm.py --images <원본 PNG 폴더> --out <출력 폴더> --iter 300 2>&1 | tee <출력 폴더>/attack_stdout.txt
```

| 인자 | 뜻 | v1 값 | 바꿔도 되나 |
|---|---|---|---|
| `--images` | 원본 사진 폴더 | `data/sets/smolvlm/clean`에서 `eval_v1.txt`의 30장만 모은 폴더 | 경로만 |
| `--out` | 공격 사진이 저장될 폴더 | `data/sets/smolvlm/adv` | 경로만 |
| `--iter` | 반복 수 | **300** (스크립트 기본값 200은 예비 실험 값. 상한 500) | 300으로 고정해서 실행 |
| `--eps` | 노이즈 상한 (0~255 눈금) | 8 | 금지 |
| `--step` | 걸음 크기 | 1 | 금지 |
| `--target_len` | 목표 답 길이 | 128 | 금지 |
| `--refresh` | 확인 주기 | 20 | 금지 |
| `--w_unc` | 손실 가중치 | 0.05 | 금지 |
| `--w_div` | 손실 가중치 | 0 | 금지 |
| `--overwrite` | 이미 있는 공격 사진을 다시 만듦 | 쓰지 않음 | 세트 고정 뒤 금지 |

운영상 알아둘 것:

- 출력 폴더에 같은 이름의 PNG가 이미 있으면 그 사진은 건너뛴다. 그래서 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 한다.
- 진행 상황은 화면에만 찍힌다. `attack_log.csv`(TODO 5)가 생기기 전까지는 반드시 `tee`로 파일에 남긴다.
- 장치는 cuda → mps → cpu 순서로 자동 선택, 모델은 fp32로 올린다. 공격 생성은 GPU 기계에서 하고, 결과 PNG만 저장소로 옮기면 평가는 어디서든 된다.
- 입력 사진은 스크립트 안에서 512×512로 다시 맞추지만, 1단계에서 만든 clean PNG는 이미 512라 바뀌는 것이 없다.
- `--images`에 48장을 다 넣지 않도록 `eval_v1.txt`의 30장만 링크·복사한 폴더를 따로 만든다(dev 5장은 다른 폴더, 다른 `--out`).

## 공식 실행 전에 맞출 것 (A팀 TODO, 순서대로)

1. **프로세서 설정.** 스크립트와 노트북 2~6 모두, 프로세서를 만들 때 `do_image_splitting=False`에 더해 `size={"longest_edge": 512}`를 준다. 기본값(2048)이면 512 사진을 키웠다 줄여 픽셀이 평균 1.3/255·최대 24/255 달라진다(docs/02 A). 바꾼 뒤 0단계 '전처리 일치' 차이가 0인지 확인한다.
2. **고정값 출처를 한 곳으로.** `MODEL_ID`, `QUESTION`, `IMG_SIZE`와 인자 기본값을 `common/settings.py`에서 읽게 한다. 값은 그대로, 출처만 바꾼다. (스크립트와 노트북은 저장소 루트 아래 두 단계에 있으므로 `sys.path`에 루트를 추가한 뒤 `from common import settings`.)
   - 0단계 '전처리 일치' 검사 방법: 같은 사진에 대해 스크립트의 `make_pixel_values(x, 0, mean, std, 1)`와 `processor(text=..., images=[사진])["pixel_values"]`의 차이 최댓값을 출력한다. 0이어야 한다.
3. **저장 시점.** 지금은 마지막 걸음의 사진을 저장한다. 확인 주기마다 잰 답 길이가 가장 길었던 시점의 사진을 저장하도록 바꾼다(docs/04 §3-3). 두 팀이 같은 코드를 써야 하므로 B팀이 복사하기 **전에** 끝낸다.
4. **시간 상한.** 사진 한 장이 5분(Qwen 20분)을 넘으면 멈추고 그때까지의 최선을 저장, 사유를 로그에 쓴다(docs/04 §2).
5. **로그.** 화면 출력에 더해 `attack_log.csv`에 docs/05 §3 열로 사진마다 한 줄. 저장본을 다시 읽어 잰 `linf`와 `saved_len`도 여기에 넣는다(1번 노트북의 저장본 확인 셀이 하던 일).
6. **저장본 확인.** 1번 노트북 뒷부분(노이즈 최대 ≤ 8, fp32·bf16에서의 답 길이) 셀은 그대로 쓴다. 8을 넘는 사진이 하나라도 있으면 세트를 쓰지 않는다.
7. **고정.** 30장이 끝나면 `shasum -a 256 *.png > SHA256SUMS`, PNG·로그·해시를 한 PR로.

## B팀이 이 스크립트에서 바꿀 곳

docs/07의 내용을 이 파일 기준으로 다시 적은 것이다. 바꾼 뒤 0단계 전처리 일치 검사를 통과해야 한다.

| 바꿀 곳 | 지금 (SmolVLM) | Qwen2-VL-2B | 참고 |
|---|---|---|---|
| `MODEL_ID` | SmolVLM-256M-Instruct | Qwen/Qwen2-VL-2B-Instruct | |
| `IMG_SIZE` | 512 | 448 | 28의 배수 |
| `main()`의 프로세서 생성 | `do_image_splitting=False`, size 512 | `min_pixels=max_pixels=448*448` | docs/07 §1 |
| `main()`의 모델 dtype | float32 | bfloat16 | 2B 역전파 메모리 |
| `main()`의 `mean`, `std` | 0.5 / 0.5 | 프로세서의 `image_mean`, `image_std` | CLIP 값 |
| `make_pixel_values()` | `(1, 조각 수, 3, H, W)` | Qwen 조각 배열 `(1024, 1176)` | docs/07 §2 순서 그대로 |
| `losses()`에 넘기는 입력 | `pixel_attention_mask` | `image_grid_thw` (프로세서 출력 그대로) | |
| `eos_token_ids()` | `<end_of_utterance>` 1개 | generation_config의 **2개 모두** | |
| `build_prompt()` | 그대로 | 그대로 (시스템 문장이 자동으로 붙음) | 지우지 않는다 |
| `load_clean_image()`, `_pil_from()`, `_save_png()` | 그대로 | 그대로 | PNG 저장 유지 |

바꾸면 안 되는 곳: 고정 인자값(eps, step, target_len, refresh, w_unc, w_div), 반복 구조와 손실 구성, 확인 주기마다 실제 답 길이를 재는 부분, 저장 형식. 바꾼 항목은 `notebooks/B_qwen2vl/README.md`의 표에 적는다.

B팀 실행 순서: dev 5장으로 go/no-go(docs/07 §6) → 통과하면 `eval_v1` 30장을 밤에 돌리고(사진당 20분 상한) → 저장본 확인 → 해시 → PR. 공격 사진은 B팀 기기와 우리가 띄운 모델에만 넣는다(docs/04 §3-7).
