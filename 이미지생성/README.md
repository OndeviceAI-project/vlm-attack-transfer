# 이미지 생성 — 공격 사진 코드와 규칙

`attack_smolvlm.py`가 공격 사진을 만드는 코드다(예비 실험 1번 노트북에서 그대로 꺼낸 것). A팀은 이 파일을 쓰고, B팀은 복사해서 모델에 맞게 고친다. 핵심은 **두 팀이 같은 노이즈 상한, 같은 예산, 같은 목표(종료 토큰 억제)로 만든다**는 것이고, 그 밖의 구현은 자유다.

## 실행

```
python attack_smolvlm.py --images <원본 PNG 폴더> --out <출력 폴더> --iter 300
```

| 인자 | 뜻 | 값 | |
|---|---|---|---|
| `--images`, `--out` | 입력·출력 폴더 | `eval_v1`의 30장 / `<팀>/data/adv` | 자유 |
| `--iter` | 반복 수 | **300** (상한 500) | 두 팀 같게 |
| `--eps` | 노이즈 상한 | 8 | 바꾸지 않음 |
| `--step`, `--target_len`, `--refresh`, `--w_unc`, `--w_div` | 공격 설정 | 1 / 128 / 20 / 0.05 / 0 | 두 팀 같게 |
| `--overwrite` | 있는 사진을 다시 만듦 | | 고정 뒤에는 쓰지 않음 |

출력 폴더에 같은 이름의 PNG가 있으면 건너뛰므로, 끊겨도 같은 명령으로 이어서 할 수 있다. 진행 상황은 화면에 찍히니 파일로 같이 남겨 두면 좋다.

## 지킬 것 네 가지

1. **공격이 보는 사진 = 평가가 보는 사진.** 코드가 모델에 넣는 pixel_values가 프로세서가 같은 PNG로 만드는 값과 같아야 한다(차이 0). 다르면 공격은 평가와 다른 사진을 보고 만든 셈이다.
2. **저장본으로 확인.** PNG로 저장했다가 다시 읽어 (a) 원본과의 최대 차이 ≤ 8, (b) 원본 모델에서의 답 길이를 잰다. 8을 넘는 파일이 있으면 세트를 쓰지 않는다.
3. **한 번 만들면 고정.** 30장이 끝나면 `shasum -a 256 *.png > SHA256SUMS`, PNG·로그·해시를 한 PR로. 조정은 dev 5장에서만.
4. **실패한 사진도 남긴다.** 답이 안 길어진 사진도 저장하고 로그에 적는다. 성공률은 30장 전체로.

## 로그 (`attack_log.csv`, 사진 한 장 = 한 줄)

`image_id, attack_case, attack_dtype, settings_version, eps, iters_limit, iters_run, minutes, start_len, best_len, saved_len, linf, stop_reason, history, device, time`

start_len 공격 전 답 길이 · best_len 반복 중 최고 · saved_len 저장본을 다시 읽어 잰 길이 · linf 저장본과 원본의 최대 픽셀 차이 · history "20:45 40:512 …"(걸음:길이) · stop_reason 조기 종료 / 반복 수 다 씀 / 시간 상한.

## 성립 기준

원본 모델에서 저장본의 답이 512에 닿은 사진이 30장 중 24장(80%) 이상이면 다음 단계로. 미달이면 dev 사진으로 반복 수(≤500)만 조정하고, 그래도 안 되면 그 성공률 그대로 보고한다.

## 지금 코드에서 손볼 것 (A팀, 공식 실행 전)

- 프로세서를 만들 때 `size={"longest_edge": 512}`를 준다. 기본값(2048)이면 512 사진을 키웠다 줄여 픽셀이 평균 1.3/255·최대 24/255 바뀐다. 평가 쪽도 같게.
- 마지막 걸음이 아니라 **답이 가장 길었던 시점**의 사진을 저장하게 한다.
- 사진 한 장 시간 상한(`settings.ATTACK_MAX_MINUTES`)과 `attack_log.csv` 쓰기를 넣는다.
- 고정값(`MODEL_ID`, `QUESTION`, `IMG_SIZE`, 기본 인자)을 `공통규칙/settings.py`에서 읽는다.

B팀은 이 수정이 끝난 파일을 복사한다.

## B팀 참고 — Qwen2-VL에서 달라지는 곳

미리 확인해 둔 것이다. 이대로 고치면 전처리 일치 검사를 통과한다. 다른 방법으로 해도 검사만 통과하면 된다.

| 곳 | SmolVLM | Qwen2-VL-2B |
|---|---|---|
| `MODEL_ID`, `IMG_SIZE` | SmolVLM-256M-Instruct, 512 | Qwen/Qwen2-VL-2B-Instruct, 448 |
| 프로세서 생성 | `do_image_splitting=False, size={"longest_edge": 512}` | `min_pixels=max_pixels=448*448` |
| 모델 dtype | float32 | bfloat16 |
| `mean`, `std` | 0.5 / 0.5 | 프로세서의 `image_mean`, `image_std` |
| `make_pixel_values()` | `(1, 1, 3, 512, 512)` | 조각 배열 `(1024, 1176)` — 순서는 `B팀/README.md` 참고 |
| `losses()`에 넘기는 입력 | `pixel_attention_mask` | `image_grid_thw` |
| `eos_token_ids()` | 1개 | generation_config의 2개 모두 |
| `build_prompt()` | 그대로 | 그대로 (시스템 문장이 자동으로 붙음) |
