# 공격 사진 생성

`attack_smolvlm.py`로 만든다(예비 실험 코드 그대로). A팀은 이 파일을 쓰고, B팀은 복사해서 모델 부분만 고친다.

```
python attack_smolvlm.py --images <원본 PNG 폴더> --out <출력 폴더> --iter 300
```

| 인자 | 값 | |
|---|---|---|
| `--images`, `--out` | 입력·출력 폴더 | 자유 |
| `--iter` | 300 (최대 500) | 두 팀 같게 |
| `--eps` | 8 | 바꾸지 않음 |
| `--step` `--target_len` `--refresh` `--w_unc` `--w_div` | 1 / 128 / 20 / 0.05 / 0 (`settings`와 같음) | 두 팀 같게 |
| `--overwrite` | 있는 사진을 다시 만듦 | 고정한 뒤에는 쓰지 않음 |

출력 폴더에 같은 이름의 PNG가 있으면 건너뛰므로, 끊겨도 같은 명령으로 이어서 할 수 있다.

## 지킬 것

1. **공격이 보는 사진 = 평가가 보는 사진.** 코드가 모델에 넣는 pixel_values가 프로세서가 같은 PNG로 만드는 값과 같아야 한다(차이 0). SmolVLM은 프로세서에 `size={"longest_edge": 512}`를 준다.
2. **저장본으로 확인.** PNG로 저장했다가 다시 읽어 원본과의 최대 차이 ≤ 8, 원본 모델에서의 답 길이를 잰다(1번 노트북 뒷부분의 확인 셀이 이 일을 한다). 8을 넘는 파일이 있으면 그 세트는 쓰지 않는다.
3. **한 번 만들면 고정.** 30장이 끝나면 `shasum -a 256 *.png > SHA256SUMS`. 조정은 dev 5장에서만.
4. **실패한 사진도 남긴다.** 답이 안 길어진 사진도 저장하고 로그에 적는다. 성공률은 30장 전체로.

로그는 사진마다 한 줄이면 된다: `image_id, iters_run, minutes, start_len, best_len, saved_len, linf`(saved_len = 저장본을 다시 읽어 잰 길이, linf = 저장본과 원본의 최대 픽셀 차이).

## 지금 코드에서 알아둘 것

- 마지막 걸음의 사진을 저장한다. 답이 가장 길었던 시점의 사진을 저장하도록 바꾸는 것을 권한다(두 팀이 같은 방식으로).
- 사진 한 장 시간 상한과 로그 파일 쓰기는 없다. 필요하면 넣는다.
- 고정값이 파일 안에 상수로 있다. `공통/settings.py`에서 읽게 바꾸면 틀릴 일이 없다.

## B팀이 바꿀 곳

이대로 고치면 전처리 일치 검사를 통과한다(transformers 5.19에서 확인). 자세한 것은 `참고/Qwen2-VL_메모.md`.

| 곳 | SmolVLM | Qwen2-VL-2B |
|---|---|---|
| `MODEL_ID`, `IMG_SIZE` | SmolVLM-256M-Instruct, 512 | Qwen/Qwen2-VL-2B-Instruct, 448 |
| 프로세서 생성 | `do_image_splitting=False, size={"longest_edge": 512}` | `min_pixels=max_pixels=448*448` |
| 모델 dtype | float32 | bfloat16 |
| `mean`, `std` | 0.5 / 0.5 | 프로세서의 `image_mean`, `image_std` |
| `make_pixel_values()` | `(1, 1, 3, 512, 512)` | 조각 배열 `(1024, 1176)` — 순서는 메모 참고 |
| `losses()`에 넘기는 입력 | `pixel_attention_mask` | `image_grid_thw` |
| `eos_token_ids()` | 1개 | generation_config의 2개 모두 |
| `build_prompt()` | 그대로 | 그대로 (시스템 문장이 자동으로 붙음) |
