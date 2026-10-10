# 이미지 생성 — 공격 사진 코드와 규칙

이 폴더의 `attack_smolvlm.py`가 공격 사진을 만드는 **유일한 코드**다(예비 실험 1번 노트북에서 그대로 꺼낸 것). A팀은 이 파일을 그대로 쓰고, B팀은 복사해서 아래 "B팀이 바꿀 곳"만 고친다. 두 팀이 다른 구현을 쓰면 결과 차이가 모델 때문인지 코드 때문인지 알 수 없다.

## 1. 입력과 출력

- 입력: `<팀>/data/clean/` 중 `공통규칙/eval_v1.txt`에 있는 30장. dev 5장은 설정 확인용으로만.
- 만드는 모델: **원본**(양자화 없음). SmolVLM은 fp32, Qwen2-VL은 bf16. 2차의 적응 공격만 예외로, 양자화한 모델로 만들고 폴더 이름을 `adv_on_<경우>`로 둔다.
- 출력: `<팀>/data/adv/<id>.png`(무손실 PNG), `adv/attack_log.csv`, `adv/SHA256SUMS`.

## 2. 실행

```
python attack_smolvlm.py --images <원본 PNG 폴더> --out <출력 폴더> --iter 300 2>&1 | tee <출력 폴더>/attack_stdout.txt
```

| 인자 | 뜻 | v1 값 | 바꿔도 되나 |
|---|---|---|---|
| `--images` | 원본 사진 폴더 | `eval_v1.txt`의 30장만 모은 폴더 | 경로만 |
| `--out` | 공격 사진 저장 폴더 | `<팀>/data/adv` | 경로만 |
| `--iter` | 반복 수 | **300** (파일 기본값 200은 예비 실험 값. 상한 500) | 300으로 고정해 실행 |
| `--eps` | 노이즈 상한 (0~255 눈금) | 8 | 금지 |
| `--step` | 걸음 크기 | 1 | 금지 |
| `--target_len` | 목표 답 길이 | 128 | 금지 |
| `--refresh` | 확인 주기 | 20 | 금지 |
| `--w_unc` | 손실 가중치 | 0.05 | 금지 |
| `--w_div` | 손실 가중치 | 0 | 금지 |
| `--overwrite` | 있는 공격 사진을 다시 만듦 | 쓰지 않음 | 세트 고정 뒤 금지 |

알아둘 것:

- 출력 폴더에 같은 이름의 PNG가 있으면 건너뛴다. 끊겨도 같은 명령을 다시 실행하면 이어서 한다.
- 진행 상황은 화면에만 찍힌다. `attack_log.csv`(아래 TODO 5)가 생기기 전까지는 반드시 `tee`로 남긴다.
- 장치는 cuda → mps → cpu 순으로 자동 선택. 공격 생성은 GPU 기계에서 하고, 결과 PNG만 저장소로 옮기면 평가는 어디서든 된다.
- 입력 사진은 안에서 512로 다시 맞추지만, 1단계에서 만든 clean PNG는 이미 512라 바뀌는 것이 없다.
- `--images`에 48장을 다 넣지 않도록 `eval_v1.txt`의 30장만 링크·복사한 폴더를 따로 만든다.

## 3. 반드시 지킬 것

1. **공격 경로 = 평가 경로.** 이 코드가 모델에 넣는 pixel_values는 프로세서가 같은 PNG로 만드는 값과 숫자까지 같아야 한다(0단계 '전처리 일치' 차이 0). 검사 방법: 같은 사진에 대해 `make_pixel_values(x, 0, mean, std, 1)`와 `processor(text=..., images=[사진])["pixel_values"]`의 차이 최댓값을 출력한다.
2. **저장본으로 다시 확인한다.** PNG로 저장했다가 다시 읽은 사진으로 (a) 원본과의 최대 차이 ≤ 8, (b) 원본 모델에서의 답 길이를 잰다(1번 노트북 뒷부분의 확인 셀이 이 일을 한다). 8을 넘는 파일이 하나라도 있으면 세트 전체를 쓰지 않는다.
3. **"가장 길었던 노이즈"를 저장한다.** 지금 코드는 마지막 걸음을 저장한다. 확인 주기마다 잰 답 길이가 가장 길었던 시점의 사진을 저장하도록 A팀이 고친 뒤 B팀이 복사한다(TODO 3).
4. **한 번 만들면 고정.** 30장이 끝나면 `shasum -a 256 *.png > SHA256SUMS`, PNG·로그·해시를 한 PR로. 이후 바꾸지 않는다. 다시 만들어야 하면 `adv_v2`처럼 새 폴더로.
5. **eval 사진으로 설정을 만지지 않는다.** 조정은 dev 5장에서만.
6. **실패한 사진도 남긴다.** 상한 안에 답이 안 길어진 사진도 저장하고 `best_len`을 쓴다. 성공률은 30장 전체로 계산한다.
7. **우리 기기에서만.** 외부 서비스·앱·다른 사람 기기에 넣지 않는다.

## 4. 공식 실행 전에 맞출 것 (A팀 TODO, 순서대로)

1. **프로세서 설정.** 프로세서를 만들 때 `do_image_splitting=False`에 더해 `size={"longest_edge": 512}`를 준다. 기본값(2048)이면 512 사진을 키웠다 줄여 픽셀이 평균 1.3/255·최대 24/255 달라진다. 평가 노트북도 같게. 바꾼 뒤 전처리 일치 차이 0 확인.
2. **고정값 출처.** `MODEL_ID`, `QUESTION`, `IMG_SIZE`, 인자 기본값을 `공통규칙/settings.py`에서 읽게 한다(`sys.path`에 `공통규칙` 폴더 추가 후 `import settings`). 값은 그대로.
3. **저장 시점.** 위 3절 3번.
4. **시간 상한.** 사진 한 장이 `settings.ATTACK_MAX_MINUTES`를 넘으면 멈추고 그때까지의 최선을 저장, 사유를 로그에.
5. **로그.** `attack_log.csv`에 사진마다 한 줄: `image_id, attack_case, attack_dtype, settings_version, eps, iters_limit, iters_run, minutes, start_len, best_len, saved_len, linf, stop_reason, history, device, time`. `start_len` 공격 전 답 길이, `best_len` 반복 중 최고, `saved_len` 저장본을 다시 읽어 잰 길이, `linf` 저장본과 원본의 최대 픽셀 차이, `history` "20:45 40:512 …"(걸음:길이), `stop_reason` 조기 종료 / 반복 수 다 씀 / 시간 상한 / nan.
6. **저장본 확인** 셀을 그대로 쓰되 결과를 로그에 합친다.
7. **고정.** 해시 → PR.

## 5. 성립 기준

원본 모델에서 저장본의 답이 상한에 닿은 사진이 30장 중 **24장(80%) 이상**이어야 다음 단계로 간다. 미달이면 dev 사진으로 반복 수(≤500)만 조정해 보고, 그래도 안 되면 그 성공률 그대로 보고하고 전이율은 성공한 사진에 대해서만 계산한다.

## 6. B팀이 바꿀 곳

바꾼 뒤 0단계 전처리 일치 검사를 통과해야 한다. 바꾼 항목은 `B팀/README.md`의 표에 적는다.

| 바꿀 곳 | 지금 (SmolVLM) | Qwen2-VL-2B |
|---|---|---|
| `MODEL_ID` | SmolVLM-256M-Instruct | Qwen/Qwen2-VL-2B-Instruct |
| `IMG_SIZE` | 512 | 448 (28의 배수) |
| `main()`의 프로세서 생성 | `do_image_splitting=False`, size 512 | `min_pixels=max_pixels=448*448` |
| `main()`의 모델 dtype | float32 | bfloat16 (2B 역전파 메모리) |
| `main()`의 `mean`, `std` | 0.5 / 0.5 | 프로세서의 `image_mean`, `image_std` |
| `make_pixel_values()` | `(1, 조각 수, 3, H, W)` | Qwen 조각 배열 `(1024, 1176)` — B팀/README 2절 순서 그대로 |
| `losses()`에 넘기는 입력 | `pixel_attention_mask` | `image_grid_thw` (프로세서 출력 그대로) |
| `eos_token_ids()` | `<end_of_utterance>` 1개 | generation_config의 **2개 모두** |
| `build_prompt()` | 그대로 | 그대로 (시스템 문장이 자동으로 붙음. 지우지 않는다) |
| `load_clean_image()`, `_pil_from()`, `_save_png()` | 그대로 | 그대로 |

바꾸면 안 되는 곳: 고정 인자값, 반복 구조와 손실 구성, 확인 주기마다 실제 답 길이를 재는 부분, PNG 저장.

B팀 순서: dev 5장으로 go/no-go(B팀/README 4절) → 통과하면 `eval_v1` 30장을 밤에(사진당 20분 상한) → 저장본 확인 → 해시 → PR.
