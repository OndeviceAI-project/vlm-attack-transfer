# B팀 코드 (Qwen2-VL-2B-Instruct)

이 폴더의 코드는 v1 B팀 코드(태그 `v1-midterm`의 `B팀/src/runtime.py`, `attack.py`)의 Qwen 전처리·공격 계산을 그대로 옮기고, 규칙 v2와 두 팀 공통 제작 방식(`이미지생성/README.md`)에 맞게 고친 것이다. A팀 코드(`A팀/code/`)와 같은 구조, 같은 규칙, 같은 기록 형식을 쓴다.

## 1. 파일

| 파일 | 단계 | 하는 일 |
|---|---|---|
| `runtime.py` | 공통 | 모델·프로세서 불러오기, 사진 전처리(448×448 → 조각 배열), 답 생성(README 5.2절), 결정적 실행 설정(5.1절), 실행 환경 기록 |
| `prepare_images.py` | 2 | COCO JPEG를 받아 정상·노이즈 PNG(448×448)를 만들고 `B팀/results/input-manifest.json`에 해시를 남긴다 |
| `attack_qwen.py` | 3 | 공격 사진 한 장을 만든다(`attack_one`) |
| `make_attack_set.py` | 3 | cand를 `order` 순서대로 처리하며 시도마다 판정하고 `B팀/adv/attempts.csv`, `attempts.json`에 남긴다 |

## 2. 실행

```
python B팀/code/prepare_images.py --splits dev            # 코드 확인용
python B팀/code/make_attack_set.py --split dev --limit 2  # 코드 확인용(기록은 B팀/data/dev_attempts/)

python B팀/code/prepare_images.py                         # dev·cand·acc 전부
python B팀/code/make_attack_set.py                        # cand 300장 전부 → B팀/adv/attempts.csv
```

모델은 Hugging Face에서 `settings.MODELS["qwen2vl"]`의 `model_id`, `revision`으로 받는다. 끊겨도 같은 명령으로 이어서 하고, 끝난 후보는 건너뛴다.

## 3. A팀과 통일한 것

| 항목 | 두 팀 공통 |
|---|---|
| 시도 한 번의 값 | `settings`의 `EPS` 8, `STEP` 1, `ATTACK_ITERS` 500, `TARGET_LEN` 128, `REFRESH` 20, `W_UNC` 0.05, `W_DIV` 0 |
| 제작 정밀도·실행 | bf16, sdpa, revision 고정, 결정적 실행 설정(`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `SEED` 2026) |
| 조기 종료 | 종료 토큰 없이 512가 두 번 연속이면 멈춘다 |
| 저장할 사진 | 멈춘 시점(두 번째 연속 512)의 사진. 끝까지 돌면 확인한 시점 중 가장 높은 사진 |
| 숫자가 깨질 때 | 그때까지 고른 사진으로 멈춘다 |
| 시도 | 최대 `ATTACK_MAX_TRIES`(2)번. 1번째는 노이즈 0, 2번째는 무작위 시작 |
| 무작위 시작의 씨앗 | `int(sha256(f"{ATTACK_SEED}:{coco_id 12자리}:{시도 번호}").hexdigest()[:15], 16)`, CPU 난수 |
| 판정 | 저장한 PNG를 다시 읽어 픽셀 차이 ≤ 8이고, `runtime.answer`로 ref에 넣은 답이 종료 없이 512면 `success` |
| 후보 처리 | cand 300장을 `order` 순서대로 끝까지 처리. 원본 답이 512면 `excluded` |
| 기록 | `attempts.csv`의 열과 순서, `attempts.json`(설정·코드 해시·실행 환경·`pixel_gap_max`) |

## 4. 모델 때문에 A팀과 다른 것

| 항목 | SmolVLM (A팀) | Qwen2-VL (B팀) |
|---|---|---|
| 사진 크기 | 512×512 | 448×448 |
| 픽셀 단위 | 0~1 실수 | 0~255 실수(`EPS`, `STEP`도 이 단위) |
| 모델 입력 | 사진 한 장 `(1, 1, 3, 512, 512)` | 조각 배열 `(1024, 1176)`, `image_grid_thw` `[[1, 32, 32]]` |
| 종료 토큰 | 1개 | 2개(`151645`, `151643`) |
| 목표 문장을 넣는 방식 | 글자로 바꾼 뒤 다시 토큰으로 넣음(v1부터) | 토큰 id를 그대로 붙임(v1부터) |
| pixel_values 확인 | 직접 정규화한 값과 비교 | 직접 정규화·조각 배열로 바꾼 값과 비교. float32와 bf16 모두 차이 0이어야 한다 |

## 5. v1 B팀 코드에서 바뀐 것

| 항목 | v1 | 지금 |
|---|---|---|
| 반복 | 최대 300회 | 최대 500회 |
| 시도 | 1번 | 최대 2번(2번째는 무작위 시작) |
| 저장할 사진 | 처음 512에 닿은 사진(같으면 앞의 것) | 멈춘 시점의 사진 |
| 숫자가 깨질 때 | 오류로 멈춤 | 그때까지 고른 사진으로 멈춤 |
| 원본 답이 512 | 공격하지 않고 원본을 저장 | 공격하지 않고 `excluded`로 적음 |
| 모델 경로 | 로컬 스냅샷 경로(`ONDEVICE_AI_ROOT`) | `model_id` + `revision` |
| 사진 목록 | v1 eval 30장 | v2 cand 300장 |

## 6. 주의사항

1. **패키지 버전을 바꾸지 않는다.** transformers 5.19에서 확인했다. 프로세서에 주는 `min_pixels`, `max_pixels`는 transformers 5.23부터 없어진다는 경고가 나온다. 버전을 바꾸면 전처리가 달라질 수 있고, README 5.1절에 따라 바뀐 뒤의 단계를 다시 해야 한다.
2. **결정적 실행 설정을 끄지 않는다.** 이 설정 때문에 느려지지만, 끄면 같은 조건에서 같은 사진이 나온다는 전제가 깨진다.
3. **판정과 4단계는 같은 `runtime.py`를 쓴다.** 4~7단계 코드를 만들 때도 `load_ref()`·`answer()`를 그대로 써야 README 3단계의 '4단계와 같은 코드·환경의 ref'가 지켜진다. 지금 `runtime.py`에는 ref만 있고 rtn·rand·bnb 경우는 아직 없다.
4. **기록 폴더 하나 = 설정 하나.** `settings.py`나 코드가 기록을 만들 때와 달라지면 `make_attack_set.py`가 멈춘다. 그때는 새 폴더(`--out`)를 쓴다.
5. **올리지 않는 것.** `B팀/data/`(COCO JPEG, 정상·노이즈 PNG, 시도마다 만든 공격 PNG). 세트가 정해지면 세트에 든 공격 PNG만 `B팀/adv/`에 올린다(README 3단계).
6. **세트가 main에 병합되기 전에는 압축 모델에 공격 사진을 넣지 않는다**(README 3단계).
7. `W_DIV`(다양성 손실)는 구현하지 않았다. 0이 아니면 멈춘다.
8. 이 코드는 다른 기계에서 옮겨 고친 것이므로, B팀 기계에서 처음 돌릴 때 dev로 먼저 확인한다.
