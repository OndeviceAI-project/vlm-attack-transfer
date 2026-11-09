# 원본 모델에서 성공한 Adversarial Attack, 압축 모델에도 전이되는가

온디바이스 AI 7조 · 규칙 v2 (`공통/settings.py`의 `SETTINGS_VERSION = "v2"`)

v2에서는 공격 사진부터 모든 실험을 새로 한다. v1(중간발표까지의 실험, 결과, 자료)은 저장소에서 내렸고, 태그 `v1-midterm`에 그대로 남아 있다.

## 1. 실험 목표

사진에 사람 눈에 보이지 않는 노이즈(픽셀마다 최대 8/255)를 넣어, VLM이 답을 끝내지 못하게 만드는 공격 사진이 있다. 공격 사진은 압축하지 않은 원본(bf16) 모델로 만든다.

이 저장소의 실험은 두 가지를 잰다.

- 원본 모델에서 성공한 공격 사진을 비전 인코더만 압축한 모델, 언어모델만 압축한 모델에 넣었을 때 나오는 답
- 같은 압축 모델에 공격하지 않은 사진을 넣었을 때 나오는 답

| 팀 | 모델 |
|---|---|
| A | SmolVLM-256M-Instruct |
| B | Qwen2-VL-2B-Instruct |

두 팀은 같은 COCO 사진 30장으로 공격 사진을 만들고, 같은 경우, 같은 입력, 같은 형식으로 잰다.

분석 지표(전이율, 정확도 등)는 실험이 모두 끝난 뒤 8단계에서 정한다. 1~7단계에서는 이 문서가 정한 원자료만 남긴다.

## 2. 저장소에 올리는 것과 올리지 않는 것

이 저장소에는 객관적 사실만 둔다. 누가 분석하든 남의 해석이 섞이지 않은 같은 원자료에서 시작하기 위해서다.

올리는 것:

- 실험을 실행하는 코드, 설정, 사진 목록
- 코드가 쓴 그대로의 결과 파일(CSV, JSON)과 실행 환경 정보
- 공격 사진 세트(PNG), 해시 목록, 시도 기록
- 실험 설계와 규칙(이 문서, `이미지생성/README.md`)

올리지 않는 것:

- 해석, 결론, 평가가 들어간 글. 보고서, 요약, 비교표, 그래프, 결과를 설명하는 문장
- 작업 메모, 진행 상황 기록, 대화 기록
- 손으로 고친 결과 파일
- 모델 가중치, COCO 원본 JPEG, 정상·노이즈 PNG. 사진은 코드로 똑같이 다시 만들 수 있으므로 해시만 올린다
- 절대 경로, 서버·계정 이름, 개인 정보

## 3. 진행 순서

날짜 대신 순서로 정한다. 각 단계는 '끝난 조건'의 파일이 main에 병합되면 끝난 것으로 본다.

| 단계 | 할 일 | 누가 | 먼저 끝나야 하는 단계 | 끝난 조건 |
|---|---|---|---|---|
| 0 | v2 규칙 | 공통 | — | 이 문서가 main에 병합 |
| 1-1 | 공격 사진 만드는 방법 합의 | A·B 함께 | 0 | `이미지생성/README.md` 작성, `settings.ATTACK_MAX_TRIES`에 정수 기입 |
| 1-2 | 실험 코드 준비와 점검 | 팀별 | 0 | `<팀>/code/`, `<팀>/results/check.json` |
| 2 | 사진 준비 | 팀별 | 1-2 | `<팀>/results/input-manifest.json` |
| 3 | 공격 사진 세트 확정 | A·B 함께 | 1-1, 2 | `공통/attack_set.csv`, `<팀>/adv/` |
| 4 | 비전 인코더 실험 | 팀별 | 3 | `<팀>/results/vision.csv`, `.json` |
| 5 | 언어모델 실험 | 팀별 | 3 | `<팀>/results/language.csv`, `.json` |
| 6 | 모델 내부 측정 | 팀별 | 3 | `<팀>/results/internal.csv`, `.json` |
| 7 | 정확도 세트 실행 | 팀별 | 2 | `<팀>/results/accuracy.csv`, `.json` |
| 8 | 분석 지표 정의 | A·B 함께 | 4~7 | 지표 정의를 이 문서에 추가 |

4·5·6단계는 3단계가 끝나면 순서와 상관없이 한다. 7단계는 공격 사진이 필요 없어서 2단계가 끝나면 언제든 한다.

### 1-1단계. 공격 사진 만드는 방법 합의

두 팀이 함께 정해 `이미지생성/README.md`에 적는다. 이 파일이 생기기 전에는 공격 사진을 만들지 않는다. 반드시 적을 것은 다음과 같다.

- 모델별 제작 코드의 위치와 실행 방법
- 시도 한 번의 설정. `settings.py`의 공격 사진 값을 쓴다
- 사진 한 장당 최대 시도 횟수. `settings.ATTACK_MAX_TRIES`에도 적는다
- 시도마다 무엇이 달라지는지와 그 값을 정하는 방법. 그 값은 `attempts.csv`에 열로 남긴다
- 제작 정밀도. 두 모델 모두 bf16이다(`settings.MODELS[모델]["attack_dtype"]`)
- 제작 코드가 모델에 넣는 `pixel_values`가, 평가 코드가 같은 PNG로 만드는 값과 같다는 확인 방법과 결과(차이 0)

지금 `이미지생성/attack_smolvlm.py`는 v1에서 쓴 SmolVLM용 제작 코드다. v2에서 이 코드를 그대로 쓸지 고쳐 쓸지도 1-1단계에서 정한다.

### 1-2단계. 실험 코드 준비와 점검

각 팀이 4~7단계에 쓸 코드를 `<팀>/code/`에 올리고, dev 사진 5장으로 아래를 점검해 `<팀>/results/check.json`에 남긴다. 하나라도 통과하지 못하면 2단계로 가지 않는다.

1. 모델: `model_id`, `revision`이 settings 값과 같다.
2. 종료 토큰: 모델 generation_config의 `eos_token_id` 목록이 `settings.MODELS[모델]["eos_token_ids"]`와 같다.
3. 전처리: dev 5장 각각에서, 프로세서가 만든 `pixel_values`가 PNG를 직접 정규화한 값과 같다(차이 최댓값 1e-6 이하). 프로세서가 사진 크기를 다시 바꾸지 않는다는 확인이다. Qwen2-VL은 직접 계산한 값을 프로세서와 같은 조각 배열 모양으로 바꿔 비교한다.
4. census: `settings.CASES_ALL`의 25경우 모두, 대상 부품의 Linear는 전부 바뀌고 나머지 부품은 하나도 바뀌지 않는다(`quant_rtn.check_census`). 부품별 Linear 수가 `settings.MODELS[모델]["linear_count"]`와 같다.
5. 가중치 변화: rtn 경우의 값이 6장 '양자화 확인값'과 0.01%p 안에서 같다. rand 경우는 같은 부품·비트의 rtn과 0.01%p 안에서 같다. bnb 경우는 값을 재서 기록만 한다.
6. 재현성: ref에 dev 정상 사진 5장을 두 번씩 넣어 두 답이 토큰까지 같다.
7. 생성: 25경우마다 dev 정상 사진 1장의 답을 생성해 기록한다. 답이 상한에 닿아도 오류가 아니라 기록할 사실이다.

### 2단계. 사진 준비

`공통/image_list.csv`의 dev 5장, cand 150장, acc 300장 전부의 정상·노이즈 PNG를 5.3절대로 만든다. 만든 PNG와 내려받은 COCO JPEG의 SHA-256을 `<팀>/results/input-manifest.json`에 적는다. PNG와 JPEG는 올리지 않는다.

### 3단계. 공격 사진 세트 확정

두 팀이 같은 사진 30장(`settings.N_ATTACK_SET`)으로 된 공격 사진 세트를 만든다.

1. **후보 순서.** `image_list.csv`의 cand를 `order` 순서대로 처리한다. 건너뛰지 않는다. 두 팀의 순서는 같다.
2. **후보 한 장의 처리.** 각 팀이 자기 모델로 한다.
   - 원본(ref)에 정상 사진을 넣은 답이 종료 토큰 없이 512에 닿으면(5.2절 `hit_cap`) 그 후보는 `excluded`로 적고 공격 사진을 만들지 않는다.
   - 1-1단계에서 정한 방법으로 공격 사진을 만든다. 공격 사진 한 장을 만드는 것이 시도 한 번이다.
   - 시도마다 저장한 PNG를 다시 읽어 판정한다. 정상 PNG와의 픽셀 차이 최댓값이 `EPS`(8) 이하이고, 4단계와 같은 코드·환경의 ref에 넣은 답이 종료 토큰 없이 512에 닿으면 `success`, 아니면 `fail`이다. 제작 코드가 출력하는 성공 여부는 판정에 쓰지 않는다.
   - `success`가 나오면 그 후보는 끝난다. `fail`이면 `ATTACK_MAX_TRIES`까지 다시 시도한다.
3. **세트 정하기.** 두 팀 모두 `success`가 나온 후보를 `order` 순서대로 앞에서 30장 고른다. 30번째 공통 성공 후보까지는 두 팀 모두 빠짐없이 처리한다.
4. **후보가 모자랄 때.** cand 150장으로 30장이 채워지지 않으면, 먼저 cand를 같은 규칙(COCO val2017 ID 오름차순으로 다음 사진)으로 늘리는 PR을 올린 뒤 이어서 처리한다. 정렬 순위 501번째부터는 acc를 뽑는 범위라 cand에 넣지 않는다.
5. **올리기와 고정.** 세트는 `공통/attack_set.csv`에 적는다. 각 팀은 세트 30장의 공격 PNG를 `<팀>/adv/<coco_id 12자리>.png`로 올리고, `<팀>/adv/SHA256SUMS`(`shasum -a 256 *.png`)를 붙인다. 확정한 뒤에는 바꾸지 않는다. 다시 만들어야 하면 settings 버전을 올리고 3단계를 처음부터 한다.
6. **순서 지키기.** 세트가 main에 병합되기 전에는 압축 모델(ref가 아닌 경우)에 공격 사진을 넣지 않는다. 압축 모델에서 나온 결과로 세트의 사진을 고르거나 바꾸지 않는다.
7. **시도 기록.** 처리한 모든 후보와 모든 시도를 `<팀>/adv/attempts.csv`에 한 줄씩 남긴다(형식은 7장). 세트에 들지 않은 시도의 PNG는 올리지 않는다.

### 4단계. 비전 인코더 실험

공격 사진 세트 30장의 정상·노이즈·공격 사진을 원본 모델과 비전 인코더만 압축한 모델에 넣고 답을 기록한다.

- 경우: `settings.CASES_VISION` 13경우

  | 방식 | 경우 | 뜻 |
  |---|---|---|
  | 원본 | `ref` | bf16 원본 |
  | rtn | `rtn-V-8`, `rtn-V-6`, `rtn-V-5`, `rtn-V-4`, `rtn-V-3` | 비전 인코더 Linear weight를 그룹 64 가짜 양자화 |
  | bnb | `bnb-V-8`, `bnb-V-4` | bitsandbytes 8비트(LLM.int8), 4비트(nf4) |
  | rand | `rand-V-8`, `rand-V-6`, `rand-V-5`, `rand-V-4`, `rand-V-3` | 대조군. 같은 비트 rtn이 가중치마다 바꾸는 크기만큼, 방향(+/−)만 무작위로 바꿈 |

- 입력: `공통/attack_set.csv`의 30장 × `clean`, `noise`, `adv` = 90개. 모든 경우에 같은 90개를 넣는다.
- 경우 만들기와 확인: 5.4절. 경우마다 census와 가중치 변화를 JSON에 남긴다.
- 출력: `<팀>/results/vision.csv`(13 × 90 = 1,170줄)와 `vision.json`. 형식은 7장.
- 정상 사진이나 노이즈 사진의 답이 상한에 닿은 줄도 지우지 않는다.

### 5단계. 언어모델 실험

4단계와 같고, 압축하는 부품만 언어모델이다.

- 경우: `settings.CASES_LANGUAGE` 13경우(`ref`, `rtn-L-8/6/5/4/3`, `bnb-L-8/4`, `rand-L-8/6/5/4/3`)
- 입력: 4단계와 같은 90개
- 출력: `<팀>/results/language.csv`(1,170줄)와 `language.json`

### 6단계. 모델 내부 측정

답을 생성하지 않고 모델 안의 값을 잰다.

- 경우: `settings.CASES_ALL` 25경우(4·5단계 경우 전부, `ref`는 한 번)
- 입력: 공격 사진 세트 30장의 `clean`, `adv`
- 그림 정보 f(x): 사진 x와 질문을 넣고 한 번 계산(forward)할 때 연결부(`parts["C"]`) 모듈이 내는 출력 전체를 1차원으로 편 float32 벡터
- ref에서 먼저 준비하는 것(사진마다):
  - a: ref가 공격 사진에 생성한 답의 앞 64토큰(5.2절과 같은 생성, 종료 토큰 없음)
  - c: ref가 정상 사진에 생성한 답에서 마지막 종료 토큰을 뺀 토큰들(마지막 토큰이 종료 토큰이 아니면 멈춘다)
  - a와 c의 토큰 id는 `internal.json`에 남긴다.

경우 k의 모델에서 사진마다 아래 여섯 값을 잰다. ‖·‖은 L2 크기, cos는 코사인 유사도다.

| 열 | 정의 |
|---|---|
| `feat_change_clean` | 100 × ‖f_k(clean) − f_ref(clean)‖ ÷ ‖f_ref(clean)‖ |
| `attack_shift` | 100 × ‖f_k(adv) − f_k(clean)‖ ÷ ‖f_k(clean)‖ |
| `attack_dir_kept` | cos(f_k(adv) − f_k(clean), f_ref(adv) − f_ref(clean)) |
| `shake_ratio` | ‖f_k(adv) − f_ref(adv)‖ ÷ ‖f_ref(adv) − f_ref(clean)‖ |
| `eos_gap_mean`, `eos_gap_min` | 공격 사진과 질문 뒤에 a를 붙여 한 번 계산한다. a의 t번째 토큰(t = 1~64)을 고르는 자리마다 (그 자리 로짓의 최댓값) − (종료 토큰 로짓 중 최댓값)을 구하고, 64개의 평균과 최솟값을 적는다 |
| `clean_end_margin` | 정상 사진과 질문 뒤에 c를 붙여 한 번 계산한다. c 다음 토큰을 고르는 자리에서 (종료 토큰 로짓 중 최댓값) − (종료 토큰이 아닌 토큰 로짓 중 최댓값) |

- 로짓은 float32로 바꿔 계산한다. 종료 토큰은 `settings.MODELS[모델]["eos_token_ids"]`다.
- ref 줄과 5단계 경우(L)는 그림 정보가 ref와 같으므로 `feat_change_clean` 0, `attack_dir_kept` 1, `shake_ratio` 0이 나와야 한다. 다르면 원인을 확인하고 JSON에 적는다.
- 출력: `<팀>/results/internal.csv`(25 × 30 = 750줄)와 `internal.json`

### 7단계. 정확도 세트 실행

공격 사진 없이, 압축이 평소 답에 주는 변화를 재기 위한 원자료를 만든다.

- 경우: `settings.CASES_ALL` 25경우
- 입력: `image_list.csv`의 acc 300장 × `clean`, `noise` = 600개
- 출력: `<팀>/results/accuracy.csv`(25 × 600 = 15,000줄)와 `accuracy.json`. 형식은 4단계와 같다.
- 답을 채점하는 방법은 8단계에서 정한다.

### 8단계. 분석 지표 정의

4~7단계가 모두 끝나면 두 팀이 분석 지표의 정의를 이 문서에 추가한다. 계산한 지표 값, 그래프, 해석은 2장에 따라 저장소에 올리지 않는다.

## 4. 사진 목록

`공통/image_list.csv`. 모든 사진은 COCO val2017이다.

| split | 장수 | 쓰는 곳 | 고른 방법 |
|---|---|---|---|
| `pilot` | 3 | 쓰지 않음(예비 실험에 썼음) | — |
| `dev` | 5 | 1-2단계 점검, 공격 사진 설정 조정 | val2017 ID 오름차순 3~7번째 |
| `cand` | 150 | 3단계 후보, `order` 순서대로 | val2017 ID 오름차순 8~157번째 |
| `acc` | 300 | 7단계 | val2017 ID 오름차순 501~5000번째 사진 중 `numpy.random.default_rng(2026).choice(..., 300, replace=False)`로 뽑아 ID 오름차순 정렬 |

dev와 pilot은 결과 CSV에 넣지 않는다. cand의 1~43번째는 v1의 eval 30장과 extra 13장과 같은 사진이다.

## 5. 공통 조건

모든 단계에 적용한다.

### 5.1 모델과 실행 환경

- `from_pretrained`에 `settings.MODELS[모델]`의 `model_id`, `revision`을 준다. dtype은 `settings.DTYPE`(bf16), `attn_implementation`은 `settings.ATTN_IMPLEMENTATION`(sdpa)이다.
- 한 모델의 2~7단계(3단계의 성공 판정 포함)는 모두 같은 기계, 같은 패키지 버전에서 한다. bnb 경우가 있으므로 bf16을 지원하는 CUDA GPU를 쓴다. 기계나 패키지 버전이 바뀌면 바뀐 뒤 단계는 다시 한다.
- `torch.manual_seed(settings.SEED)`. CUDA에서는 `torch.use_deterministic_algorithms(True)`, 환경 변수 `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.benchmark = False`로 실행한다. 이 설정으로 돌지 않는 연산이 있으면 그 사실을 JSON에 적고, 1-2단계의 재현성 점검을 통과한 설정을 쓴다.
- 입력은 한 번에 하나씩 넣는다(배치 없음).

### 5.2 질문과 답 생성

- 메시지는 사용자 메시지 하나이고, 내용은 사진 하나와 글 `settings.PROMPT`다. `processor.apply_chat_template(messages, add_generation_prompt=True)`로 만든다. Qwen2-VL은 시스템 문장이 자동으로 붙고, 그대로 둔다.
- `model.generate(**inputs, **settings.GEN_KWARGS, eos_token_id=settings.MODELS[모델]["eos_token_ids"])`
- 답은 generate가 새로 만든 토큰 전부다.
  - `n_tokens`: 새로 만든 토큰 수. 종료 토큰으로 끝났으면 종료 토큰도 센다.
  - `ended_eos`: 마지막 토큰이 종료 토큰 목록에 있다.
  - `hit_cap`: `n_tokens == 512`이고 `ended_eos`가 아니다.
  - `text`: `tokenizer.decode(새 토큰, skip_special_tokens=True)`의 결과를 그대로 적는다.
  - `latency_s`: generate 호출 바로 전과 바로 뒤의 `time.perf_counter()` 차이. 재기 직전과 직후에 `torch.cuda.synchronize()`를 부른다.

### 5.3 사진

- 프로세서: `AutoProcessor.from_pretrained(model_id, revision=revision, **processor_kwargs)`
- 정상 사진(`clean`): COCO JPEG을 `Image.open(...).convert("RGB").resize((S, S), Image.BICUBIC)`로 바로 리사이즈한다(자르지 않음). S는 `img_size`(SmolVLM 512, Qwen2-VL 448)다. PNG로 저장한다.
- 노이즈 사진(`noise`): 정상 사진 배열 `arr`(int16)에 `arr + EPS × sign`을 하고 0~255로 자른 뒤 uint8 PNG로 저장한다. `sign = numpy.random.default_rng([settings.NOISE_SEED, coco_id]).choice(numpy.array([-1, 1], dtype=numpy.int16), size=arr.shape)`
- 공격 사진(`adv`): 3단계에서 확정한 `<팀>/adv/<coco_id 12자리>.png`
- 사진은 PNG만 쓴다. JPEG로 저장하면 노이즈가 지워진다.
- 같은 PNG라면 공격 사진 제작 코드, 4~7단계 코드, 내부 측정 코드가 모델에 넣는 `pixel_values`가 모두 같아야 한다(차이 0).

### 5.4 경우 만들기

경우 이름은 `<방식>-<부품>-<비트>`다. 부품 이름은 `settings.MODELS[모델]["parts"]`의 V(비전 인코더), L(언어모델)이다. 연결부(C)와 출력층(H)은 어느 경우에도 바꾸지 않는다. 모든 방식은 Linear 층의 weight만 바꾸고, bias·정규화·Conv·임베딩은 그대로 둔다.

- `ref`: bf16 원본을 그대로 올린다.
- `rtn`: bf16 원본을 올린 뒤 `quant_rtn.apply_rtn(model, parts, targets=부품, bits=비트, group=settings.QUANT_GROUP)`
- `rand`: bf16 원본을 올린 뒤 `quant_rtn.apply_rand(model, parts, targets=부품, bits=비트, group=settings.QUANT_GROUP, seed=settings.RAND_SEED)`
- `bnb`: `from_pretrained`에 `quantization_config`를 준다. `llm_int8_skip_modules`는 parts 중 대상이 아닌 세 부품의 이름이다(예: `bnb-V-4`면 C, L, H의 이름).
  - 8비트: `BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)`
  - 4비트: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip)`
- 경우마다 모델을 새로 올린다. 한 모델에 방식을 두 번 적용하지 않는다.
- 경우마다 확인한다: `quant_rtn.check_census(quant_rtn.census(model, parts), targets, settings.PART_NAMES)`(ref는 `targets=""`). 부품별 Linear 수는 `linear_count`와 같아야 한다.
- 가중치 변화(%) = 100 × √(Σ‖W_경우 − W_원본‖² ÷ Σ‖W_원본‖²). 합은 대상 부품의 Linear weight 전부에 대해 한다. rtn과 rand는 함수가 돌려주는 값을 쓴다. bnb는 가중치를 실수로 되돌려 같은 식으로 계산한다. 8비트는 `int8 weight × SCB ÷ 127`(행마다), 4비트는 `bitsandbytes.functional.dequantize_4bit(weight, weight.quant_state)`로 되돌린다.

## 6. 양자화 확인값

rtn 경우의 가중치 변화(%)다. 1-2단계 점검에서 이 값과 0.01%p 안에서 같아야 한다. rand는 같은 부품·비트의 rtn과 0.01%p 안에서 같아야 한다.

| 모델 | 부품 | 8비트 | 6비트 | 5비트 | 4비트 | 3비트 |
|---|---|---|---|---|---|---|
| SmolVLM | V | 0.57 | 2.24 | 4.54 | 9.38 | 20.10 |
| SmolVLM | L | 0.57 | 2.21 | 4.48 | 9.26 | 19.86 |
| Qwen2-VL | V | 0.57 | 2.21 | 4.49 | 9.27 | 19.88 |
| Qwen2-VL | L | 0.57 | 2.21 | 4.49 | 9.27 | 19.87 |

부품별 Linear 수: SmolVLM V 72, C 1, L 210, H 1 / Qwen2-VL V 128, C 2, L 196, H 1

## 7. 결과 형식

### 파일

| 단계 | 파일 |
|---|---|
| 1-2 | `<팀>/results/check.json` |
| 2 | `<팀>/results/input-manifest.json` |
| 3 | `공통/attack_set.csv`, `<팀>/adv/<coco_id 12자리>.png` 30장, `<팀>/adv/SHA256SUMS`, `<팀>/adv/attempts.csv` |
| 4 | `<팀>/results/vision.csv`, `vision.json` |
| 5 | `<팀>/results/language.csv`, `language.json` |
| 6 | `<팀>/results/internal.csv`, `internal.json` |
| 7 | `<팀>/results/accuracy.csv`, `accuracy.json` |

### 답 CSV (4·5·7단계)

한 답이 한 줄이다. 줄 순서는 경우(settings 목록 순서) → 사진(attack_set의 `set_order` 또는 acc의 `order`) → 입력(`clean`, `noise`, `adv`) 순이다.

| 열 | 뜻 | 예 |
|---|---|---|
| `model` | `smolvlm` / `qwen2vl` | `qwen2vl` |
| `case` | 경우 이름 | `rtn-V-4` |
| `image_id` | COCO id 12자리 문자열 | `000000001425` |
| `input` | `clean` / `noise` / `adv` | `adv` |
| `n_tokens` | 5.2절 | `512` |
| `ended_eos` | 5.2절 | `False` |
| `hit_cap` | 5.2절 | `True` |
| `latency_s` | 5.2절 | `18.2` |
| `text` | 5.2절 | … |

### 내부 측정 CSV (6단계)

열: `model`, `case`, `image_id`, `feat_change_clean`, `attack_shift`, `attack_dir_kept`, `shake_ratio`, `eos_gap_mean`, `eos_gap_min`, `clean_end_margin`. 줄 순서는 경우 → 사진이다.

### 결과 JSON

결과 CSV마다 같은 이름의 JSON을 둔다. 아래를 적는다.

- `settings_version`, 결과를 만든 코드의 git 커밋
- `model_id`, `revision`, `eos_token_ids`, dtype, attention 구현
- GPU 이름, 패키지 버전(python, torch, transformers, bitsandbytes, pillow, numpy), CUDA 버전
- 5.1절의 결정적 실행 설정과, 돌지 않은 설정이 있었다면 그 내용
- 경우마다 census, 가중치 변화(%)
- 입력 사진의 SHA-256(input-manifest와 같은 값)
- CSV의 줄 수와 SHA-256
- 끊겼다가 이어서 돌렸다면 그 사실과 이어 붙인 지점
- 6단계는 사진마다 a와 c의 토큰 id

### 공격 사진 세트와 시도 기록 (3단계)

`공통/attack_set.csv` 열: `set_order`(1~30), `cand_order`(image_list의 order), `coco_id`

`<팀>/adv/attempts.csv`는 처리한 후보의 시도마다 한 줄이다. `excluded` 후보는 `attempt`를 0으로 한 줄만 적고, 공격 사진에 관한 열(`ref_adv_n_tokens`, `ref_adv_hit_cap`, `linf`, `png_sha256`)은 비운다.

| 열 | 뜻 |
|---|---|
| `cand_order`, `coco_id` | 후보 |
| `attempt` | 시도 번호(1부터) |
| `status` | `excluded` / `success` / `fail` |
| `ref_clean_n_tokens`, `ref_clean_hit_cap` | ref에 정상 사진을 넣은 답 |
| `ref_adv_n_tokens`, `ref_adv_hit_cap` | ref에 그 시도의 공격 사진을 넣은 답 |
| `linf` | 공격 사진과 정상 사진의 픽셀 차이 최댓값(0~255 눈금) |
| `png_sha256` | 그 시도의 공격 PNG의 SHA-256 |

1-1단계에서 정한 '시도마다 달라지는 값'은 열을 더해 남긴다.

## 8. 폴더

```
README.md      이 문서(규칙 v2)
공통/          settings.py · quant_rtn.py · image_list.csv · attack_set.csv(3단계에서 생김)
이미지생성/     attack_smolvlm.py(v1 제작 코드) · README.md(1-1단계에서 생김)
A팀/  B팀/     code/ · adv/ · results/
```

## 9. Git

- `main`에 직접 push하지 않는다. 브랜치를 만들어 PR로 올리고 승인을 받아 병합한다. main은 GitHub 설정으로 보호되어 직접 push, force push, 삭제가 막혀 있다.
- `A팀/`, `B팀/`을 바꾸는 PR은 그 팀이 승인한다(`.github/CODEOWNERS`). `README.md`, `공통/`, `이미지생성/`을 바꾸는 PR은 두 팀에서 한 명씩 승인한다.
- 커밋 메시지는 `[공통] 내용`, `[A] 내용`, `[B] 내용`처럼 쓴다.
- PR 하나에는 한 단계의 결과만 담는다.
- 결과 파일은 코드가 쓴 그대로 올린다. 노트북은 출력을 지우고 올린다.
- 결과를 만든 코드는 그 결과보다 먼저, 또는 같은 PR로 올린다.
- 저장소는 공개다. 2장의 '올리지 않는 것'을 지킨다.
- 공격 사진은 우리 기기와 우리가 띄운 모델에만 넣는다.
