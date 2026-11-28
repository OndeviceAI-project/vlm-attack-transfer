# 원본 모델에서 성공한 Adversarial Attack, 압축 모델에도 전이되는가

온디바이스 AI 7조 · 규칙 v2 (`공통/settings.py`의 `SETTINGS_VERSION = "v2"`)

v2에서는 공격 사진부터 모든 실험을 새로 한다. v1(중간발표까지의 실험, 결과, 자료)은 저장소에서 내렸고, 태그 `v1-midterm`에 그대로 남아 있다.

## 1. 실험 목표

사진에 사람 눈에 보이지 않는 노이즈(픽셀마다 최대 8/255)를 넣어, VLM이 답을 끝내지 못하게 만드는 공격 사진이 있다. 공격 사진은 압축하지 않은 원본(bf16) 모델로 만든다.

이 저장소의 실험은 두 가지를 잰다.

- 원본 모델에서 성공한 공격 사진을 비전 인코더만 압축한 모델, 언어모델만 압축한 모델, 둘을 함께 압축한 모델에 넣었을 때 나오는 답
- 같은 압축 모델에 공격하지 않은 사진을 넣었을 때 나오는 답

| 팀 | 모델 |
|---|---|
| A | SmolVLM-256M-Instruct |
| B | Qwen2-VL-2B-Instruct |

두 팀은 같은 COCO 사진 200장으로 공격 사진을 만들고, 같은 경우, 같은 입력, 같은 형식으로 잰다.

9단계는 위 두 가지와 따로 한다. 압축 모델로 만든 공격 사진을 그 압축 모델과 다른 경우에 넣었을 때 나오는 답을 잰다. 1~7단계에서 쓰는 공격 사진은 모두 원본 모델로 만든 것이고, 9단계의 사진과 결과는 파일을 따로 둔다.

분석 지표(전이율, 정확도 등)는 실험이 모두 끝난 뒤 8단계에서 정한다. 1~7단계에서는 이 문서가 정한 원자료만 남긴다. 9단계에서 세는 값은 9단계 절에 미리 적어 둔다.

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
| 1-2 | ref 점검 | 팀별 | 0 | `<팀>/code/`, `<팀>/results/check_ref.json` |
| 1-3 | 경우 점검 | 팀별 | 2 | `<팀>/results/check_cases.json`(`passed`가 true) |
| 2 | 사진 준비 | 팀별 | 1-2 | `<팀>/results/input-manifest.json` |
| 3 | 공격 사진 세트 확정 | A·B 함께 | 1-1, 2 | `공통/attack_set.csv`, `<팀>/adv/` |
| 4 | 비전 인코더 실험 | 팀별 | 1-3, 3 | `<팀>/results/vision.csv`, `.json` |
| 5 | 언어모델 실험 | 팀별 | 1-3, 3 | `<팀>/results/language.csv`, `.json` |
| 5-2 | 함께 압축 실험 | 팀별 | 1-3, 3 | `<팀>/results/both.csv`, `.json` |
| 6 | 모델 내부 측정 | 팀별 | 1-3, 3 | `<팀>/results/internal.csv`, `.json` |
| 7 | 정확도 세트 실행 | 팀별 | 1-3, 2 | `<팀>/results/accuracy.csv`, `.json` |
| 8 | 분석 지표 정의 | A·B 함께 | 4~7 | 지표 정의를 이 문서에 추가 |
| 9-1 | 압축 모델에서 공격 사진 만들기 | 팀별 | 4~7 | `<팀>/adv_direct/`(PNG, `SHA256SUMS`, `attempts.csv`, `attempts.json`) |
| 9-2 | 만든 사진을 여러 경우에 넣기 | 팀별 | 9-1 | `<팀>/results/direct.csv`, `.json` |

2단계는 1-2단계(`check_ref.json`)가 끝난 뒤에, 4~7단계는 1-3단계(`check_cases.json`)가 끝난 뒤에 한다. 이 문서에서 '4~7단계'는 5-2단계를 포함한다. 1-3단계는 dev 사진과 `input-manifest.json`을 쓰므로 2단계 뒤에 하고, 3단계와는 상관없이 해도 된다. 4·5·5-2·6단계는 1-3단계와 3단계가 끝나면 한다. 7단계는 공격 사진이 필요 없어서 1-3단계와 2단계가 끝나면 언제든 한다. 4·5·5-2·7단계를 먼저 하고 6단계는 마지막에 한다. 9단계는 4~7단계가 끝난 뒤에 한다. 9단계의 실행은 8단계와 함께 진행해도 되지만, 9단계 결과 PR은 8단계가 끝난 뒤에 올린다(9단계 절).

### 1-1단계. 공격 사진 만드는 방법 합의

두 팀이 함께 정해 `이미지생성/README.md`에 적는다. 이 파일이 생기기 전에는 공격 사진을 만들지 않는다. 반드시 적을 것은 다음과 같다.

- 모델별 제작 코드의 위치와 실행 방법
- 시도 한 번의 설정. `settings.py`의 공격 사진 값을 쓴다
- 사진 한 장당 최대 시도 횟수. `settings.ATTACK_MAX_TRIES`에도 적는다
- 시도마다 무엇이 달라지는지와 그 값을 정하는 방법. 그 값은 `attempts.csv`에 열로 남긴다
- 제작 정밀도. 두 모델 모두 bf16이다(`settings.MODELS[모델]["attack_dtype"]`)
- 제작 코드가 모델에 넣는 `pixel_values`가, 평가 코드가 같은 PNG로 만드는 값과 같다는 확인 방법과 결과(차이 0)

`이미지생성/attack_smolvlm.py`는 v1의 SmolVLM 제작 코드를 1-1단계에서 정한 방식(bf16, 시도마다 시작점, 멈춘 시점의 사진 저장, 목표 문장을 토큰 id 그대로 붙이기)으로 고친 것이다. Qwen2-VL 제작 코드는 `B팀/code/attack_qwen.py`다.

### 1-2단계. ref 점검

각 팀이 2·3단계에 쓸 코드(모델 불러오기, 답 생성, 사진 준비, 공격 사진 시도)를 `<팀>/code/`에 올리고, ref와 dev 사진 5장으로 아래를 점검해 `<팀>/results/check_ref.json`에 남긴다. 하나라도 통과하지 못하면 2단계로 가지 않는다.

1. 모델: `model_id`, `revision`이 settings 값과 같다.
2. 종료 토큰: 모델 generation_config의 `eos_token_id` 목록이 `settings.MODELS[모델]["eos_token_ids"]`와 같다.
3. 전처리: dev 5장의 정상·노이즈 PNG 각각에서, 프로세서가 만든 `pixel_values`가 PNG를 직접 정규화한 값과 같다(차이 최댓값 1e-6 이하, float32와 모델에 넣는 bf16 모두). 프로세서가 사진 크기를 다시 바꾸지 않는다는 확인이다. Qwen2-VL은 직접 계산한 값을 프로세서와 같은 조각 배열 모양으로 바꿔 비교한다.
4. 재현성: ref에 dev 정상 사진 5장을 두 번씩 넣어 두 답이 토큰까지 같다.

### 1-3단계. 경우 점검

4~7단계에 쓰는 32경우(`settings.CASES_ALL`)가 의도대로 만들어지는지 점검한다. 경우를 만드는 코드(`공통/cases.py`)와 점검 코드(`공통/check_cases.py`)는 두 팀이 같은 것을 쓴다. 팀이 할 일은 자기 실험 기계에서 점검을 실행하고 결과 파일을 올리는 것이다. 하나라도 통과하지 못하면 4~7단계로 가지 않는다.

**할 일(팀마다)**

1. 2·3단계를 한 그 기계, 그 패키지 환경에서 최신 main을 받는다(기계를 옮겼다면 5.1절의 조건을 따른다). `공통/`, `이미지생성/`, `<팀>/code/`에 커밋하지 않은 수정이 없게 한다(있으면 결과의 `code_commit`에 `-dirty`가 붙고 점검을 통과하지 못한다).
2. 저장소 폴더에서 `python 공통/check_cases.py --team A`를 실행한다(B팀은 `--team B`). 2단계에서 만든 dev 1번 정상 PNG가 `<팀>/data/dev/clean/`에 그대로 있어야 한다.
3. `<팀>/results/check_cases.json`이 생긴다. `passed`가 `true`인지 본다.
4. 이 파일 하나만 PR로 올린다. 커밋 메시지와 PR 제목은 `[A] 1-3단계 결과: check_cases.json`처럼 쓴다. PR 본문에는 `passed`, 여섯 점검의 `pass`, bnb 경우의 가중치 변화, 답이 512에 닿은 경우의 이름, `bnb_repeat_differs`를 파일에서 옮겨 적는다.

**지킬 것**

- `<팀>/code/runtime.py`를 고치지 않는다. 3단계 시도 기록(`attempts.json`)에 이 파일의 SHA-256이 적혀 있고, 3단계의 성공 판정과 4단계의 측정이 같은 코드의 ref를 썼다는 근거가 그 해시다.
- `공통/`의 파일을 팀에서 고쳐서 쓰지 않는다. 고쳐야 하면 공통 PR로 올려 두 팀이 승인한 뒤, 두 팀 모두 점검을 다시 한다.
- 통과하지 못하거나 오류가 나도, 통과시키려고 설정(결정적 실행, 허용 차이, 경우 목록, 확인값)을 바꾸지 않는다. 나온 `check_cases.json`을 그대로 PR로 올리되 병합하지 않고, 두 팀이 원인과 처리 방법을 함께 정한다.
- `check_cases.json`을 손으로 고치지 않는다.
- GPU와 패키지 버전은 그 팀의 `check_ref.json`에 적힌 것과 같아야 한다(5.1절). 점검 코드가 확인하고, 다르면 통과하지 못한다.

**점검 내용(코드가 확인한다)**

1. 실행 환경: GPU, CUDA, 패키지 버전, 결정적 실행 설정, 모델(`model_id`, `revision`, 종료 토큰, dtype, attention 구현)이 그 팀 `check_ref.json`의 `environment`와 같고, `check_ref.json`의 `passed`가 true다. 코드 커밋은 달라도 된다.
2. 코드: 결과의 `code_commit`이 커밋 해시이고 `-dirty`가 붙지 않았다.
3. 입력: dev 1번 정상 PNG(`order` 1, `000000000632`)의 SHA-256이 `input-manifest.json`의 값과 같다.
4. census: 32경우 모두, 대상 부품의 Linear는 전부 바뀌고 나머지 부품은 하나도 바뀌지 않는다(`quant_rtn.check_census`). 부품별 Linear 수가 `settings.MODELS[모델]["linear_count"]`와 같다.
5. 가중치 변화: rtn 경우의 값이 6장 '양자화 확인값'과 0.01%p 안에서 같다. rand 경우는 같은 부품·비트의 rtn을 잰 값과 0.01%p 안에서 같다. `VL` 경우는 V와 L을 따로 재서 각각 비교한다. bnb 경우는 값을 재서 기록만 한다.
6. 생성: 경우마다 dev 1번 정상 사진의 답을 두 번 생성해 기록한다. ref·rtn·rand 경우는 두 답이 같아야 한다. bnb 경우는 두 답이 다르면 그 사실만 `bnb_repeat_differs`에 기록한다. 답이 상한에 닿아도 오류가 아니라 기록할 사실이다.

어느 경우에서 오류가 나면 그 경우의 `error`에 내용을 적고 다음 경우로 넘어간다. 그 경우는 census와 생성 점검을 통과하지 못한 것으로 센다.

### 2단계. 사진 준비

`공통/image_list.csv`의 dev 5장, cand 300장, acc 300장 전부의 정상·노이즈 PNG를 5.3절대로 만든다. 만든 PNG와 내려받은 COCO JPEG의 SHA-256을 `<팀>/results/input-manifest.json`에 적는다. split을 나눠 만들면 한 파일에 합친다. PNG와 JPEG는 올리지 않는다.

### 3단계. 공격 사진 세트 확정

두 팀이 같은 사진 200장(`settings.N_ATTACK_SET`)으로 된 공격 사진 세트를 만든다.

1. **후보 순서.** `image_list.csv`의 cand 300장(`settings.N_CAND`)을 `order` 순서대로 처리한다. 건너뛰지 않는다. 두 팀의 순서는 같다.
2. **후보 한 장의 처리.** 각 팀이 자기 모델로 한다.
   - 원본(ref)에 정상 사진을 넣은 답이 종료 토큰 없이 512에 닿으면(5.2절 `hit_cap`) 그 후보는 `excluded`로 적고 공격 사진을 만들지 않는다.
   - 1-1단계에서 정한 방법으로 공격 사진을 만든다. 공격 사진 한 장을 만드는 것이 시도 한 번이다.
   - 시도마다 저장한 PNG를 다시 읽어 판정한다. 정상 PNG와의 픽셀 차이 최댓값이 `EPS`(8) 이하이고, 4단계와 같은 코드·환경의 ref에 넣은 답이 종료 토큰 없이 512에 닿으면 `success`, 아니면 `fail`이다. 제작 코드가 출력하는 성공 여부는 판정에 쓰지 않는다.
   - `success`가 나오면 그 후보는 끝난다. `fail`이면 `ATTACK_MAX_TRIES`까지 다시 시도한다.
3. **세트 정하기.** 각 팀은 cand 300장을 끝까지 모두 처리한다. 상대 팀의 진행을 기다리거나 맞추지 않는다. 두 팀 모두 처리가 끝나면, 두 팀 모두 `success`가 나온 후보를 `order` 순서대로 앞에서 200장 고른다. 세트를 정하기 전에 `N_ATTACK_SET`을 바꾸는 PR은 settings 버전을 올리지 않는다. 시도 기록과 사진은 이 값과 관계없다.
4. **후보가 모자랄 때.** 공통 `success`가 200장이 되지 않으면, 먼저 cand를 같은 규칙(COCO val2017 ID 오름차순으로 다음 사진)으로 100장씩 늘리는 PR을 올리고, 두 팀 모두 늘린 후보를 끝까지 처리한 뒤 3번처럼 다시 고른다. cand를 늘리는 PR은 `N_CAND`와 `image_list.csv`(cand 줄 추가)만 바꾸고 settings 버전은 올리지 않는다. 이미 처리한 후보의 기록과 사진은 바뀌지 않으므로, 각 팀은 같은 기록에 이어서 처리한다. cand는 정렬 순위 500번째까지만 늘린다(501번째부터는 acc를 뽑는 범위). 지금 cand는 307번째까지라서 한 번 늘리면 407번째까지, 두 번째에는 500번째까지 93장이다. 500번째까지 늘려도 200장이 되지 않으면 두 팀이 다시 정해 이 문서와 settings를 고친다.
5. **올리기와 고정.** 세트는 `공통/attack_set.csv`에 적는다. 각 팀은 세트 200장의 공격 PNG를 `<팀>/adv/<coco_id 12자리>.png`로 올리고, `<팀>/adv/SHA256SUMS`(`shasum -a 256 *.png`)를 붙인다. 확정한 뒤에는 바꾸지 않는다. 다시 만들어야 하면 settings 버전을 올리고 3단계를 처음부터 한다.
6. **순서 지키기.** 세트가 main에 병합되기 전에는 압축 모델(ref가 아닌 경우)에 공격 사진을 넣지 않는다. 압축 모델에서 나온 결과로 세트의 사진을 고르거나 바꾸지 않는다.
7. **시도 기록.** 처리한 모든 후보와 모든 시도를 `<팀>/adv/attempts.csv`에 한 줄씩 남긴다(형식은 7장). 세트에 들지 않은 시도의 PNG는 올리지 않는다.

### 4·5·5-2·7단계의 실행

네 단계는 넣는 사진과 경우 목록만 다르고 하는 일은 같다. 두 팀이 같은 코드(`공통/run_answers.py`)로 실행한다. 6단계는 `공통/run_internal.py`로 한다(6단계 절).

| 단계 | 명령 (B팀은 `--team B`) | 결과 (`<팀>/results/`) |
|---|---|---|
| 4 | `python 공통/run_answers.py --team A --step vision` | `vision.csv`, `vision.json` |
| 5 | `python 공통/run_answers.py --team A --step language` | `language.csv`, `language.json` |
| 5-2 | `python 공통/run_answers.py --team A --step both` | `both.csv`, `both.json` |
| 7 | `python 공통/run_answers.py --team A --step accuracy` | `accuracy.csv`, `accuracy.json` |

**코드가 하는 일**

1. 시작 조건을 확인하고, 하나라도 맞지 않으면 모델을 올리지 않고 멈춘다.
   - 실행 환경이 그 팀 `check_ref.json`과 같고, 코드에 커밋하지 않은 수정이 없다(1-3단계 점검 1·2번과 같은 확인).
   - 그 팀 `check_cases.json`의 `passed`가 true다.
   - 넣을 사진의 SHA-256이 기록과 같다. 정상·노이즈 PNG는 `input-manifest.json`, 공격 PNG는 `adv/SHA256SUMS`와 비교한다.
2. 경우(settings 목록 순서)마다 `cases.build`로 모델을 새로 올려, 사진마다 `clean`, `noise`, `adv` 순으로 `rt.answer`를 한 번씩 부른다.
3. 4·5·5-2단계는 첫 경우인 ref가 끝나면 3단계 기록과 맞는지 본다. ref의 공격 사진 답 200개가 모두 512에 닿고, 정상 사진 답은 하나도 닿지 않으며, 정상 사진 답의 토큰 수가 `adv/attempts.csv`의 `ref_clean_n_tokens`와 사진마다 같아야 한다. 3단계와 같은 코드, 같은 환경이면 그대로 재현된다. 다르면 멈춘다.
4. 경우 하나가 끝날 때마다 `<팀>/data/runs/<step>/`에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는 건너뛴다). 환경, 코드, 넣는 사진이 처음과 다르면 이어서 하지 않고 멈춘다.
5. 모든 경우가 끝나면 `<팀>/results/<step>.csv`와 `<step>.json`을 쓴다. 끝나기 전에는 결과 폴더에 아무것도 쓰지 않는다.

**할 일(팀마다)**

1. 2·3단계를 한 그 기계, 그 패키지 환경에서 최신 main을 받는다(기계를 옮겼다면 5.1절의 조건을 따른다). 자기 팀의 1-3단계 결과가 main에 병합되어 있어야 한다.
2. 먼저 짧은 시험을 한다: `python 공통/run_answers.py --team A --step vision --limit 2 --cases ref,rtn-V-4`. 사진 2장, 경우 2개만 돌리고 결과는 `<팀>/data/trial/vision/`에 쓴다. 오류 없이 끝나고, `vision.json`의 `ref_check`와 `checks`가 모두 통과인지 본다.
3. 7단계를 먼저 하고, 4단계, 5-2단계, 5단계 순으로 한다. 7단계는 답이 짧아 가장 빨리 끝난다.
4. 단계가 끝나면 그 단계의 `.csv`와 `.json` 두 파일만 PR로 올린다. 단계마다 PR 하나다. 커밋 메시지와 PR 제목은 `[A] 4단계 결과: vision.csv·vision.json`처럼 쓴다. PR 본문에는 `.json`에서 옮겨 적는다: `csv`의 줄 수, `ref_check`, 경우마다 `clean_hit_cap`(정상 사진 답이 512에 닿은 수), `runs`(몇 번에 나눠 돌렸는지), `environment`의 `code_commit`·GPU·패키지 버전.

**지킬 것**

- 1-3단계의 '지킬 것'과 같다. `<팀>/code/runtime.py`와 `공통/`의 파일을 고치지 않고, 통과시키려고 설정을 바꾸지 않으며, 결과 파일을 손으로 고치지 않는다.
- 멈추면 화면의 '멈춤:' 뒤에 나온 글을 그대로 알린다. `--restart`(처음부터 다시)는 코드가 그렇게 하라고 알릴 때만 쓴다.
- 도는 동안 `<팀>/data/` 폴더를 지우거나 옮기지 않는다. 정상·노이즈 사진과 이어 하기 기록이 들어 있다.
- 한 단계를 여러 기계에 나눠 돌리지 않는다.

### 4단계. 비전 인코더 실험

공격 사진 세트 200장의 정상·노이즈·공격 사진을 원본 모델과 비전 인코더만 압축한 모델에 넣고 답을 기록한다.

- 경우: `settings.CASES_VISION` 13경우

  | 방식 | 경우 | 뜻 |
  |---|---|---|
  | 원본 | `ref` | bf16 원본 |
  | rtn | `rtn-V-8`, `rtn-V-6`, `rtn-V-5`, `rtn-V-4`, `rtn-V-3` | 비전 인코더 Linear weight를 그룹 64 가짜 양자화 |
  | bnb | `bnb-V-8`, `bnb-V-4` | bitsandbytes 8비트(LLM.int8), 4비트(nf4) |
  | rand | `rand-V-8`, `rand-V-6`, `rand-V-5`, `rand-V-4`, `rand-V-3` | 대조군. 같은 비트 rtn이 가중치마다 바꾸는 크기만큼, 방향(+/−)만 무작위로 바꿈 |

- 입력: `공통/attack_set.csv`의 200장 × `clean`, `noise`, `adv` = 600개. 모든 경우에 같은 600개를 넣는다.
- 경우 만들기와 확인: 5.4절. 경우마다 census와 가중치 변화를 JSON에 남긴다.
- 출력: `<팀>/results/vision.csv`(13 × 600 = 7,800줄)와 `vision.json`. 형식은 7장.
- 정상 사진이나 노이즈 사진의 답이 상한에 닿은 줄도 지우지 않는다.

### 5단계. 언어모델 실험

4단계와 같고, 압축하는 부품만 언어모델이다.

- 경우: `settings.CASES_LANGUAGE` 13경우(`ref`, `rtn-L-8/6/5/4/3`, `bnb-L-8/4`, `rand-L-8/6/5/4/3`)
- 입력: 4단계와 같은 600개
- 출력: `<팀>/results/language.csv`(7,800줄)와 `language.json`

### 5-2단계. 함께 압축 실험

4단계와 같고, 비전 인코더와 언어모델을 함께 압축한다. 연결부와 출력층은 그대로 둔다.

- 경우: `settings.CASES_BOTH` 8경우(`ref`, `rtn-VL-8/6/5/4/3`, `bnb-VL-8/4`). 두 부품에 같은 방식, 같은 비트를 쓴다. rand 경우는 두지 않는다.
- 입력: 4단계와 같은 600개
- 출력: `<팀>/results/both.csv`(8 × 600 = 4,800줄)와 `both.json`

### 6단계. 모델 내부 측정

답을 채점하지 않고 모델 안의 값을 잰다. 두 팀이 같은 코드(`공통/run_internal.py`)로 실행한다.

- 경우: `settings.CASES_ALL` 32경우(4·5·5-2단계 경우 전부, `ref`는 한 번)
- 입력: 공격 사진 세트 200장의 `clean`, `adv`
- 그림 정보 f(x): 사진 x와 질문을 넣고 한 번 계산(forward)할 때 연결부(`parts["C"]`) 모듈이 내는 출력 전체를 1차원으로 편 벡터. 사진만으로 정해지는 값이므로 아래 로짓을 구하는 계산에서 함께 얻는다. 아래 식은 float64로 바꿔 계산한다.
- ref에서 먼저 준비하는 것(사진마다):
  - a: ref가 공격 사진에 생성한 답의 앞 64토큰. 5.2절의 생성에서 `max_new_tokens`만 64로 줄여 만든다(샘플링이 없으므로 512까지 생성한 답의 앞 64토큰과 같다). 그 안에 종료 토큰이 있으면 멈춘다.
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
- ref 줄과 언어모델만 압축한 경우(L)는 그림 정보가 ref와 같으므로 `feat_change_clean` 0, `attack_dir_kept` 1, `shake_ratio` 0이 나와야 한다. 다르면 원인을 확인하고 JSON에 적는다. 코드는 그림 정보가 ref와 값까지 같은 사진 수를 경우마다 `same_features`에 적고, 언어모델만 압축한 경우 가운데 200장이 아닌 것이 있으면 `checks.language_features`에 그 경우 이름을 적는다.
- 출력: `<팀>/results/internal.csv`(32 × 200 = 6,400줄)와 `internal.json`

**코드가 하는 일**

1. 시작 조건을 확인하고, 하나라도 맞지 않으면 모델을 올리지 않고 멈춘다. '4·5·5-2·7단계의 실행'의 시작 조건과 같다(실행 환경, 코드, `check_cases.json`의 `passed`, 넣을 사진의 SHA-256).
2. 첫 경우인 ref에서 사진마다 a와 c를 만들고, 3단계 기록과 맞는지 본다. 정상 사진 답의 토큰 수가 `adv/attempts.csv`의 `ref_clean_n_tokens`와 사진마다 같아야 한다. 다르면 멈춘다.
3. 경우(`settings.CASES_ALL` 순서)마다 `cases.build`로 모델을 새로 올려, 사진마다 두 번 계산한다. 공격 사진과 질문 뒤에 a를 붙인 계산에서 f(adv)와 `eos_gap`을, 정상 사진과 질문 뒤에 c를 붙인 계산에서 f(clean)과 `clean_end_margin`을 얻는다. 입력은 5.2·5.3절과 같게 만든다.
4. ref에서는 다시 계산한 로짓이 ref가 생성한 답(a의 64토큰, c의 토큰, c 다음의 종료 토큰)을 고르는 비율을 `ref_check.replay`에 적는다. 입력과 자리 계산이 5.2절의 생성과 같다는 확인이다. bf16에서는 로짓이 동점인 자리가 있어 1이 되지 않을 수 있다. 0.8보다 낮으면 멈춘다.
5. 경우 하나가 끝날 때마다 `<팀>/data/runs/internal/`에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는 건너뛴다). 환경, 코드, 넣는 사진이 처음과 다르면 이어서 하지 않고 멈춘다.
6. 모든 경우가 끝나면 `<팀>/results/internal.csv`와 `internal.json`을 쓴다. 끝나기 전에는 결과 폴더에 아무것도 쓰지 않는다.

**할 일(팀마다)**

1. 4·5·5-2·7단계를 한 그 기계, 그 패키지 환경에서 최신 main을 받는다(기계를 옮겼다면 5.1절의 조건을 따른다). 커밋하지 않은 수정이 없게 한다.
2. 먼저 짧은 시험을 한다: `python 공통/run_internal.py --team A --limit 2 --cases rtn-V-4,rtn-L-4`(B팀은 `--team B`). 사진 2장, ref와 두 경우만 돌리고 결과는 `<팀>/data/trial/internal/`에 쓴다. 오류 없이 끝나고, `internal.json`의 `ref_check`의 `pass`와 `replay_pass`, `checks`가 모두 통과인지 본다.
3. 본 실행을 한다: `python 공통/run_internal.py --team A`. 답을 512토큰까지 생성하지 않으므로 4단계보다 훨씬 빨리 끝난다.
4. `internal.csv`와 `internal.json` 두 파일만 PR로 올린다. 커밋 메시지와 PR 제목은 `[A] 6단계 결과: internal.csv·internal.json`처럼 쓴다. PR 본문에는 `.json`에서 옮겨 적는다: `csv`의 줄 수, `ref_check`, `checks`, 경우마다 `same_features`, `runs`, `environment`의 `code_commit`·GPU·패키지 버전.

지킬 것은 '4·5·5-2·7단계의 실행'의 '지킬 것'과 같다.

### 7단계. 정확도 세트 실행

공격 사진 없이, 압축이 평소 답에 주는 변화를 재기 위한 원자료를 만든다.

- 경우: `settings.CASES_ALL` 32경우
- 입력: `image_list.csv`의 acc 300장 × `clean`, `noise` = 600개
- 출력: `<팀>/results/accuracy.csv`(32 × 600 = 19,200줄)와 `accuracy.json`. 형식은 4단계와 같다.
- 답을 채점하는 방법은 8단계에서 정한다.

### 8단계. 분석 지표 정의

4~7단계가 모두 끝나면 두 팀이 분석 지표의 정의를 이 문서에 추가한다. 계산한 지표 값, 그래프, 해석은 2장에 따라 저장소에 올리지 않는다.

### 9단계. 압축 모델에서 만든 공격 사진

1~7단계의 공격 사진은 모두 원본(ref)으로 만들었다. 9단계는 공격 사진을 압축 모델로 만들어, 그 모델과 다른 경우에 넣었을 때의 답을 기록한다. 1~7단계의 사진, 기록, 결과는 바꾸지 않는다.

- 출발 경우(공격 사진을 만드는 모델): `settings.CASES_DIRECT` 4경우. `rtn-V-4`, `rtn-V-3`, `rand-V-3`, `bnb-V-4`다. 두 팀이 같은 경우를 쓴다.
- 사진: `공통/attack_set.csv`의 `set_order` 1~30(`settings.N_DIRECT`). 4단계 결과와 같은 사진끼리 맞대 보려고 세트의 앞에서부터 쓴다.
- 도착 경우(만든 사진을 넣어 보는 모델): `settings.CASES_DIRECT_TARGETS`. 4단계의 13경우와 같다.
- 9-1단계(만들기)는 팀의 코드로, 9-2단계(넣어 보기)는 두 팀이 같은 코드(`공통/run_direct.py`)로 한다.
- 2~7단계를 한 그 기계, 그 패키지 환경에서 한다(5.1절).

**순서와 고정**

- 9단계는 그 팀의 4~7단계 결과가 모두 main에 병합된 뒤에 시작한다.
- 출발 경우, 사진 수, 도착 경우, 아래 '9단계에서 세는 값'은 9-1단계 본 실행을 시작하기 전에만 바꿀 수 있다. 사진 수를 줄이는 것만 아래 규칙대로 시작한 뒤에도 할 수 있다. 바꿀 때는 공통 PR로 올려 두 팀이 승인한다.
- 코드 확인과 짧은 시험은 dev 사진으로만 한다. 세트 사진으로 미리 돌려 보지 않는다.
- `bnb-V-4`에서 9-1단계 코드가 돌지 않으면(dev 사진 시험에서 확인한다), 본 실행 전에 `CASES_DIRECT`에서 이 경우를 빼는 PR을 올린다. 한 팀에서만 돌지 않아도 두 팀 모두 뺀다.
- 사진 수를 줄여야 하면 `N_DIRECT`를 줄인다. 두 팀이, 모든 출발 경우에 같은 수를 쓰고 세트의 앞에서부터 쓴다. 결과를 보고 사진을 고르지 않는다.
  - 본 실행을 시작한 뒤에 줄일 때는 줄이는 이유와 그때까지 처리한 범위를 PR 본문에 적는다.
  - 이미 만든 것 가운데 줄인 수 밖의 사진(`set_order`가 `N_DIRECT`보다 큰 것)의 기록과 PNG는 지우거나 고치지 않고 그대로 둔다. 9-2단계 코드는 그 사진을 넣지 않고, `direct.json`의 `sources`에 그 수(`outside`)만 적는다.
  - '9단계에서 세는 값'은 두 팀 모두 `set_order` 1~`N_DIRECT`만 쓴다. 다른 팀이 9-2단계를 이미 끝냈으면 그 결과 파일은 다시 만들지 않는다.
- 9단계의 실행은 8단계와 함께 진행해도 된다. 9단계 결과 PR(9-1단계의 `adv_direct/`, 9-2단계의 `direct.csv`)은 8단계의 지표 정의가 main에 병합된 뒤에 올린다.

#### 9-1단계. 만들기

3단계와 같은 규칙으로 하되, 원본 대신 출발 경우의 모델을 쓴다. 만드는 방식과 값은 `이미지생성/README.md` 2·3·5장과 `settings.py`의 공격 사진 값 그대로다. 9단계용 값을 따로 두지 않는다. 달라지는 것은 모델뿐이다.

1. **모델.** 출발 경우마다(`settings.CASES_DIRECT` 순서) `cases.build`로 모델을 올린다. 만들기와 판정은 이 모델 한 벌로 한다.
2. **사진 한 장의 처리.** `set_order` 순서대로 하고, 건너뛰지 않는다.
   - 그 모델에 정상 사진을 넣은 답이 종료 토큰 없이 512에 닿으면(5.2절 `hit_cap`) `excluded`로 적고 공격 사진을 만들지 않는다.
   - 공격 사진 한 장을 만드는 것이 시도 한 번이다. 시도마다 달라지는 것(시작점, 씨앗)은 3단계와 같다.
   - 시도마다 저장한 PNG를 다시 읽어 판정한다. 정상 PNG와의 픽셀 차이 최댓값이 `EPS`(8) 이하이고, 그 모델에 `rt.answer`로 넣은 답이 종료 토큰 없이 512에 닿으면 `success`, 아니면 `fail`이다. 제작 코드가 출력하는 성공 여부는 판정에 쓰지 않는다.
   - `success`가 나오면 그 사진은 끝난다. `fail`이면 `ATTACK_MAX_TRIES`까지 다시 시도한다.
3. **끝까지 처리.** 모든 출발 경우에서 `set_order` 1~`N_DIRECT`를 모두 처리한다. `success`가 적게 나와도 그대로 기록한다. `success`를 늘리려고 설정이나 방식을 바꾸지 않는다.
4. **기록과 올리기.** 처리한 모든 사진의 모든 시도를 `<팀>/adv_direct/attempts.csv`에 한 줄씩 남긴다(형식은 7장). `success`가 나온 시도의 PNG만 `<팀>/adv_direct/<출발 경우>/<coco_id 12자리>.png`로 올리고, `<팀>/adv_direct/` 안에서 `shasum -a 256 */*.png`로 만든 `SHA256SUMS`를 붙인다. `fail`인 시도의 PNG는 올리지 않는다.

**코드(팀마다)**

- 9-1단계의 코드는 팀이 만들어 `<팀>/code/`에 올린다. 코드의 위치와 실행 방법을 `이미지생성/README.md` 1장에 더한다. 코드는 결과보다 먼저, 또는 같은 PR로 올린다(9장).
- `<팀>/adv/attempts.json`의 `config`에 SHA-256이 적힌 코드 파일(3단계의 제작 코드, `make_attack_set.py`, `runtime.py`)은 고치지 않는다. 3단계 기록이 그 SHA-256을 근거로 삼는다. 9-1단계에 더 필요한 것은 새 파일에 둔다.
- 모델은 `cases.build`로만 만든다. `공통/`의 파일은 팀에서 고쳐 쓰지 않는다.
- 3단계 코드가 하던 확인을 그대로 한다. 처리할 사진의 정상 PNG가 `input-manifest.json`의 SHA-256과 같은지, 시도마다 저장한 PNG의 `pixel_values`가 5.3절대로 같은지(차이 1e-6 이하) 확인하고, 다르면 멈춘다.
- 끊겨도 이어서 할 수 있게 하고, 제작·판정 설정이나 코드, 실행 환경이 기록과 다르면 이어서 하지 않는다.
- 결과 PR의 커밋 메시지와 제목은 `[A] 9-1단계 결과: adv_direct`처럼 쓴다.

#### 9-2단계. 넣어 보기

9-1단계에서 `success`가 나온 공격 사진을 도착 경우마다 넣어 답을 기록한다. 실행은 `python 공통/run_direct.py --team A`다(B팀은 `--team B`).

**코드가 하는 일**

1. 시작 조건을 확인하고, 하나라도 맞지 않으면 모델을 올리지 않고 멈춘다.
   - 실행 환경이 그 팀 `check_ref.json`과 같고, 코드에 커밋하지 않은 수정이 없으며, `check_cases.json`의 `passed`가 true다('4·5·5-2·7단계의 실행'과 같은 확인).
   - 9-1단계 기록이 온전하다. `attempts.csv`의 줄 수와 SHA-256이 `attempts.json`에 적힌 값과 같고, 만든 때의 실행 환경이 지금과 같으며, 출발 경우마다 `set_order` 1~`N_DIRECT`가 모두 끝까지 처리되어 있다. `set_order`가 `N_DIRECT`보다 큰 사진의 기록은 넣지도 세지도 않는다.
   - 넣을 PNG가 기록과 같다. `success` 시도의 PNG가 모두 있고, SHA-256이 `attempts.csv`의 `png_sha256`, `SHA256SUMS`와 모두 같다. `SHA256SUMS`에 9-1단계 기록의 `success` PNG가 아닌 파일이 없다.
2. 도착 경우(`settings.CASES_DIRECT_TARGETS` 순서)마다 `cases.build`로 모델을 새로 올려, 출발 경우(`settings.CASES_DIRECT` 순서)마다 `success` 사진(`set_order` 순서)을 넣어 `rt.answer`를 한 번씩 부른다. 정상·노이즈 사진은 넣지 않는다. 같은 사진, 같은 경우의 정상·노이즈 답은 4단계 `vision.csv`에 있다.
3. 도착 경우가 출발 경우이기도 하면, 그 경우에서 만든 사진의 답이 모두 512에 닿는지 본다. 9-1단계의 판정이 같은 코드, 같은 환경에서 그대로 재현된다는 확인이다. 다르면 멈춘다.
4. 도착 경우 하나가 끝날 때마다 `<팀>/data/runs/direct/`에 저장한다. 끊기면 같은 명령으로 이어서 한다(끝난 경우는 건너뛴다). 환경, 코드, 넣는 사진이 처음과 다르면 이어서 하지 않고 멈춘다.
5. 모든 도착 경우가 끝나면 `<팀>/results/direct.csv`와 `direct.json`을 쓴다. 끝나기 전에는 결과 폴더에 아무것도 쓰지 않는다.

**할 일(팀마다)**

1. 9-1단계를 한 그 기계에서, `<팀>/adv_direct/`가 그대로 있고 커밋하지 않은 수정이 없는 상태로 시작한다.
2. 먼저 짧은 시험을 한다: `python 공통/run_direct.py --team A --limit 2 --cases ref,rtn-V-4`. 사진 2장, 도착 경우 2개만 돌리고 결과는 `<팀>/data/trial/direct/`에 쓴다. 오류 없이 끝나고, `direct.json`의 `checks`와 `self_check`가 모두 통과인지 본다.
3. 본 실행을 한다: `python 공통/run_direct.py --team A`
4. `direct.csv`와 `direct.json` 두 파일만 PR로 올린다. 커밋 메시지와 PR 제목은 `[A] 9-2단계 결과: direct.csv·direct.json`처럼 쓴다. PR 본문에는 `.json`에서 옮겨 적는다: `csv`의 줄 수, `sources`, `self_check`, `checks`, `runs`, `environment`의 `code_commit`·GPU·패키지 버전.

지킬 것은 '4·5·5-2·7단계의 실행'의 '지킬 것'과 같다.

#### 9단계에서 세는 값

9-1단계 본 실행 전에 정해 둔다. 계산한 값과 해석은 2장에 따라 저장소에 올리지 않는다.

쓰는 사진은 `set_order` 1~`N_DIRECT`다. 출발 경우마다, 그 가운데 `excluded`가 아닌 사진을 유효 사진이라 한다.

| 값 | 정의 |
|---|---|
| 성공 비율 | 마지막 상태가 `success`인 유효 사진 수 ÷ 유효 사진 수. 첫 시도(`attempt` 1)만으로 센 값도 함께 낸다 |
| 반복 수 | 사진마다, `success`가 나온 시도까지의 `iters_run`을 모두 더한 값. 앞선 `fail` 시도의 것도 더한다 |
| 반복 비 | 9-1단계와 3단계 모두 `success`인 사진에서, 사진마다 (9-1단계의 반복 수 ÷ 3단계의 반복 수)를 구한 것의 중앙값 |
| 도착 비율 | 출발 경우 s에서 `success`가 나온 사진을 도착 경우 t에 넣은 답 가운데 `hit_cap`인 것의 비율. t에서 그 사진의 정상 답이 `hit_cap`인 사진(4단계 `vision.csv`)은 빼고 센다 |

- 비교 기준은 3단계 시도 기록(`<팀>/adv/attempts.csv`)이다. 같은 사진(`set_order` 1~`N_DIRECT`)의 값과, 3단계에서 처리한 후보 전체(`excluded` 제외)의 값을 함께 낸다. 세트의 사진은 ref에서 `success`가 나온 것만 고른 것이라, 같은 사진의 ref 성공 비율은 고른 방법 때문에 1이다.
- 같은 사진끼리의 비교는 4단계 `vision.csv`의 `adv` 줄(ref에서 만든 사진)과 `direct.csv`의 줄을 사진마다 맞대어 센다.
- 구분선: 성공 비율 0.8 이상과 0.5 이하, 반복 비 2 이상.
- `iters_run`은 `REFRESH`(20)걸음마다 답 길이를 확인하는 방식 때문에 20걸음 단위로만 달라진다.

## 4. 사진 목록

`공통/image_list.csv`. 모든 사진은 COCO val2017이다.

| split | 장수 | 쓰는 곳 | 고른 방법 |
|---|---|---|---|
| `pilot` | 3 | 쓰지 않음(예비 실험에 썼음) | — |
| `dev` | 5 | 1-2·1-3단계 점검, 공격 사진 설정 조정 | val2017 ID 오름차순 3~7번째 |
| `cand` | 300 | 3단계 후보, `order` 순서대로 | val2017 ID 오름차순 8~307번째 |
| `acc` | 300 | 7단계 | val2017 ID 오름차순 501~5000번째 사진 중 `numpy.random.default_rng(2026).choice(..., 300, replace=False)`로 뽑아 ID 오름차순 정렬 |

dev와 pilot은 결과 CSV에 넣지 않는다. cand의 1~43번째는 v1의 eval 30장과 extra 13장과 같은 사진이다.

## 5. 공통 조건

모든 단계에 적용한다.

### 5.1 모델과 실행 환경

- `from_pretrained`에 `settings.MODELS[모델]`의 `model_id`, `revision`을 준다. dtype은 `settings.DTYPE`(bf16), `attn_implementation`은 `settings.ATTN_IMPLEMENTATION`(sdpa)이다.
- 한 모델의 2~7단계(3단계의 성공 판정 포함)는 모두 같은 기계, 같은 패키지 버전에서 한다. bnb 경우가 있으므로 bf16을 지원하는 CUDA GPU를 쓴다. 실행 환경(GPU 모델, CUDA, 패키지 버전, 결정적 실행 설정)이 달라지면 이어서 하지 않고 1-2단계부터 다시 한다.
- 3단계를 끝낸 뒤에 GPU 모델과 패키지 버전이 같은 다른 기계로 옮겨야 할 때는, 아래를 모두 지키면 3단계까지의 결과를 그대로 쓰고 이어서 한다. 하나라도 통과하지 못하면 새 기계에서 1-2단계부터 다시 한다.
  - 새 기계에서 1-3단계 점검을 한다(이미 했더라도 다시 한다). 실행 환경이 그 팀 `check_ref.json`과 같아야 통과한다.
  - 정상·노이즈 PNG는 옮겨 오거나 새 기계에서 2단계 코드로 다시 만든다. 어느 쪽이든 SHA-256이 `input-manifest.json`의 값과 모두 같아야 하고, `input-manifest.json`은 바꾸지 않는다. 공격 PNG는 다시 만들지 않고 저장소의 것을 쓴다.
  - 새 기계에서 4·5·5-2단계의 ref 확인(ref의 공격 사진 답이 모두 512에 닿고, 정상 사진 답의 토큰 수가 3단계 기록과 사진마다 같음)을 통과한다. 3단계의 성공 판정이 새 기계에서 그대로 재현된다는 근거가 이것이다. 7단계에는 ref 확인이 없으므로, 7단계 결과는 같은 기계에서 한 4·5·5-2단계 가운데 하나가 ref 확인을 통과한 것을 근거로 삼는다.
  - 옮긴 뒤의 단계는 모두 새 기계에서 한다.
  - 옮긴 사실과 위 근거를, 새 기계에서 한 결과 PR의 본문에 적는다.
- B팀은 1-3단계부터 2·3단계를 한 기계가 아닌 다른 기계(같은 RTX 4090, 같은 패키지 버전)에서 했다. 근거는 PR #27(실행 환경, 사진 해시)과 #29(4단계의 ref 확인)에 있다.
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
- 9단계의 공격 사진: `<팀>/adv_direct/<출발 경우>/<coco_id 12자리>.png`. 9단계에서만 쓰고, 1~7단계에는 넣지 않는다
- 사진은 PNG만 쓴다. JPEG로 저장하면 노이즈가 지워진다.
- 같은 JPEG라도 Pillow 버전이 다르면 정상 사진의 픽셀이 다를 수 있다(13.0.0과 11.2.1은 일부 세로 사진의 리사이즈 결과가 다르다). 그래서 각 팀은 2~7단계 내내 자기 `input-manifest.json`의 `packages.pillow` 버전을 쓴다(A 11.2.1, B 13.0.0).
- 같은 PNG라면 공격 사진 제작 코드(3단계, 9-1단계), 4~7단계 코드, 내부 측정 코드, 9-2단계 코드가 모델에 넣는 `pixel_values`가 모두 같아야 한다(차이 0).

### 5.4 경우 만들기

경우 이름은 `<방식>-<부품>-<비트>`다. 부품은 `settings.MODELS[모델]["parts"]`의 V(비전 인코더), L(언어모델), 또는 둘 다인 VL이다. 연결부(C)와 출력층(H)은 어느 경우에도 바꾸지 않는다. 모든 방식은 Linear 층의 weight만 바꾸고, bias·정규화·Conv·임베딩은 그대로 둔다.

경우는 `공통/cases.py`의 `build(rt, 이름, reference)`로 만든다. `rt`는 팀의 `runtime.py`이고, `reference`는 `cases.reference_weights(rt)`가 돌려주는 원본 가중치다. 1-3단계와 4~7단계의 코드는 경우를 직접 만들지 않고 이 함수를 부른다. 함수가 하는 일은 아래와 같다.

- `ref`: `rt.load_ref()`로 bf16 원본을 그대로 올린다.
- `rtn`: bf16 원본을 올린 뒤 `quant_rtn.apply_rtn(model, parts, targets=부품, bits=비트, group=settings.QUANT_GROUP)`
- `rand`: bf16 원본을 올린 뒤 `quant_rtn.apply_rand(model, parts, targets=부품, bits=비트, group=settings.QUANT_GROUP, seed=settings.RAND_SEED)`
- `bnb`: `from_pretrained`에 `quantization_config`와 `device_map={"": 0}`을 준다. 그 밖의 인자(`model_id`, `revision`, dtype, `attn_implementation`)는 ref와 같다. `llm_int8_skip_modules`는 parts 중 대상이 아닌 부품의 이름이다(`bnb-V-4`면 C, L, H의 이름, `bnb-VL-4`면 C, H의 이름).
  - 8비트: `BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)`
  - 4비트: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip)`
  - bitsandbytes의 8비트 층은 bf16 입력을 float16으로 바꿔 계산하고 그때마다 안내 글을 찍는다. `cases.py`는 이 안내 글만 끈다.
- 경우마다 모델을 새로 올린다. 한 모델에 방식을 두 번 적용하지 않는다.
- 경우마다 확인한다: `quant_rtn.check_census(quant_rtn.census(model, parts), targets, settings.PART_NAMES)`(ref는 `targets=""`). 부품별 Linear 수는 `linear_count`와 같아야 한다. bnb 경우는 모델의 종료 토큰 목록이 settings와 같은지도 확인한다(나머지 경우는 `rt.load_ref()`가 확인한다).
- 가중치 변화(%) = 100 × √(Σ‖W_경우 − W_원본‖² ÷ Σ‖W_원본‖²). 합은 대상 부품의 Linear weight 전부에 대해 한다. rtn과 rand는 함수가 돌려주는 값을 쓴다. bnb는 가중치를 실수로 되돌려 같은 식으로 계산한다. 8비트는 `int8 weight × SCB ÷ 127`(행마다), 4비트는 `bitsandbytes.functional.dequantize_4bit(weight, weight.quant_state)`로 되돌린다. `VL` 경우는 V와 L을 따로 계산한다.

## 6. 양자화 확인값

rtn 경우의 가중치 변화(%)다. `settings.RTN_CHANGE_PCT`에 같은 값이 있다. 1-3단계 점검에서 이 값과 0.01%p(`settings.CHANGE_TOL`) 안에서 같아야 한다. rand는 같은 부품·비트의 rtn과 0.01%p 안에서 같아야 한다. `VL` 경우의 V는 V 줄과, L은 L 줄과 비교한다.

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
| 1-2 | `<팀>/results/check_ref.json` |
| 1-3 | `<팀>/results/check_cases.json` |
| 2 | `<팀>/results/input-manifest.json` |
| 3 | `공통/attack_set.csv`, `<팀>/adv/<coco_id 12자리>.png` 200장, `<팀>/adv/SHA256SUMS`, `<팀>/adv/attempts.csv`, `<팀>/adv/attempts.json` |
| 4 | `<팀>/results/vision.csv`, `vision.json` |
| 5 | `<팀>/results/language.csv`, `language.json` |
| 5-2 | `<팀>/results/both.csv`, `both.json` |
| 6 | `<팀>/results/internal.csv`, `internal.json` |
| 7 | `<팀>/results/accuracy.csv`, `accuracy.json` |
| 9-1 | `<팀>/adv_direct/<출발 경우>/<coco_id 12자리>.png`, `<팀>/adv_direct/SHA256SUMS`, `<팀>/adv_direct/attempts.csv`, `<팀>/adv_direct/attempts.json` |
| 9-2 | `<팀>/results/direct.csv`, `direct.json` |

### 답 CSV (4·5·5-2·7단계)

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

열: `model`, `case`, `image_id`, `feat_change_clean`, `attack_shift`, `attack_dir_kept`, `shake_ratio`, `eos_gap_mean`, `eos_gap_min`, `clean_end_margin`. 줄 순서는 경우(`settings.CASES_ALL` 순서) → 사진(attack_set의 `set_order`)이다. 앞의 네 값은 소수 여섯째 자리, 뒤의 세 값은 넷째 자리까지 적는다.

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

### 답 JSON (4·5·5-2·7단계)

`공통/run_answers.py`가 쓴다. 위 '결과 JSON'의 항목을 아래 이름으로 적는다.

| 항목 | 내용 |
|---|---|
| `environment` | 실행 환경(코드의 git 커밋, 모델, GPU, 패키지 버전, 결정적 실행 설정) |
| `step` | `vision` / `language` / `both` / `accuracy` |
| `cases` | 경우마다 `census`, `weight_change_pct`, `rows`(줄 수), `seconds`(걸린 시간), `clean_hit_cap`(정상 사진 답이 512에 닿은 수) |
| `inputs` | 넣은 사진의 SHA-256(`clean_sha256`, `noise_sha256`, `adv_sha256`) |
| `ref_check` | ref의 답이 3단계 기록과 맞는지(4·5·5-2단계). `adv_hit_cap`, `clean_hit_cap`, `clean_tokens_differ` |
| `checks` | 시작 조건 `environment`, `code`, `check_cases` 각각의 `pass` |
| `csv` | CSV의 줄 수(`rows`)와 SHA-256 |
| `runs` | 실행마다 처음 쓴 줄 번호(`first_row`)와 더한 줄 수(`rows_added`). 한 번에 끝냈으면 하나다 |

### 내부 측정 JSON (6단계)

`공통/run_internal.py`가 쓴다. `environment`, `inputs`, `csv`, `runs`와 `checks`의 시작 조건 세 가지는 '답 JSON'과 같다(`inputs`는 `clean_sha256`, `adv_sha256`).

| 항목 | 내용 |
|---|---|
| `step` | `internal` |
| `cases` | 경우마다 `census`, `weight_change_pct`, `rows`(줄 수), `seconds`(걸린 시간), `same_features`(그림 정보가 ref와 값까지 같은 사진 수) |
| `ref_check` | ref의 정상 사진 답이 3단계 기록과 맞는지(`pass`, `clean_tokens_differ`)와, 다시 계산한 로짓이 ref가 생성한 답을 고른 비율(`replay`의 `a`, `c`, `end`, 그리고 `replay_pass`) |
| `tokens` | 사진마다 a와 c의 토큰 id |
| `checks` | 시작 조건 세 가지와 `language_features`(언어모델만 압축한 경우의 그림 정보가 모두 ref와 같은가, 다른 경우의 이름 `differs`) |

### 경우 점검 JSON (1-3단계)

`<팀>/results/check_cases.json`은 `공통/check_cases.py`가 쓴다.

| 항목 | 내용 |
|---|---|
| `environment` | 위 '결과 JSON'의 실행 환경 항목(코드의 git 커밋, 모델, GPU, 패키지 버전, 결정적 실행 설정) |
| `inputs` | dev 1번 정상 PNG의 `coco_id`, `clean_sha256`, `matches_manifest` |
| `cases` | 경우마다 `method`, `targets`, `bits`, `census`(부품별 [바뀐 Linear 수, 전체 Linear 수]), `weight_change_pct`(부품별 %), `answer`(5.2절의 다섯 값), `repeat_same`. 오류가 난 경우는 `error` |
| `checks` | `environment`, `code`, `input`, `census`, `weight_change`, `generation` 각각의 `pass`와 근거. `environment`에는 `check_ref.json`과 다른 항목(`differs`)을 적는다 |
| `passed` | 여섯 점검이 모두 통과했는가 |

### 공격 사진 세트와 시도 기록 (3단계)

`공통/attack_set.csv` 열: `set_order`(1~200), `cand_order`(image_list의 order), `coco_id`

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

`<팀>/adv/attempts.json`은 `attempts.csv`의 결과 JSON이다. 위 '결과 JSON'의 항목을 적는다. 경우는 `ref` 하나이고(census, 가중치 변화 0), 입력 사진은 처리한 후보의 정상 PNG다. 여기에 이어서 돌릴 때 확인하는 제작 설정과, 5.3절 `pixel_values` 확인의 가장 큰 차이(`pixel_gap_max`)를 더한다.

### 9단계 시도 기록 (9-1단계)

`<팀>/adv_direct/attempts.csv`는 출발 경우의 사진마다, 시도마다 한 줄이다. 줄 순서는 출발 경우(`settings.CASES_DIRECT` 순서) → 사진(`set_order`) → 시도다. `excluded`인 사진은 `attempt`를 0으로 한 줄만 적고, 공격 사진에 관한 열(`adv_n_tokens`, `adv_hit_cap`, `linf`, `png_sha256`)은 비운다.

| 열 | 뜻 |
|---|---|
| `source_case` | 출발 경우 이름 |
| `set_order`, `coco_id` | 사진. attack_set의 `set_order`와 COCO id 12자리 |
| `attempt` | 시도 번호(1부터) |
| `status` | `excluded` / `success` / `fail` |
| `clean_n_tokens`, `clean_hit_cap` | 출발 경우의 모델에 정상 사진을 넣은 답 |
| `adv_n_tokens`, `adv_hit_cap` | 출발 경우의 모델에 그 시도의 공격 사진을 넣은 답 |
| `linf` | 공격 사진과 정상 사진의 픽셀 차이 최댓값(0~255 눈금) |
| `png_sha256` | 그 시도의 공격 PNG의 SHA-256 |
| `init`, `init_seed`, `iters_run`, `stopped_early`, `nan_stop`, `attack_seconds` | 3단계와 같다(`이미지생성/README.md` 3장) |

`<팀>/adv_direct/attempts.json`은 `attempts.csv`의 결과 JSON이다. 3단계의 `attempts.json`과 같은 항목을 적되, `cases`에는 출발 경우마다 census와 가중치 변화(`cases.build`가 돌려주는 기록)를 적는다. 9-2단계 코드는 시작할 때 이 파일의 `environment`와 `csv`(줄 수 `rows`, `sha256`)를 확인한다.

`<팀>/adv_direct/SHA256SUMS`의 한 줄은 `<SHA-256>  <출발 경우>/<coco_id 12자리>.png`다.

### 9단계 답 CSV와 JSON (9-2단계)

`공통/run_direct.py`가 쓴다.

`direct.csv`는 한 답이 한 줄이다. 줄 순서는 도착 경우(`settings.CASES_DIRECT_TARGETS` 순서) → 출발 경우(`settings.CASES_DIRECT` 순서) → 사진(`set_order`)이다. 열은 '답 CSV'의 `model` 뒤에 `source_case`를 더한 것이다: `model`, `source_case`, `case`, `image_id`, `input`, `n_tokens`, `ended_eos`, `hit_cap`, `latency_s`, `text`. `source_case`는 출발 경우, `case`는 도착 경우이고, `input`은 모두 `adv`다. 줄 수는 (출발 경우마다 `success` 사진 수를 더한 값) × 13이다.

`direct.json`의 `environment`, `csv`, `runs`와 `checks`의 시작 조건 세 가지는 '답 JSON'과 같다.

| 항목 | 내용 |
|---|---|
| `step` | `direct` |
| `sources` | 출발 경우마다 9-1단계에서 처리한 사진 수(`photos`)와 마지막 상태별 수(`success`, `fail`, `excluded`), `set_order`가 `N_DIRECT`보다 커서 쓰지 않은 사진 수(`outside`) |
| `cases` | 도착 경우마다 `census`, `weight_change_pct`, `rows`(줄 수), `seconds`(걸린 시간), `hit_cap`(출발 경우별로 512에 닿은 답의 수) |
| `inputs` | 9-1단계 `attempts.csv`의 SHA-256(`attempts_csv_sha256`)과 넣은 공격 PNG의 SHA-256(`adv_sha256`, 출발 경우별) |
| `self_check` | 출발 경우마다, 그 경우에서 만든 사진을 그 경우에 넣은 답이 모두 512에 닿았는지(`pass`, `photos`, `hit_cap`, `not_hit`) |
| `checks` | 시작 조건 세 가지와 `direct_attempts`(9-1단계 기록의 확인. 다른 환경 항목 `environment_differs`, 끝나지 않은 사진 수 `unfinished`) |

## 8. 폴더

```
README.md      이 문서(규칙 v2)
공통/          settings.py · quant_rtn.py · cases.py(경우 만들기) · check_cases.py(1-3단계 점검)
               run_answers.py(4·5·5-2·7단계 실행) · run_internal.py(6단계 실행) · run_direct.py(9-2단계 실행)
               image_list.csv
               select_attack_set.py(3단계 세트 고르기) · attack_set.csv(공격 사진 세트)
이미지생성/     attack_smolvlm.py(SmolVLM 제작 코드) · README.md(제작 방식, 1-1단계)
A팀/  B팀/     code/ · adv/ · adv_direct/(9단계) · results/
```

## 9. Git

- `main`에 직접 push하지 않는다. 브랜치를 만들어 PR로 올리고 승인을 받아 병합한다. main은 GitHub 설정으로 보호되어 직접 push, force push, 삭제가 막혀 있다.
- `A팀/`, `B팀/`을 바꾸는 PR은 그 팀이 승인한다(`.github/CODEOWNERS`). `README.md`, `공통/`, `이미지생성/`을 바꾸는 PR은 두 팀에서 한 명씩 승인한다.
- 커밋 메시지와 PR 제목은 `[공통] 내용`, `[A] 내용`, `[B] 내용`처럼 쓴다. PR 제목은 병합 커밋의 메시지에 그대로 들어간다.
- PR 하나에는 한 단계의 결과만 담는다.
- 결과 파일은 코드가 쓴 그대로 올린다. 노트북은 출력을 지우고 올린다.
- 결과를 만든 코드는 그 결과보다 먼저, 또는 같은 PR로 올린다.
- 저장소는 공개다. 2장의 '올리지 않는 것'을 지킨다.
- 공격 사진은 우리 기기와 우리가 띄운 모델에만 넣는다.
