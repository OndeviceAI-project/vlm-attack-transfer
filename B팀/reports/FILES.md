# 업로드 파일 안내와 필드 미리보기

부품 비교 요약은 [quantization_components_comparison_v1/README.md](quantization_components_comparison_v1/README.md)에 있다. 기존 V와 새 C·L·VCL의 5,760행을 서버에서 집계했다. 같은 폴더의 `group_stats.csv`는 192개 조건·입력별 통계, `timing_memory.csv`는 64개 조건별 시간·메모리 보조값, `summary.json`은 대조군 예외·ref 일치 확인, `sources.json`은 원본 해시·환경을 담는다. `build.py`가 원시 CSV를 바꾸지 않고 이 파일들을 재생성한다. 상한 도달 횟수와 공격 성공률·전이율은 구분한다.

아래 기존 V 파일 설명에서 `<실험 ID>`는 `qwen2vl_vision_common_v1_r01` 또는 `qwen2vl_vision_extended_v1_r01`이다. 파일명은 현재 이름이며, 원본 메타데이터 내부의 이전 이름은 보존했다. 새 부품 실험의 추가 파일은 별도 표와 [GitHub 업로드 범위](../experiments/github_upload.md)에 설명한다.

## 파일별 목적과 포함 근거

| 위치 | 내용 | 포함 이유 |
|---|---|---|
| `README.md` | 폴더 구조, 파일 추가·명명 규칙, 공통 코드, 실험별 한 줄 진행 현황·보고서 링크 | B팀 폴더를 사용하는 공통 안내 |
| `requirements.in` | 직접 사용하는 패키지의 고정 버전 | 환경 재구성 |
| `requirements.lock.txt` | 측정 당시 패키지 버전 전체 | 당시 환경 추적 |
| `.gitignore` | 가중치·체크포인트·후보 작업물 제외 | 실험 산출물과 대용량 작업 파일 구분 |
| `src/runtime.py` | 서버 모델 캐시 로딩, 전처리, EOS, V/C/L/VCL 양자화, 생성 시간 | 공통 설정의 입력·EOS·대상 부품·시간 규칙 |
| `src/data.py` | 이미지 준비·해시 검사, 명시적 다운로드 옵션 | 공통 이미지 목록·448 PNG·고정 noise 재현 |
| `src/attack.py` | 공통 최선 후보 저장, 확장 25후보·20% 재검사, 상태 저장 | 실제 사용한 두 공격 방식의 실행 구현 |
| `src/evaluate.py` | 고정 clean/noise/adv 측정, CSV 저장, 기존 세트 재사용 | 공통 결과 형식과 쌍 비교 |
| `src/pipeline.py` | 설정 분기·단계 실행·재시작·원본 결과 보호 | 향후 실험에서도 공통 코드 재사용 |
| `src/report.py` | 저장 CSV에서 비교표·기술 통계 생성, README의 자동 요약 구간만 갱신 | 모델을 다시 돌리지 않고 실험 설명을 보존하며 결과 정리 |
| `scripts/run.sh` | 기본 점검, `--start`일 때 tmux 실행 | 밤새 실행·실수로 시작 방지 |
| `scripts/worker.sh` | 환경·GPU 잠금·실행 로그 | 동시 실행 충돌 예방 |
| `scripts/check_status.sh` | 해당 실행의 상태 읽기 | 진행 상황 확인 |
| `tests/test_protocol.py` | EOS 경계, 답변 확장, 손실·기울기 동등성 | 토큰 집계·공격 손실 오류 예방 |
| `tests/test_overnight.py` | 후보 순위·20% 경계·중복 행·중단 재개 | 사용자 지정 채택 절차 검증 |
| `tests/test_artifacts.py` | 1,440행·60 PNG·원본 해시·상세 응답 검사 | 파일 정리 중 내용 변형·누락 예방 |
| `adv/<세트 ID>/*.png` | 채택한 448×448 RGB PNG 30장씩 | 같은 공격 이미지를 모든 양자화 조건에 적용하라는 공통 원칙 |
| `adv/<세트 ID>/selection.json` | 사진별 SHA·채택 정보·입력 해시 | 세트 식별·재사용·채택 과정 추적 |
| `adv/<세트 ID>/SHA256SUMS` | PNG 파일명별 SHA-256 | 고정 입력 확인 |
| `adv/<공통 세트>/attack_log.csv` | 반복 수·시간·최선/저장 토큰 수·픽셀 제한 | 공통 공격 생성 로그 |
| `results/<실험 ID>.csv` | 모델·입력별 전체 답변 포함 720행 | **공통 지침의 기본 제출 형식**. 확장은 별도 CSV |
| `results/<실험 ID>.json` | 설정·모델 revision·패키지·census·해시 | **공통 지침의 재현성 메타데이터** |
| `results/input-manifest.json` | dev/eval clean/noise 총 70개 입력의 SHA | 데이터 동일성 확인; 원본 clean/noise는 서버에서 복원 |
| `results/<공통 ID>/attack-records.jsonl` | 원래 공격 기록 30개를 한 줄씩 보존 | 30개 JSON을 합쳐 파일 수 감소 |
| `results/<확장 ID>/responses.jsonl` | 원래 상세 응답 720개를 한 줄씩 보존 | 토큰 ID·전처리·답변 검증 기록 보존, 파일 수 감소 |
| `results/<확장 ID>/candidate-measurements.json` | 후보 750회의 반복·토큰·SHA·픽셀 변화 요약 | 25후보 탐색 결과 확인 |
| `results/<실험 ID>/provenance.json` | 원본/현재 ID 매핑, 원본 해시 | 이름 변경 후에도 출처 추적 |
| `results/<실험 ID>/status.json` | 완료 시각·진행 상태·행 수 등 원래 상태 | 실제 실행 기록 |
| `results/<공통 ID>/PROTOCOL.md` | 실행 당시 공통 실험 계획 | 공통 예시와 다른 구현 사항 설명 |
| `reports/<실험 ID>/README.md` | 실험 목적·설정·환경, 모델·입력별 측정 요약, 사진별 원본·양자화 모델 토큰 수 비교 | 실험 설명과 두 종류의 결과 표를 한 문서에서 읽기 |
| `reports/<실험 ID>/measurements-summary.json` | 위 기술 통계의 기계 판독본 | 표 재사용·후속 분석 |
| `reports/<실험 ID>/run_manifest.json` | 현재 실험 ID·이미지 세트·원본 이름·경로·설정 | 결과와 고정 PNG 세트 연결 |

공통 지침이 파일별로 이 폴더 구조나 JSONL 형식까지 강제한 것은 아니다. 제출 CSV·메타데이터·고정 입력·로그 규칙을 충족하면서 이후 실험에서 재사용하기 위해 B팀에서 정리한 구조다.

## 부품 실험 추가 파일

| 위치 | 내용 |
|---|---|
| `src/case_config.py`, `src/components.py` | C·L·VCL의 8조건 구성, 고정 공격 세트·모델·입력 확인 및 측정 실행 |
| `src/check_extended_gpu.py` | GPU 서버에서 실제 모델의 양자화·보호 층·전처리·개발 입력 확인 |
| `scripts/run_components.sh`, `component_worker.sh`, `run_component_sequence.sh` | tmux와 GPU 잠금을 사용한 부품별 실행 및 512→5,000 순차 실행 |
| `tests/test_components.py` | 부품 선택·보호 층·재개·보고서의 검사 코드. GPU 서버에서 실행 |
| `results/qwen2vl_<connector/language/vcl>_<common/extended>_components_v1_both_r01.csv` 및 `.json` | 6개 실험 각각 720행의 답변 전문과 원본 메타데이터 |
| `results/components-common-then-extended-both-r01/` | 시작 기록·최종 상태·서버 완료 확인 |
| `reports/<부품 실험 ID>/` | 실험별 README·측정 요약·run manifest |
| `reports/quantization_components_comparison_v1/` | V/C/L/VCL 비교 보고서와 집계 자료·출처·재생성 코드 |
| `experiments/` | 설정 계획·공통 규칙 대조·GPU 서버 준비·실행·완료·업로드 범위 안내 |

부품 실험의 응답별 토큰 ID·전처리 상세 파일과 전체 실행 로그는 서버에 보관한다. 기존 V의 `responses.jsonl`과 동일한 추가 파일을 이번에 내려받아 포함한 것은 아니다. 여섯 원본 CSV에는 답변 전문이 들어 있으며 수집·서버 확인 범위는 [완료 기록](../experiments/sequence_completion_20261017.md)에 명시했다.

## 실험·이미지 세트 이름의 변경 기준

이름의 기본 형식은 [B팀 README](../README.md)에 있다. 아래는 기존 기록을 덮어쓰지 않고 후속 실험을 추가하기 위한 B팀 규칙이다.

| 변경 사항 | 실험 ID | 이미지 세트 ID |
|---|---|---|
| 같은 설정·코드로 중단된 실행 재개 | 기존 `r` 유지 | 기존 `s` 유지 |
| 기존 PNG를 같은 규칙으로 다시 측정 | 다음 `r` | 기존 세트 재사용 |
| 같은 규칙으로 공격 PNG를 새로 생성해 측정 | 다음 `r` | 다음 `s` |
| 모델 revision·출력 상한·공격 예산·질문·seed·전처리·채택 규칙 변경 | 새 프로토콜 버전의 `r01` | 생성 조건이 달라지면 새 버전의 `s01` |
| 양자화 방법·비트 조건·평가 입력 종류 변경 | 평가 규칙의 새 버전으로 구분 | 생성 조건이 같으면 기존 세트 재사용 |
| 양자화 부품 변경 | 대상 이름을 바꾸고 해당 조합의 `r01` | 생성 조건이 같으면 기존 세트 재사용 |
| 설명·코드 배치만 정리하고 실험 동작 유지 | 버전 유지 가능. 이후 새 측정은 다음 `r` | PNG가 같으면 기존 세트 유지 |

`r`과 `s`는 최소 두 자리로 쓴다. 예를 들어 `qwen2vl_vision_extended_v1_r02`가 기존 `qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01`을 재사용할 수 있다.

현재 `qwen2vl`은 Qwen2-VL-2B-Instruct를 뜻한다. 다른 크기·변형의 모델은 구별되는 이름을 정하고 정확한 모델 ID·revision을 메타데이터에 남긴다. 부품 이름은 `vision`(V), `connector`(C), `language`(L), `head`(H), `full`(VCLH)로 구분한다. 별도 `components_v1` 러너는 `connector`·`language`·`vcl`을 지원하며, `vcl`은 **출력층을 제외한 V+C+L**이다. H·VCLH 양자화는 이 러너에서 허용하지 않는다. 부품 실험 이름은 `qwen2vl_<부품>_<common/extended>_components_v1_<rtn/bnb/both>_r01`이며, 조건과 행 수는 해당 실행의 메타데이터에 기록한다. 자세한 설정은 [부품 실험 안내](../experiments/README.md)에 있다.

이름의 `v`는 B팀 프로토콜 버전이다. 공동 설정의 `SETTINGS_VERSION`은 별도 메타데이터로 남기며, 공동 설정 파일 변경은 두 팀 합의가 필요하다. B팀 확장 설정은 공통 설정을 덮어쓰지 않고 별도로 관리한다.

## 실험별 보고서 관리

`reports/<실험 ID>/README.md`에는 실험 설명을 적고, 자동으로 만든 측정값 요약을 아래 두 표시 사이에 둔다. 요약 안에는 **모델·입력별 요약 → 사진별 같은 공격 이미지의 출력 토큰 수** 순서로 두 표를 넣는다. `src/report.py`는 이 구간만 갱신하므로 그 밖의 목적·설정·환경 설명은 유지된다. 표 자체를 수정하지 말고 원본 CSV와 이미지 세트의 `selection.json`에서 다시 생성한다.

```markdown
<!-- measurements:start -->
## 측정값 요약

### 모델·입력별 요약

모델·입력 종류별 기술 통계 표

### 사진별 같은 공격 이미지의 출력 토큰 수

사진별 채택 당시 원본·본 측정 원본·양자화 모델 비교표
<!-- measurements:end -->
```

표시가 중복되거나 한쪽만 있으면 오류를 내며, 표시가 없는 수동 작성 문서는 덮어쓰지 않는다. 기존 형식의 요약 전용 문서는 계속 생성할 수 있다.

## CSV 필드

공통 CSV는 **8열 순서를 그대로 유지**했다. 확장 CSV는 두 상태를 분리해 9열이다. 두 파일 모두 한 행은 `사진 1장 × 모델 조건 1개 × 입력 종류 1개`다. 설명용 열을 원본 CSV에 덧붙이지 않았다.

| 필드 | 예시 | 의미 |
|---|---|---|
| `model` | `qwen2vl` | 공통 모델 식별자 |
| `case` | `ref`, `rtn-V-4`, `bnb-V-8` | 원본 또는 V 양자화 방법·비트. `ref`는 bf16 |
| `image_id` | `000000000872` | COCO 12자리 문자열. Excel 등에서는 텍스트로 읽어 앞의 0 유지 |
| `input` | `clean`, `noise`, `adv` | 원본, 고정 무작위 노이즈, 채택 공격 PNG |
| `n_tokens` | `30`, `512`, `5000` | 생성 토큰 수. EOS가 생성되었다면 그 토큰도 포함 |
| `hit_cap` | `True` / `False` | **공통만**. EOS 없이 512토큰에 도달했는지. 성공 판정과 구분 |
| `ended_eos` | `True` / `False` | **확장만**. 생성 토큰에 두 EOS 중 하나가 포함됐는지 |
| `output_limit_reached` | `True` / `False` | **확장만**. EOS 없이 5,000토큰 상한에 도달했는지 |
| `latency_s` | `0.82` | 생성 호출의 초 단위 시간; 공격 최적화·모델 로딩 제외 |
| `text` | `A man ...` | 전체 디코딩 답변. 쉼표·줄바꿈·긴 반복도 CSV 인용 규칙에 따라 보존 |

```text
공통: model,case,image_id,input,n_tokens,hit_cap,latency_s,text
확장: model,case,image_id,input,n_tokens,ended_eos,output_limit_reached,latency_s,text
```

`n_tokens=512/5000`이어도 마지막에 EOS가 있으면 제한 도달 플래그는 False다. 두 실험의 cap이 달라 토큰 수나 플래그를 같은 조건의 결과처럼 합산하면 안 된다.

## JSON·JSONL 필드

| 파일 | 주요 필드와 의미 |
|---|---|
| 결과 `.json` | `fingerprint`: 설정 식별 해시; `config`: 이미지 ID·조건·설정·환경; `cases`: 조건별 census·EOS·dtype·peak 메모리; `completed`, `row_count`: 완료·행 수; `csv_sha256`: 원본 CSV 검증 |
| 공통 결과 `.json` 추가 | `preprocessing`: 사진별 전처리 오차; `config.environment`: model·GPU·CUDA·패키지·코드 해시; `execution_protocol`: 실제 실행 예산·후보 확인·중단·채택 규칙 |
| 확장 결과 `.json` 추가 | `inputs_sha256`: 고정 입력 해시; `config.checkpoint_iterations`, `max_new_tokens`, `tolerance`, `deterministic_algorithms`, `cublas_workspace_config`: 25후보·5,000토큰·20%·결정적 설정 |
| `selection.json` 공통 머리말 | `image_set_id`, `experiment`, `original_experiment`, `split`, `source_model`, `source_revision`, `attack_protocol`, `settings_version`: 세트 정체성과 출처 |
| `selection.json` 사진 공통 | `images[]`의 `image_id`, `path`, `sha256`, `clean_sha256`, `noise_sha256`, `iterations_run`, `linf`, `seconds`: 파일·입력·반복·최대 채널 변화·사진 처리 시간 |
| `selection.json` 공통 사진 | `start_tokens`, `best_tokens`, `saved_tokens`: 원본·최선·저장 PNG 검사 토큰 수. `iterations_run`은 실제 중단 시점이며 최선 후보의 정확한 갱신 위치를 뜻하지 않음 |
| `selection.json` 확장 사진 | `selected_iteration`, `initial_tokens`, `recheck_tokens`, `relative_difference`: 채택 후보 위치·최초/재검사 토큰·상대 차이 |
| `responses.jsonl` | 한 줄: `source_file`=원래 파일명, `record`=원본 JSON. 그 안의 `row`=CSV 행, `tokens`=실제 토큰 ID 배열, `input_sha256`=입력 해시, `preprocessing`=전처리 오차, `fingerprint`=실행 설정 식별 |
| `attack-records.jsonl` | 한 줄: `source_file`, `record`. 원본 `record`의 `clean`, `saved`에 답변·토큰 수·EOS 상태·생성 시간, `iters_run`, `seconds`, `linf`, `sha256` 등 생성 검증 기록 |
| `candidate-measurements.json` | 750개 객체의 `image_id`, `iteration`, `n_tokens`, `ended_eos`, `sha256`, `linf`: 후보 위치별 측정 요약. 후보별 전체 답변 파일은 아님 |
| `input-manifest.json` | `settings_version`, `resize`, `records[]`: 각 `image_id`, `split`, `input`, `path`, `sha256`. 경로는 기존 데이터 기준의 원본 기록 |
| `provenance.json` | 원본 ID, 현재 ID·세트 ID, 원본 메타/공격 매니페스트 SHA, 원본 CSV/JSON 바이트 동일성 |
| `run_manifest.json` | 현재 이름·프로토콜·이미지 세트·CSV/JSON 경로와 SHA. `source_archive=null`, `source_archive_storage=server_only`는 당시 코드 사본을 서버에만 보관함을 표시. `published_record=True`는 보존된 과거 실행임을 표시해 새 실행의 덮어쓰기를 차단 |

상대 오차는 `abs(재검사 토큰 − 최초 토큰) / 최초 토큰`이다. 예를 들어 5,000→4,000은 20%로 채택 가능하고 5,000→3,999는 탈락한다. 재검사는 안정성 선택 절차이며 공격 성공 평가가 아니다.

## 보존 범위

두 결과 CSV와 메타 JSON은 바이트를 보존했다. JSONL로 합친 각 `record`도 원본 객체와 같다. 실행 당시 코드·설정 사본 20개와 예비 실험 상세 파일 16개는 업로드 대상에서 제외하고 서버에 별도 보관했다. 과거 경로·모듈명·코드 해시는 원본 메타데이터에 그대로 남아 있다. 실행에는 현재 `src/`와 `scripts/`를 사용하며 공통·확장 본실험 결과는 모두 포함한다.

기존 V 자료를 처음 정리했을 때 CPU 코드 검사·저장 파일 검사·실행 전 점검을 수행했다. 이후 사용자 지시에 따라 검사는 GPU 서버에서만 수행했다. 새 부품 실험은 서버에서 검사와 4,320행 본측정을 마쳤으며 기존 V 1,440행을 재측정한 것은 아니다. 이번 공개 자료 정리에서는 파일·문서·Git 목록만 확인했다.
