# 업로드 파일 안내와 필드 미리보기

이 문서에서 `<실험 ID>`는 `qwen2vl_vision_common_v1_r01` 또는 `qwen2vl_vision_extended_v1_r01`이다. 파일명은 현재 이름이며, 원본 메타데이터 내부의 이전 이름은 보존했다.

## 파일별 목적과 포함 근거

| 위치 | 내용 | 포함 이유 |
|---|---|---|
| `README.md` | 실험 구분, 폴더 안내, 서버 실행법, 해석 범위 | B팀 설명·다음 실행을 위한 문서 |
| `requirements.in` | 직접 사용하는 패키지의 고정 버전 | 환경 재구성 |
| `requirements.lock.txt` | 측정 서버 패키지 버전 전체 | 당시 환경 추적 |
| `.gitignore` | 가중치·체크포인트·후보 작업물 제외 | 실험 산출물과 대용량 작업 파일 구분 |
| `src/runtime.py` | 로컬 모델 캐시 로딩, 전처리, EOS, V 양자화, 생성 시간 | 공통 설정의 입력·EOS·대상 부품·시간 규칙 |
| `src/data.py` | 이미지 준비·해시 검사, 명시적 다운로드 옵션 | 공통 이미지 목록·448 PNG·고정 noise 재현 |
| `src/attack.py` | 공통 최선 후보 저장, 확장 25후보·20% 재검사, 상태 저장 | 실제 사용한 두 공격 방식의 실행 구현 |
| `src/evaluate.py` | 고정 clean/noise/adv 측정, CSV 저장, 기존 세트 재사용 | 공통 결과 형식과 쌍 비교 |
| `src/pipeline.py` | 설정 분기·단계 실행·재시작·원본 결과 보호 | 향후 실험에서도 공통 코드 재사용 |
| `src/report.py` | 저장 CSV에서 사진별 비교표와 기술 통계 생성 | 모델을 다시 돌리지 않고 결과 정리 |
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
| `reports/<실험 ID>/README.md` | 조건별 clean/noise/adv 기술 통계 | 측정 결과 빠르게 읽기 |
| `reports/<실험 ID>/image-model-tokens.md` | 세로 사진, 가로 채택 당시 원본·후속 원본·양자화 모델 토큰 수 | 사용자가 요청한 같은 사진 비교 |
| `reports/<실험 ID>/measurements-summary.json` | 위 기술 통계의 기계 판독본 | 표 재사용·후속 분석 |
| `reports/<실험 ID>/run_manifest.json` | 현재 실험 ID·이미지 세트·원본 이름·경로·설정 | 결과와 고정 PNG 세트 연결 |
| `reports/experiment_history.md` | 예비 실험·재현성 점검·본실험 진행 과정 요약 | 과정을 설명하되 상세 예비 기록은 서버에 보관 |

공통 지침이 파일별로 이 폴더 구조나 JSONL 형식까지 강제한 것은 아니다. 제출 CSV·메타데이터·고정 입력·로그 규칙을 충족하면서 이후 실험에서 재사용하기 위해 B팀에서 정리한 구조다.

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

현재 코드 정리 후 CPU 검증·저장 파일 검사·실행 전 점검을 수행했다. 정리한 코드로 GPU 전체 실험을 다시 돌려 동일 결과를 얻었다고 주장하지 않는다.
