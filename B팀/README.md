# B팀 — Qwen2-VL 시각 인코더 양자화 실험

Qwen2-VL-2B-Instruct 원본 bf16 모델에서 만든 공격 PNG를 고정한 뒤, 같은 사진의 clean·noise·adv 입력을 원본과 시각 인코더(V) 양자화 모델 7종에 각각 넣은 **측정 기록과 실행 코드**다. 각 실험은 30장 × 8조건 × 3입력 = 720행이다. 성공·실패, ASR, 전이율은 아직 평가하지 않았다.

## 실험과 이미지 세트

| 구분 | 공통 설정 | 확장 설정 |
|---|---|---|
| 실험 ID | `qwen2vl_vision_common_v1_r01` | `qwen2vl_vision_extended_v1_r01` |
| 이미지 세트 ID | `qwen2vl_attack_common_v1_i300_cap512_s01` | `qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01` |
| 출력 상한 | 512토큰 | 5,000토큰 |
| 공격 갱신 | 최대 300회 | 500회 모두 수행 |
| 후보 출력 확인 | 1, 21, …, 281회 및 마지막 300회 | 20, 40, …, 500회, 총 25회 |
| 중단·채택 | EOS 없이 512토큰인 확인이 두 번 연속이면 중단. 관찰 후보 중 `(상한 도달, 토큰 수)`가 최대인 이미지를 저장 | 최초 출력 토큰 수가 많은 후보부터 재검사하고, 최초 대비 오차가 20% 이하인 첫 후보를 채택 |
| 저장 확인 | PNG 재로딩 후 채택 후보와 토큰 수·상한 도달 여부 일치 확인 | 재검사 입력과 최종 PNG의 SHA-256 일치 확인 |
| CSV | 공통 8열 | EOS·출력 제한을 분리한 9열 |

`vision`은 이후 **시각 인코더 Linear만 양자화**했다는 뜻이다. 공격 생성은 원본 bf16에서 했으므로 이미지 세트 이름에는 양자화 대상을 넣지 않았다. 같은 세트를 다른 부품 양자화 실험에서도 재사용할 수 있다. `i300/i500`은 공격 반복 예산, `cap`은 출력 상한, `tol20`은 재검사 오차 20%, `s01`은 첫 이미지 세트다.

`r01`은 해당 실험의 첫 독립 실행 번호다. 중단 후 이어 실행하면 같은 번호를 사용한다. 새 독립 실행은 `r02`, 과학적 설정 변경은 새 프로토콜 버전으로 구분한다. 버전이 같은 경우 코드 정리와 실험 설정 변경을 혼동하지 않도록 실행 때 코드 해시를 함께 남긴다.

공통 설정은 저장소의 `공통/settings.py` v1, 이미지 목록, CSV 8열, V 전용 양자화 조건을 따른다. 다만 예시 공격 코드의 **500토큰 이상 중단·마지막 상태 저장**과 달리, B팀 실행은 **EOS 없는 정확한 512토큰 중단·최선 후보 저장**을 사용했다. 예시 구현과 완전히 같은 실험이라고 해석하면 안 된다. 실행 당시 세부 규칙은 `results/<실험 ID>/PROTOCOL.md`, 현재 이름과 원본 기록의 연결은 `reports/<실험 ID>/run_manifest.json`에 있다. 확장 설정은 팀 내 추가 측정이며 공통 설정 결과와 합쳐 하나의 지표로 계산하지 않는다.

## 폴더 안내

```text
B팀/
├─ README.md
├─ requirements.in / requirements.lock.txt
├─ .gitignore
├─ src/       runtime.py, data.py, attack.py, evaluate.py, pipeline.py, report.py
├─ scripts/   run.sh, worker.sh, check_status.sh
├─ tests/     CPU 검사와 저장 결과 무결성 검사
├─ adv/
│  ├─ qwen2vl_attack_common_v1_i300_cap512_s01/
│  └─ qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01/
├─ results/
│  ├─ qwen2vl_vision_common_v1_r01.csv / .json
│  ├─ qwen2vl_vision_extended_v1_r01.csv / .json
│  ├─ input-manifest.json
│  └─ <실험 ID>/  상세 기록, provenance
└─ reports/
   ├─ qwen2vl_vision_common_v1_r01/
   ├─ qwen2vl_vision_extended_v1_r01/
   ├─ experiment_history.md
   └─ FILES.md
```

- `src/`: 공통·확장 설정을 실행하는 실제 코드다. 코드 복사본을 실험마다 만들지 않고 공통 처리와 프로토콜 분기를 재사용한다.
- `scripts/`: 환경을 읽고 tmux에서 명시적으로 시작하며, GPU 동시 사용을 잠금으로 막는다. 기본 동작은 점검이다.
- `adv/<세트 ID>/`: 최종 PNG 30장, `selection.json`, `SHA256SUMS`. 공통 세트에는 `attack_log.csv`도 있다. 짧게 출력한 사진도 제거하지 않았다.
- `results/`: CSV는 측정값, 같은 이름의 JSON은 실행 환경·설정·해시다. 실행 당시 JSON 내부의 예전 ID와 절대 경로는 원본 출처를 보존하려고 유지했다.
- `reports/`: 사진별 모델 비교표, 기술 통계, 이름 매핑과 실험 진행 요약이다. 예비 실험 상세 파일은 서버에 별도 보관한다.

각 파일의 내용·필드·포함 이유는 [FILES.md](reports/FILES.md)를 참고한다.

## 원격 실험 서버에서 실행

Python 3.12, CUDA를 지원하는 PyTorch, bf16 GPU와 기존 모델 캐시가 필요하다. 측정 환경의 정확한 패키지 버전은 `requirements.lock.txt`에 있다. 모델은 `Qwen/Qwen2-VL-2B-Instruct`, revision `895c3a49bc3fa70a340399125c650a463535e71c`이다. 가중치와 데이터 캐시는 Git에 포함하지 않는다.

- 저장소를 포함한 경로: `ONDEVICE_AI_REPO` (기본: `B팀/`의 상위 폴더).
- 모델·데이터 경로: `ONDEVICE_AI_ROOT` (기본: 서버의 `$HOME/ondevice-ai`).
- `$ONDEVICE_AI_ROOT/model.json`은 `model_id`, `revision`, `snapshot_path`를 가진다. `snapshot_path`에는 해당 revision의 로컬 모델 캐시를 지정한다.
- 입력 파일은 `$ONDEVICE_AI_ROOT/data/qwen2vl/{dev,eval}/{clean,noise}/<12자리 ID>.png`, 입력 해시는 `data/manifest.json`에서 읽는다.
- `src/data.py`는 기본적으로 기존 입력을 검사하기만 한다. `--download`를 명시하면 원격 서버에서 COCO 원본을 받아 준비하며, 기존의 다른 해시 파일은 덮어쓰지 않는다. 학습·가중치 다운로드는 이 코드가 자동 실행하지 않는다.

아래 명령은 **서버의 B팀 폴더에서** 사용한다. `r01`과 `s01`은 이미 완료한 자료이므로 새 실험에는 새 이름을 쓴다.

```bash
# 검사만 수행 — 모델 추론·공격·tmux 시작 없음
bash scripts/run.sh common_v1 qwen2vl_vision_common_v1_r02 qwen2vl_attack_common_v1_i300_cap512_s02 --check
bash scripts/run.sh extended_v1 qwen2vl_vision_extended_v1_r02 qwen2vl_attack_extended_v1_i500_cap5000_tol20_s02 --check

# 별도 실행을 결정한 뒤에만 사용 — 이미지 생성부터 720행 측정까지 tmux 실행
bash scripts/run.sh common_v1 qwen2vl_vision_common_v1_r02 qwen2vl_attack_common_v1_i300_cap512_s02 --start
bash scripts/run.sh extended_v1 qwen2vl_vision_extended_v1_r02 qwen2vl_attack_extended_v1_i500_cap5000_tol20_s02 --start

# 이미 공개용으로 고정한 PNG를 다시 측정 — 새 결과 이름, 기존 이미지 세트
bash scripts/run.sh extended_v1 qwen2vl_vision_extended_v1_r02 qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01 --check --measure-only
# 위 검사를 확인하고 --check를 --start로 바꾸면 재측정 실행

bash scripts/check_status.sh qwen2vl_vision_extended_v1_r02
PYTHONPATH=src python -m unittest discover -s tests -v
python src/report.py qwen2vl_vision_extended_v1_r01
```

확장 설정은 20회마다 delta·target·CPU/CUDA RNG·모델 RoPE 상태를 저장해 해당 체크포인트부터 이어 간다. 공통 설정은 완료 PNG 단위로 이어 간다. 같은 실행 이름으로 다른 코드·설정을 이어 쓰지 않는다. 원격 프로세스가 끊겨도 tmux의 실험은 계속되며, 중간 기록은 `results/<새 실행 ID>/status.json`, `pipeline.log`에서 확인한다.

## 입력·측정 해석

공통 이미지 목록의 dev 5장·eval 30장을 사용했다. RGB 448×448 직접 BICUBIC 리사이즈 후, noise는 seed `[2026, image_id]`로 각 채널에 ±8을 더하고 0…255로 제한했다. 공격도 채널별 변화량 최대 8, 즉 정규화 기준 `8/255`로 제한했다. 프로세서 전처리는 미분 가능한 경로와 일치하는지 확인했다.

프롬프트는 `Describe this image in one sentence.`이며 greedy 생성이다. 토큰 수에는 생성된 EOS가 있으면 포함한다. 두 EOS ID는 `151645`, `151643`이다. `latency_s`는 CUDA 동기화를 포함한 **생성 호출 시간**이며 모델 로딩·공격 이미지 최적화 시간은 제외한다. 입력별 상세 지연시간을 곧바로 토큰당 속도 우열로 해석하지 않는다.

RTN은 group 64의 양자화 값을 부동소수점 가중치로 되돌려 사용하는 모의 양자화다. bnb는 실제 8/4비트 모듈을 사용하며 4비트는 NF4다. 대상 부품을 census로 확인했다. 이 결과는 답변의 정확성, 실제 메모리 절감이나 공격 성공률을 증명하지 않는다.

## 보존과 업로드 범위

CSV 2개와 원본 메타 JSON 2개는 실행 원본과 바이트 단위로 같다. 확장 상세 응답 720개는 `responses.jsonl` 한 파일로 묶었으며 `source_file`에 예전 파일명을 남겼다. 공통 공격 기록 30개도 `attack-records.jsonl`로 묶었다. JSONL은 **한 줄에 JSON 객체 하나**인 보조 기록이며 CSV를 대체하지 않는다.

750개 후보 PNG, GPU 체크포인트, 모델 가중치, 캐시, 전체 중간 후보 답변 중복본은 원격 서버에 보존하고 업로드 대상에서 제외했다. 후보 750회의 토큰 수·반복 위치 요약은 `candidate-measurements.json`에 남겼다. 실험 진행 과정과 재현성 점검 요약은 `reports/experiment_history.md`에 남겼다.

실제 실행 당시 코드·설정 사본 20개와 예비 실험 상세 자료 16개는 업로드 대상에서 제외하고 원격 서버에 별도 보관했다. 현재 `src/`에는 공통·확장 설정의 실행 코드가 모두 남아 있다. 원본 메타데이터의 코드 해시와 과거 경로는 그대로 보존했다.

이 폴더는 본실험 코드·이미지·결과·보고서 104개 파일로 구성한다. 파일 정리 과정에서는 새 GPU 실험을 수행하지 않았다.
