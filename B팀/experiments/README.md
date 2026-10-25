# 연결부·언어모델·VCL 양자화 실험 (components_v1)

**완료(2026-10-17 12:25 KST):** tmux에서 512토큰 C→L→VCL 뒤 5,000토큰 C→L→VCL을 순차 측정해 6개 실험·4,320행을 모두 저장했다. [서버 완료 확인](sequence_completion_20261017.md), [기존 V를 포함한 비교 보고서](../reports/quantization_components_comparison_v1/README.md), [GitHub 업로드 범위](github_upload.md)를 참조한다. 실행 과정은 [실행 기록·상태 확인 방법](sequence_run_20261017.md)에 남겼다.

준비 당시에는 [5,000토큰 준비 결과·서버 실행 안내](extended_v1_ready.md)에 따라 GPU 서버에서 실제 생성·저장·중단 후 재개를 확인했다. 초기 [2026-10-17 서버 확인 결과](server_validation_20261017.md)의 22조건 확인에 이어, 최종 코드로 24조건·개발 입력 48건과 실제 5,000토큰 출력 저장을 확인했다. 본측정은 저장된 공격 PNG를 다시 만들지 않고 그대로 사용했다. 검증은 GPU 서버에서만 진행하며, 과거 로컬 코드 검사 결과를 서버 준비 완료의 근거로 사용하지 않는다. 초기 [공통 지침 대조 결과](common_rules_review.md)의 입력 검사 두 빈틈도 보강했다.

## 실험 조건

| 실험 이름 | 양자화 대상 | 대상 Linear 수 | 출력층·입력 임베딩 |
|---|---|---:|---|
| connector | 연결부 C (`model.visual.merger`) | 2 | bf16 유지, 가중치 공유 유지 |
| language | 언어모델 본체 L (`model.language_model`의 Linear) | 196 | bf16 유지, 가중치 공유 유지 |
| vcl | 비전 V + 연결부 C + 언어모델 본체 L | 326 | bf16 유지, 가중치 공유 유지 |

`vcl`은 출력층까지 포함하는 `VCLH`와 다른 조건이다. `공통/settings.py`의 기존 1·2차 조건은 변경하지 않았으며, 이 설정은 B팀의 별도 `components_v1`으로 기록한다.

- 모델: Qwen/Qwen2-VL-2B-Instruct, revision `895c3a49bc3fa70a340399125c650a463535e71c`.
- 기본 양자화는 기존 V 실험과 같은 **RTN 8·6·5·4·3비트 + bnb 8·4비트**의 7가지다. 원본 `ref`를 포함해 부품별 8조건이며 기본 backend는 `both`다.
- RTN은 group 64의 Linear weight만 바꾸며 bias·정규화·입력 패치 Conv는 유지한다. bnb는 8비트 LLM.int8와 4비트 NF4 (4비트 계산 dtype bf16)를 사용한다.
- 각 실험에 원본 bf16 `ref`를 포함한다. 각 모델은 같은 원본 체크포인트에서 새로 로딩한다.
- 양자화 직후 V/C/L/H census가 각각 128/2/196/1인지, 의도한 부품만 양자화됐는지 확인한다. H 양자화와 가중치 공유 해제를 허용하지 않는다.
- RTN 전후에는 임베딩·출력층 Parameter의 정체성·저장 위치·버전·dtype 및 공유 상태를 검사한다. bnb 로딩 후에도 두 층이 bf16이며 공유 상태인지 확인한다.
- 질문: `Describe this image in one sentence.`. greedy, seed 2026, 결정적 계산 설정. Qwen 종료 토큰 두 개를 모두 사용한다.
- COCO eval 30장 × clean·noise·adv. 원본·노이즈·공격 입력의 해시와 448×448 전처리를 검사한다. eval 일부만 골라 조건을 조정하지 않는다.
- CSV에 생성 토큰 수·생성 시간·답변 전문을 저장한다. 메타데이터에는 census·가중치 변화·보호된 층·모델 메모리 footprint·GPU peak allocation을 기록한다.
- RTN은 모의 양자화여서 bf16 저장 용량이 줄지 않는다. bnb의 메모리 측정과 구분한다. 토큰 수 요약이 정상 답변 품질이나 공격 성공률을 자동으로 평가하는 것은 아니다.

| backend | 각 실험의 조건 | 실험별 행 수 | C·L·VCL 합계 |
|---|---|---:|---:|
| `both` (기본) | ref + RTN 8·6·5·4·3 + bnb 8·4 | 720 | 2,160 |
| `rtn` (선택한 경우만) | ref + RTN 8·6·5·4·3 | 540 | 1,620 |
| `bnb` | ref + bnb 8·4 | 270 | 810 |

## 공통·확장 프로토콜

| protocol | 출력 상한 | 재사용하는 공격 세트 |
|---|---:|---|
| `common_v1` (기본) | 512 | `qwen2vl_attack_common_v1_i300_cap512_s01` |
| `extended_v1` | 5,000 | `qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01` |

공통과 확장은 입력 세트·출력 상한이 다르므로 별도로 실행하고 분석한다. 두 프로토콜의 기본 8조건 계획은 [components_v1_plans.json](components_v1_plans.json)에 저장한다. 기존 V 결과와는 해당 프로토콜의 같은 이미지·같은 양자화 방법·같은 비트 조건을 비교한다.

## 실행 전 확인

아래 명령은 모두 **저장소 루트**에서 실행한다. Python 3.12와 기존 GPU 서버 환경을 사용한다. 패키지 버전은 기존 측정과 일치해야 하며 `B팀/requirements.in`·`requirements.lock.txt`에 기록돼 있다.

GPU 서버에서 계획·공격 PNG 해시만 확인하기 (모델 로딩·추론·결과 저장 없음):

```bash
python B팀/src/components.py common_v1 --plan
python B팀/src/components.py extended_v1 --plan
```

GPU 서버에서 모델 캐시·원본 입력·환경·GPU를 검사하기 (모델 로딩·추론·결과 저장 없음):

```bash
bash B팀/scripts/run_components.sh common_v1 --check both r01
```

`ONDEVICE_AI_ROOT`의 기본값은 `$HOME/ondevice-ai`이며, 그 안의 `activate.sh`가 있으면 기존 환경을 활성화한다. `model.json`, 고정 revision의 모델 캐시, `data/manifest.json`, `data/qwen2vl/eval/{clean,noise}/`의 PNG 60장이 필요하다. 누락된 입력이 있을 때는 기존 `B팀/src/data.py`의 준비 절차를 사용한다. 실행 준비 검사는 파일을 내려받거나 덮어쓰지 않는다.

`--plan` 성공은 코드·공격 세트의 일치 확인이다. 실제 서버에서 `--check`가 통과해야 실행 환경도 준비된 상태다. `--check`는 모델을 올리지 않으므로 양자화 모듈의 실제 동작은 실행 시 census 검사로 추가 확인한다.

## 실제 측정 시작 (준비 단계에서는 실행하지 않음)

공통 설정의 세 부품 실험을 각각 8조건으로 순서대로 실행:

```bash
bash B팀/scripts/run_components.sh common_v1 --start both r01
```

이 명령은 서버 준비 검사를 먼저 통과한 뒤 tmux에서 **C → L → VCL** 순서로 실행한다. 하나의 GPU 잠금을 전체 작업 동안 유지하여 다른 기존 실험과 동시에 GPU를 사용하지 않는다. 하나가 실패하면 다음 실험으로 넘어가지 않는다.

실제 압축 bnb 조건만 별도로 실행하거나, 확장 프로토콜을 선택하는 예:

```bash
bash B팀/scripts/run_components.sh common_v1 --start bnb r01
bash B팀/scripts/run_components.sh extended_v1 --start both r01
```

`--start`는 실제 측정을 시작한다. 인자 없이 실행하면 `common_v1 --plan both r01` 계획 확인만 한다. `rtn`·`bnb`는 사용자가 명시적으로 선택했을 때에만 해당 방법만 실행하는 부분 실험이다.

## 결과·상태·재개

기본 공통 8조건 실행의 ID는 다음과 같다.

```text
qwen2vl_connector_common_components_v1_both_r01
qwen2vl_language_common_components_v1_both_r01
qwen2vl_vcl_common_components_v1_both_r01
```

각 실행은 `B팀/results/<ID>.csv`, `<ID>.json`, `<ID>/status.json`에 따로 저장한다. 확장 프로토콜은 응답별 토큰 ID와 전처리 기록도 보관한다. 실험 완료 후 `B팀/reports/<ID>/`에 실제 조건 수에 맞는 보고서를 생성한다.

```bash
bash B팀/scripts/check_status.sh qwen2vl_connector_common_components_v1_both_r01
tail -f B팀/results/components-common_v1-both-r01/pipeline.log
```

같은 코드·환경·입력에서 중단 후 이어갈 때는 같은 `--start ... r01` 명령을 다시 사용한다. 이미 기록된 행은 건너뛴다. 코드·조건이 바뀌거나 새로 재측정할 때는 `r02` 등 새 실행 번호를 쓴다. 완료된 원본 실험과 공격 PNG는 덮어쓰지 않는다.

개별 부품만 직접 검사하려면 `python B팀/src/components.py common_v1 --check --component connector`를 사용할 수 있다. 직접 `--run`하면 전경에서 측정하므로, GPU 공유 서버에서는 잠금을 제공하는 위의 Bash 실행 도우미를 사용한다.

## GPU 서버에서의 코드 검사

아래 검사는 앞으로 GPU 서버에서만 실행한다. 합성 코드 검사만으로 실제 모델의 양자화·전처리·종료 동작이 검증되는 것은 아니다. 현재 `--check`도 실제 모델을 로딩하지 않으므로 실제 모델 검증과 구분한다.

```bash
python -X utf8 -m unittest discover -s B팀/tests -p test_components.py -v
python -X utf8 -m unittest discover -s B팀/tests -p test_artifacts.py -v
```

첫 검사는 PyTorch 없이 설정·대상 선택·공유 가중치 보호·중단 후 재개·가변 행 수 보고서를 합성 데이터로 확인한다. 실제 모델의 추론 결과나 GPU 메모리를 검증한 것으로 해석하지 않는다. 두 번째는 기존 결과·공격 PNG의 해시와 자료 구조를 검사한다.
