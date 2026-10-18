# C·L·VCL 5,000토큰 확장 실험

후속 실행: 2026-10-17 01:18 KST에 사용자 요청으로 512토큰 세 실험 완료 후 이 확장 실험으로 이어지는 tmux 작업을 시작했다. [실행 기록·현재 상태 확인](sequence_run_20261017.md)을 참조한다. 아래는 시작 전 준비 검증 기록이다.

2026-10-17 01:08 KST. 상태: **GPU 서버 배치·5,000토큰 실제 생성·저장·중단 후 재개 검증 완료. 본측정 미시작.** [서버 확인 요약 JSON](extended_v1_server_readiness.json)에 실제 관찰 결과와 코드 해시를 기록했다.

## 측정 조건

- 모델은 Qwen2-VL-2B-Instruct, 고정 revision `895c3a49bc3fa70a340399125c650a463535e71c`다.
- C → L → VCL 순서로 실행한다. 각각 ref + RTN8·6·5·4·3 + bnb8·4의 8조건이다.
- 각 조건은 원본 체크포인트에서 새로 로딩한다. Linear weight를 대상으로 하며 출력층·입력 임베딩은 bf16과 공유 상태를 유지한다.
- `max_new_tokens=5000`, greedy, seed 2026, EOS `[151645, 151643]` 전체를 사용한다. 입력 이미지·질문은 생성 토큰 상한에 포함되지 않는다.
- 기존 V 확장 실험의 고정 공격 PNG 세트 `qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01`를 재사용한다. 공격 이미지를 다시 생성하지 않는다.
- eval 30장 × clean/noise/adv × 8조건 = 실험별 720행, 세 실험 총 2,160행이다.
- 확장 CSV는 9열이며 EOS 종료와 출력 상한 도달을 따로 기록한다. 답변 전문·실제 토큰 ID·전처리 차이·해시·환경·census·GPU 메모리도 보관한다.
- 기본 실행 도우미의 기본 프로토콜은 기존대로 512토큰 `common_v1`이다. 5,000토큰은 아래처럼 `extended_v1`을 명시한다.

## 서버 배치와 실행

새 코드는 기존 V 실행·결과와 분리된 체크아웃에 배치했다. 서버의 기존 모델 캐시·원본 입력을 이용하며, 과거 V 결과 CSV·공격 PNG를 덮어쓰지 않는다.

```bash
cd /home/multimodal/ondevice-ai/experiments/vlm-components-v1-20261017

# 준비 상태 재확인: 모델 추론·본측정 없음
bash B팀/scripts/run_components.sh extended_v1 --check both r01

# 본측정 시작: 아직 실행하지 않은 명령
bash B팀/scripts/run_components.sh extended_v1 --start both r01
```

시작하면 tmux 세션 `bteam-components-extended_v1-both-r01`에서 실행한다. GPU 잠금으로 기존 실험과 동시 사용을 막고, 한 실험이 실패하면 다음 부품으로 넘어가지 않는다.

생성될 실험 ID는 다음과 같다.

```text
qwen2vl_connector_extended_components_v1_both_r01
qwen2vl_language_extended_components_v1_both_r01
qwen2vl_vcl_extended_components_v1_both_r01
```

결과는 이 체크아웃의 `B팀/results/<ID>.csv`, `<ID>.json`, `<ID>/`와 `B팀/reports/<ID>/`에 저장한다. 코드·환경·입력이 같은 중단 실행은 같은 실행 번호로 재개한다. 조건이나 코드가 바뀌면 새 실행 번호를 사용한다.

## 서버에서 수행한 준비 검사

서버에서 코드 검사 14개, 기존 자료 검사 3개, 셸 스크립트 문법 검사 5개를 통과했다. C/L/VCL 각각의 공통·확장 실행 전 검사도 통과했다. 로컬에서는 테스트나 모델 추론을 실행하지 않았다.

별도 GPU 확인은 다음 명령으로 수행한다. 개발 이미지로만 추론하며, eval 입력은 해시·픽셀·전처리 확인에만 사용한다.

```bash
source /home/multimodal/ondevice-ai/activate.sh
export ONDEVICE_AI_REPO="$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -B -u B팀/src/check_extended_gpu.py \
  --dev-adv-dir /home/multimodal/ondevice-ai/vlm-attack-transfer/B팀/adv/dev-20261014-max500 \
  --out /home/multimodal/ondevice-ai/validation/components-extended-v1-20261017-r02
```

첫 검사에서 실제 5,000토큰 생성·CSV/응답 저장 이후 완료 시각 함수의 누락이 발견돼 `evaluate.py`에서 수정했다. 수정 후에는 별도 검사 폴더 `r02`에서 다시 확인해 통과했다. 이 `r02`는 진단 기록 번호이며, 본측정의 `r01`과 별개다.

수정한 최종 코드의 서버 확인 결과는 다음과 같다.

- C/L/VCL 각각 ref 포함 8조건, 총 24조건에서 개발 이미지 `000000000632`의 clean/noise 48건을 실제로 생성했다. 모두 상한 5,000, 전체 EOS, 예상 census, 보호된 출력층·임베딩을 확인했다.
- 확장 eval 입력 90개의 전처리 최대 차이는 모두 0이었다. 실제 파일의 고정 해시·노이즈 패턴·공격 PNG의 L∞ 검사도 통과했다. eval 응답 추론은 하지 않았다.
- 실제 확장 측정 함수를 개발 이미지 한 장의 clean/noise/adv에 적용했다. 생성 길이는 각각 29/29/5,000토큰이었다. adv는 EOS 없이 상한에 도달했고 해당 플래그도 True로 저장됐다.
- clean 한 행을 실제로 저장한 뒤 의도적으로 중단하고 재개했다. 세 행이 완성됐고 첫 응답의 해시는 유지됐다. 완료 후 다시 호출해도 CSV의 해시가 같았다.
- 수정 후 코드 검사 14개를 다시 통과했다. 이전에 통과한 기존 자료 검사 3개·셸 문법 검사 5개와 함께 준비 상태를 확인했다. tmux/flock/git 실행 파일도 확인했다.
- 정식 세 실험의 CSV·실행 상태 파일은 아직 생성되지 않았다. 검사 종료 후 GPU는 메모리 16 MiB, 사용률 0%, 계산 프로세스 없음이었고 GPU 잠금도 해제됐다.

상세 서버 기록은 `/home/multimodal/ondevice-ai/validation/components-extended-v1-20261017-r02/`의 `check.log`, `readiness.json`, `dev-pipeline/`에 보관했다.

## 입력 검사 보강

초기 정적 검토에서 발견한 두 빈틈을 실행 경로에서 보강했다.

1. 실제 서버 clean/noise 해시를 B팀의 고정 input manifest와 연결하고, 공격 세트에 기록된 noise 해시도 대조한다. 지정 seed로 노이즈 픽셀을 재구성해 일치 여부를 검사한다.
2. 실제 저장된 공격 PNG와 clean의 픽셀 차이를 다시 계산해 ε≤8 및 기록된 L∞와의 일치를 확인한다.

노이즈의 범위가 ±8 안에 있어도 패턴이 다르면 중단하고, 메타데이터가 8이라고 해도 실제 공격 PNG가 9만큼 변했으면 중단하는 부정 사례를 서버 테스트에 포함했다.

## 범위

준비 확인과 개발 이미지 추론 결과를 평가 이미지 30장의 본실험 결과로 사용하지 않는다. RTN은 모의 양자화이고 bnb는 실제 저비트 저장 방식이다. ASR·전이율 계산과 답변 품질 분석은 별도 작업이며, 현재 보고서는 원시 측정과 기술 통계를 다룬다. GitHub commit/push/PR은 수행하지 않았다.
