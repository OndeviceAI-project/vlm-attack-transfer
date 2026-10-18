# Qwen2-VL 공격 전이 실험 안내

B팀은 Qwen2-VL 원본 모델에서 만든 공격 이미지가 양자화한 모델에서도 긴 답변을 유도하는지 측정한다. 이 폴더에는 실험에 재사용하는 코드, 고정한 공격 이미지, 측정 결과와 보고서를 모은다.

이 문서는 B팀 폴더의 사용·관리 안내다. **각 실험의 목적·설정·실행 환경·결과는 `reports/<실험 ID>/README.md`에 기록한다.**

## 1. 폴더 구조

```text
B팀/
├─ README.md                         # 폴더 사용법과 공통 관리 규칙
├─ requirements.in                   # 직접 사용하는 패키지 버전
├─ requirements.lock.txt             # 측정 당시 전체 패키지 목록
├─ .gitignore                        # 공유하지 않을 작업 파일 지정
├─ src/                              # 여러 실험에서 재사용하는 코드
├─ scripts/                          # 실행·상태 확인 도우미
├─ tests/                            # 코드 동작·저장 결과 검사
├─ adv/
│  └─ <이미지 세트 ID>/               # 채택 PNG·selection.json·SHA256SUMS·로그
├─ results/
│  ├─ input-manifest.json             # 공통 입력 목록과 해시
│  ├─ <실험 ID>.csv                   # 응답별 측정값과 답변 전문
│  ├─ <실험 ID>.json                  # 해당 실행의 설정·환경
│  └─ <실험 ID>/                      # 상세 응답·검증·진행 기록
└─ reports/
   ├─ FILES.md                       # 파일별 용도·CSV/JSON 필드 설명
   └─ <실험 ID>/
      ├─ README.md                   # 실험 설명·환경·측정 요약·사진별 모델 비교
      ├─ measurements-summary.json   # 요약표의 수치
      └─ run_manifest.json           # 실험·이미지 세트·결과 파일의 연결
```

`results/`의 CSV·JSON과 같은 이름의 폴더는 **한 실험의 자료 묶음**이다. CSV는 측정값, JSON은 설정, 폴더는 상세 근거를 담는다. `reports/`에는 이를 읽기 쉽게 설명한 문서와 표를 둔다.

**실험 ID**는 측정 실행을, **이미지 세트 ID**는 고정한 공격 PNG 묶음을 식별한다. 한 세트를 여러 실험에서 재사용할 수 있으며, 어떤 세트를 썼는지는 `run_manifest.json`의 `image_set_id`로 찾는다.

## 2. 파일 추가와 이름 규칙

### 이름 짓기

영문 소문자·숫자·밑줄·하이픈을 사용하고 공백을 넣지 않는다.

```text
실험: <모델>_<양자화대상>_<프로토콜>_v<버전>_r<실행번호>
예시: qwen2vl_vision_common_v1_r01

이미지 세트: <모델>_attack_<프로토콜>_v<버전>_i<반복예산>_cap<출력상한>[_tol<오차백분율>]_s<세트번호>
예시: qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01
```

- `qwen2vl`은 현재 Qwen2-VL-2B-Instruct, `vision`은 시각 인코더 양자화, `common`·`extended`는 서로 다른 실험 규칙을 뜻한다. 다른 모델·양자화 대상은 구별되는 이름을 쓴다.
- `v1`은 실험 규칙의 버전이다. 출력 상한·공격 방식·질문·전처리 등 비교 조건을 바꾸면 새 버전으로 구분한다.
- `r01`은 같은 조건의 첫 측정 실행이다. 다시 측정하면 `r02`로 올리며, 중단된 실행을 그대로 이어갈 때는 유지한다.
- `s01`은 같은 생성 규칙으로 만든 첫 이미지 세트다. 새로 생성하면 `s02`로 올린다. 재측정에서 기존 세트를 쓰면 `r`만 바뀌므로 `r`과 `s`는 같을 필요가 없다.
- `i500`은 최대 공격 반복 500회, `cap5000`은 출력 상한 5,000토큰, `tol20`은 재검사 허용 오차 20%다. `tol`은 필요한 경우에만 쓰며 대괄호는 실제 이름에 넣지 않는다.

이미지는 원본 모델에서 만들므로 세트 이름에 양자화 대상을 넣지 않는다. 이름에 담지 않은 모델 revision·seed·세부 설정은 메타데이터에 남긴다. 자세한 변경 기준은 [FILES.md](reports/FILES.md)에 있다.

### 새 실험 추가하기

1. 바꿀 조건과 공통 지침과의 차이를 정하고, 실험 ID와 사용할 이미지 세트를 정한다.
2. 기존 `src/`를 재사용하고 필요한 동작을 추가한다. 이름만 바꿔도 실행 설정이 바뀌는 것은 아니므로 실제 설정을 확인한다.
3. 새 공격 세트는 `adv/`에 따로 저장한다. PNG 이름은 COCO 12자리 ID를 유지하고, 채택 정보·해시·검증 로그를 남긴다. 같은 비교에는 같은 PNG를 사용한다.
4. `results/`에 CSV·메타 JSON·상세 기록을 저장한다. 완료된 결과와 이미지 세트는 덮어쓰지 않고, 입력 목록이 달라지면 해당 실행의 목록·해시를 따로 남긴다.
5. `reports/<실험 ID>/README.md`에 **목적과 변경점 → 설정·채택 규칙 → 실행 환경 → 결과와 해석**을 적고, `run_manifest.json`으로 사용한 결과와 이미지를 연결한다. 아래 실험 진행 현황에도 내용·상태를 한 줄 추가하고 상세 보고서가 있으면 링크한다.
6. 공유 전에는 예상 행 수·중복·누락·파일 해시를 확인한다. 측정 CSV는 손으로 고치지 않고, 새 분석은 `reports/`에 추가한다.

폴더 구성·이름은 B팀 규칙이다. 공통 값·고정 입력·양자화 대상·제출 CSV 형식은 [프로젝트 공동 지침](../README.md)을 따른다. 공동 설정 변경은 두 팀이 합의하고 버전을 올리며, 별도 확장 실험은 구분해서 기록한다.

## 3. 공통 코드

| 파일 | 역할 |
|---|---|
| [src/data.py](src/data.py) | 원본·무작위 노이즈 입력 준비와 해시 검사 |
| [src/runtime.py](src/runtime.py) | 모델 로딩, 전처리, 양자화, 토큰·생성 시간 측정 |
| [src/attack.py](src/attack.py) | 공격 이미지 갱신, 후보 확인·채택, 상태 저장 |
| [src/evaluate.py](src/evaluate.py) | 고정한 clean·noise·adv를 모델별로 측정하고 CSV 저장 |
| [src/pipeline.py](src/pipeline.py) | 설정 선택, 단계 실행, 중단 후 재개, 기존 결과 보호 |
| [src/report.py](src/report.py) | 저장된 CSV로 비교표와 요약 수치 생성 |
| [src/case_config.py](src/case_config.py) | C·L·VCL 대상·비트·예상 census·프로토콜 정의 |
| [src/components.py](src/components.py) | 부품 실험 계획·공격 세트 검증·서버 준비 검사·순차 측정 |
| [src/check_extended_gpu.py](src/check_extended_gpu.py) | GPU 서버에서 개발 입력으로 5,000토큰 생성·저장·재개 경로 확인 |

실행·상태 확인 도우미는 `scripts/`, 동작과 저장 결과 검사는 `tests/`에 있다. B팀 `src/`는 저장소의 `공통/settings.py`와 `공통/quant_rtn.py`를 사용한다.

현재 실행 코드는 기존 시각 인코더 8조건과, 출력층·임베딩을 bf16으로 유지하는 **C·L·VCL 부품 실험**을 지원한다. 부품 실험은 `components_v1`으로 구분하며 기본 조건은 V 실험과 같은 **ref + RTN 8·6·5·4·3비트 + bnb 8·4비트**의 8조건이다. 공통·확장 프로토콜 모두 고정 공격 PNG를 재사용한다.

설정·실행 준비 검사·시작·재개 방법은 [부품 실험 안내](experiments/README.md)에 있다. `src/components.py`와 `scripts/run_components.sh`는 기본적으로 계획 확인만 하며, 명시적으로 시작해야 측정한다.

## 4. 실험 진행 현황

| 실험 | 내용 | 상태 |
|---|---|---|
| [시각 인코더 · 확장 설정 v1 · r01](reports/qwen2vl_vision_extended_v1_r01/README.md) | eval 30장, 500회·5,000토큰·±20% 재검사로 이미지 채택 후 8모델 조건 × 3입력 = 720행 측정 | 완료 |
| [시각 인코더 · 공통 설정 v1 · r01](reports/qwen2vl_vision_common_v1_r01/README.md) | eval 30장, 최대 300회·512토큰·조기 종료로 이미지 채택 후 8모델 조건 × 3입력 = 720행 측정 | 완료 |
| [연결부·언어모델·VCL · components v1](experiments/README.md) | 각 부품 ref + RTN 8·6·5·4·3 + bnb 8·4의 8조건·720행, 기존 공격 PNG 재사용, 출력층·임베딩 유지. 공통·확장 프로토콜 지원 | 512·5,000 본측정 완료, V 포함 비교 요약 작성 |

시각 인코더 본실험은 측정까지 완료했으며, 공격 성공률·전이율 평가는 아직 진행하지 않았다. 부품 실험의 준비 검증은 [5,000토큰 준비 결과](experiments/extended_v1_ready.md), 이후 본측정 시작과 상태 확인 방법은 [순차 실행 기록](experiments/sequence_run_20261017.md)에 기록했다.

### 참고자료

- [V·C·L·VCL 양자화 결과 비교 — 512·5,000토큰](reports/quantization_components_comparison_v1/README.md)
- [파일·필드 상세 설명과 이름 변경 기준](reports/FILES.md)
- [프로젝트 공동 지침](../README.md)

패키지 목록은 환경 재구성의 참고 자료다. 각 실행에서 실제 사용한 GPU·패키지·모델 revision·코드 정보는 해당 실험의 메타데이터와 보고서에 남긴다. 가중치·캐시·중간 후보 전체는 공유 대상에서 제외한다.
