# 공통 지침 대조 결과 — components_v1

검토일: 2026-10-17 (Asia/Seoul).

**아래 본문은 초기 정적 검토 기록이다. 이후 사용자 요청에 따라 코드 보강 전에 실제 GPU 서버 확인을 추가했다. [2026-10-17 서버 확인 결과](server_validation_20261017.md)에서 실제 모델 22조건 통과, 전처리 130개 입력의 차이 0, 실제 noise 규칙·공격 PNG의 ε 준수를 확인했다.** 로컬 테스트는 추가 실행하지 않았다. 본문의 ‘서버 확인 대기’는 초기 검토 시점을 가리키며, 서버 배치·전체 파이프라인·분석 구현 등 아직 남은 범위는 새 기록에서 구분한다.

**후속 조치(2026-10-17 01:08 KST):** 5,000토큰 준비 요청에 따라 별도 서버 체크아웃에 코드를 배치했다. 본문 문제 1·2의 고정 noise 해시/픽셀 패턴과 실제 공격 PNG의 L∞ 검사를 실행 경로에서 보강했다. 실제 GPU에서 24조건·개발 입력 48건, 실제 5,000토큰 출력의 저장·중단 후 재개·완료 처리까지 확인했다. [최신 준비 결과](extended_v1_ready.md)를 참조한다. 본측정과 ASR/전이율 분석은 미실행·미구현이며 과거 로컬 기록의 공개 전 정리도 별개다.

## 기준과 검토 범위

**공개 자료 정리 후속 기록(2026-10-18):** 본측정 6개·4,320행은 [서버 완료 확인](sequence_completion_20261017.md)을 마쳤고, 기존 V를 포함한 비교 요약도 작성했다. 과거 로컬 진단 파일 `local_validation.txt`, `preparation_checks.json`은 원본을 보관하되 `B팀/.gitignore`로 업로드 대상에서 제외했다. 아래 미완료·공개 전 정리 지적은 초기 검토 시점의 기록이다. ASR·전이율 계산은 비교 보고서에 포함하지 않았다.

GitHub API로 현재 기본 브랜치 `main`의 커밋 `3884c76bd14182e3161aac43922b6e55bf8df1f5`와 다음 자료를 직접 읽었다.

- [공통 README](https://github.com/OndeviceAI-project/vlm-attack-transfer/blob/3884c76bd14182e3161aac43922b6e55bf8df1f5/README.md)
- [공통/settings.py](https://github.com/OndeviceAI-project/vlm-attack-transfer/blob/3884c76bd14182e3161aac43922b6e55bf8df1f5/공통/settings.py)
- [공통/quant_rtn.py](https://github.com/OndeviceAI-project/vlm-attack-transfer/blob/3884c76bd14182e3161aac43922b6e55bf8df1f5/공통/quant_rtn.py)
- [공격 이미지 생성 규칙](https://github.com/OndeviceAI-project/vlm-attack-transfer/blob/3884c76bd14182e3161aac43922b6e55bf8df1f5/이미지생성/README.md)
- `공통/image_list.csv`, `.github/CODEOWNERS`, 실제 활성 ruleset `main 보호`(24831213).

로컬 작업 트리의 `README.md`, `공통/`, `이미지생성/`, `.github/`는 위 main 커밋과 Git diff상 차이가 없다. 이번 변경은 B팀 폴더에 한정돼 있다. 현재 브랜치는 `b/qwen2vl-vision-results-v1-r01`이며 main이 아니다. 이번 부품 실험 수정은 아직 작업 트리의 미커밋 변경이다.

## 결론

생성·양자화·저장 설정은 공통 규칙에 맞게 연결돼 있다. 그러나 **입력 확인의 빈틈 두 가지가 있어 ‘전부 검증 완료’라고 판정할 수 없다.** 실제 모델 로딩·양자화 census·전처리 차이 0도 GPU 서버에서 확인해야 한다.

사용자가 지정한 C·L·VCL 및 7가지 양자화 조건은 공통 파일을 변경하지 않은 별도 부품 실험이다. 공통 `CASES_PHASE2` 자체와 동일한 실험으로 표시하면 안 된다.

## 공통 필수 규칙 전체 대조

| 공통 규칙 | 코드상 대조 결과 | 근거·남은 확인 |
|---|---|---|
| 공통 설정 import·설정 버전 유지 | 일치 | `runtime.py`가 공통 settings·quant_rtn을 import한다. 실제 생성은 `settings.GEN_KWARGS`, RTN은 `settings.QUANT_GROUP`을 사용한다. `pipeline.py:32`에서 v1 고정값을 대조한다. 공통 파일은 수정하지 않았다. |
| 1. 질문·greedy·512 상한 | 공통 프로토콜 일치 | `runtime.py:61`, `runtime.py:117`에서 공통 질문과 GEN_KWARGS를 사용한다. 5,000토큰은 `extended_v1`에만 속하는 별도 확장 조건이다. |
| 1. EOS 전체·토큰 수 | 코드상 일치 | `runtime.py:105`가 generation_config의 EOS 목록을 가져온다. 생성된 토큰 전체를 세고 EOS가 있으면 상한 도달로 판정하지 않는다(`runtime.py:112`). 실제 설정·종료 동작은 서버 확인 대기. |
| 2. eval 30장·dev/extra/pilot 분리 | 코드상 일치 | `runtime.py:199`가 공통 CSV의 split으로 고른다. 부품 pipeline은 eval 전체를 사용한다. `components.py:45`에서 중복 없는 30장과 공격 세트의 ID를 맞춘다. |
| 2. Qwen 448×448·직접 리사이즈·PNG | 생성·평가 코드상 일치 | `data.py:24`에서 RGB 직접 BICUBIC 리사이즈, PNG 저장. 평가 `runtime.py:66`에서 크기를 검사한다. 서버 입력 파일은 이번에 열거나 검사하지 않았다. |
| 3. 노이즈 seed [2026, coco_id]·각 픽셀 ±8 | 생성 코드는 일치, 재사용 검사 보강 필요 | `data.py:25`의 생성식은 공통 식과 같다. 하지만 새 부품 실행 경로에서는 그 패턴을 재구성하지 않으며, 서버 manifest와 고정된 B팀 manifest의 noise 해시를 직접 연결하는 검사도 빠져 있다. 아래 문제 1 참고. |
| 4. 공격 ε=8·반복 300/최대 500·원본에서 생성 후 고정 | 재사용 구조 일치, 실행 전 확인 보강 필요 | 부품 실험은 `--measure-only`를 강제해 재생성하지 않는다(`pipeline.py:260`). 30장을 그대로 쓰므로 실패 이미지도 제외하지 않는다. 원본 모델 revision을 대조한다. 다만 실제 PNG 대 clean의 최대 픽셀 차이를 새 부품 경로에서 다시 재지는 않는다. 아래 문제 2 참고. |
| 5. 공격 경로와 평가 전처리 차이 0 | 검사 로직 있음, 실제 서버 미확인 | `runtime.py:94`가 CPU float32·GPU bf16 차이 모두 정확히 0인지 확인한다. 공통 평가는 `evaluate.py:69`에서 90개 입력에 적용한다. 이 검사는 `--check`가 아니라 실제 평가 경로에서 실행된다. |
| 6. Linear weight·공통 RTN·bnb8/4·census | 선택한 부품에 적용하도록 구현 | `runtime.py:154`에서 RTN은 공통 apply_rtn(group 64), bnb는 LLM.int8/NF4를 사용한다. 공통 census와 check_census 및 예상 층 수를 검사한다. 실제 변환 결과는 서버 확인 대기. |
| 7. 같은 실행에 bf16 ref 포함 | 코드상 일치 | 기본 C/L/VCL 각각 ref + RTN8/6/5/4/3 + bnb8/4의 8조건. 같은 공격·clean·noise 입력에 적용한다. 기본은 실험별 720행, 세 실험 2,160행이다. |
| 8. 성공 정의·유효 이미지·전이율 분모 | 수집 항목은 있음, 분석은 미구현 | 공통 CSV의 hit_cap은 EOS 없는 512 도달 여부다. 원본 사진과 공격 사진을 둘 다 저장한다. 그러나 ref-clean 상한 이미지를 제외하고 ref 성공 이미지로 전이율 분모를 만드는 계산은 아직 없다. 보고서는 상한 도달 개수를 성공률·전이율로 표시하지 않는다. |
| 9. 공통 CSV 8열·원본 그대로 저장 | 공통 경로 일치 | `runtime.py:26`의 열 순서를 그대로 사용하고 `evaluate.py:119`에서 행을 생성한다. 누락·중복·최종 행 수와 완료 CSV 해시를 검사한다. 기존 측정 CSV를 손으로 수정하지 않았다. 확장은 별도 9열이다. |
| 10. 자체 기기·자체 모델에서만 공격 이미지 사용 | 코드상 일치 | 로컬 캐시 모델을 `local_files_only=True`로 로딩하고 CUDA에서 직접 generate한다. 외부 추론 서비스로 이미지를 보내는 호출은 없다. |

## 보강이 필요한 입력 검사

### 문제 1: 노이즈 대조군을 공통 기준에 끝까지 연결하지 못함

- `components.py:78`은 selection의 clean/noise 해시를 저장소의 고정 input-manifest와 대조한다.
- 실제 서버 파일을 읽는 `evaluate.py:23`은 서버의 `ROOT/data/manifest.json`을 기준으로 검사한다.
- `evaluate.py:34`는 실제 clean 해시를 selection의 clean 해시와 연결하지만 **noise에는 같은 연결 검사가 없다.**
- `pipeline.py:50`도 최대 차이 ≤8만 확인한다. 따라서 서버 manifest와 noise 입력이 함께 다른 것으로 준비돼 있으면, 지정 seed의 노이즈인지 확인하지 못한 채 통과할 수 있다.

실제 서버에 잘못된 파일이 있다는 뜻은 아니다. 새 실행의 사전 확인으로 공통 규칙을 보장하기에 부족한 지점이다. 서버 파일의 clean/noise 해시를 저장소의 고정 manifest 및 selection과 모두 대조하고, 지정 RNG로 재구성한 noise 픽셀과도 같음을 확인해야 한다.

### 문제 2: 공격 PNG의 실제 ε 재확인이 새 부품 경로에 없음

- `components.py:68`, `evaluate.py:36`은 저장된 `linf` 숫자를 읽어 ≤8인지 검사한다.
- 공격 생성 코드에는 실제 저장 PNG와 clean의 픽셀 차이를 재는 검사가 있다(`attack.py:88`). 기존 V 실험의 `verify_finished`에도 있다(`pipeline.py:60`).
- 그러나 새 부품 `--measure-only` 경로는 그 실제 픽셀 비교를 호출하지 않는다.

현재 고정 PNG가 ε를 넘었다는 판정은 아니다. 기존 검증 기록과 해시를 재사용하고 있지만, GPU 서버에서 준수 여부를 재확인하려면 현재 사용할 30쌍에 대해 실제 최대 차이를 다시 계산해 기록하는 것이 필요하다. 같은 원본 모델에서의 답 길이는 각 실행의 ref로 다시 측정한다.

## 공통안과 구분해야 하는 실험 범위

| 구분 | 공통 저장소의 정의 | 이번 사용자가 지정한 설정 | 판정 |
|---|---|---|---|
| 부품 범위 | 1차 V; 2차 C·L·H·VCLH | C·L·VCL, H·임베딩 유지 | 사용자 지정 확장. VCL을 VCLH나 모든 파라미터의 전체 양자화라고 표시하지 않는다. |
| 부품별 비트·방법 | CASES_PHASE2는 RTN 8·4·3 | 기존 V와 같은 RTN 8·6·5·4·3 + bnb 8·4 | 사용자 지정 확장. 공통 파일을 바꾸지 않고 components_v1 메타데이터에 따로 기록한다. |
| 출력 상한·스키마 | 512, 8열 | 기본 common_v1은 동일. 선택하는 extended_v1은 5,000, 9열 | 확장 결과를 공통 결과와 같은 조건으로 합치지 않는다. |

## GPU 서버에서 아직 확인하지 않은 항목

현재 `pipeline.py:288`의 `--check`는 데이터·설정과 CUDA/bf16 지원을 확인한 뒤 종료한다. **실제 모델을 불러오거나 7가지 양자화를 적용하지 않는다.** 따라서 이를 실제 GPU 양자화 검증 완료로 해석하면 안 된다.

서버에서 원본 revision, 7가지 양자화의 실제 대상 층 수, 출력층·임베딩의 bf16/공유 유지, 전처리 차이 0, EOS·출력 상한 동작을 확인해야 한다. 코드 조정이나 점검용 입력은 공통 지침대로 dev를 사용하며 eval로 조건을 조정하지 않는다. 이번 정적 검토에서는 이 항목들을 실행하지 않았다.

과거 `local_validation.txt`와 `preparation_checks.json`의 검사 수는 로컬 합성 코드·자료 검사 이력이다. GPU 모델 검증 결과가 아니다. 해당 파일에도 actual_model_loaded=false, inference_started=false, gpu_preflight_passed=false로 기록돼 있다. 이후 검증은 GPU 서버에서만 수행한다.

## Git·공유 규칙

- 실제 활성 ruleset [main 보호](https://github.com/OndeviceAI-project/vlm-attack-transfer/rules/24831213)를 읽었다. 기본 브랜치에 PR, 승인 1명, CODEOWNER 리뷰가 요구되며 삭제·non-fast-forward가 차단돼 있다.
- `.github/CODEOWNERS`의 B팀 담당자는 yongsansam·yunheeseong1·20223101이다.
- 현재 main에 직접 커밋·push·병합하지 않았다. 부품 변경은 B팀 작업 트리에 남아 있으며 PR/승인은 아직 진행하지 않았다.
- 커밋할 때 팀 표시 `[B]`가 필요하다. 이번 검토에서 커밋은 만들지 않았다.
- 새 파일 목록에는 모델 가중치 파일이 없다. 기존 공격 PNG·측정 CSV는 변경하지 않았다.
- **초기 검토의 공유 전 정리 지적:** `local_validation.txt`, `preparation_checks.json`에는 로컬 사용자 디렉터리를 포함한 절대 경로가 있었고 당시 Git ignore 대상이 아니었다. 2026-10-18 공개 자료 정리에서 두 파일을 ignore 대상으로 추가했다. 과거 기록 자체는 수정하지 않았다.

이번 검토에서 실행한 것은 GitHub 문서 읽기와 파일·Git diff 대조뿐이다. 로컬 테스트, 모델 로딩, GPU 추론, 본실험 시작은 하지 않았다.
