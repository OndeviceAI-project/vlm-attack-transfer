# components_v1 GitHub 업로드 범위

정리일: 2026-10-18 (Asia/Seoul). 이 문서는 업로드할 변경의 범위를 설명하며 커밋·push·PR 생성 완료를 뜻하지 않는다.

## 포함할 자료

변경 범위는 `B팀/` 안이다. 공통 설정·루트 안내·`.github/`는 변경하지 않았다.

| 위치 | 이번에 추가·갱신하는 내용 |
|---|---|
| `src/`, `scripts/`, `tests/` | C·L·VCL에 ref + RTN8·6·5·4·3 + bnb8·4를 적용하는 실행 코드, 입력·보호 층 검사, 512→5,000 순차 tmux 실행 및 GPU 서버에서 사용한 검사 코드 |
| `results/qwen2vl_<connector/language/vcl>_<common/extended>_components_v1_both_r01.csv` 및 `.json` | 6개 실험의 원본 측정 CSV와 환경·설정·완료 메타데이터. 각 720행, 총 4,320행 |
| `results/components-common-then-extended-both-r01/` | 시작 기록, 최종 상태, 서버 완료·무결성 확인 결과 |
| `reports/qwen2vl_<connector/language/vcl>_<common/extended>_components_v1_both_r01/` | 실험별 설명·측정 요약·원본 연결 manifest |
| `reports/quantization_components_comparison_v1/` | 기존 V와 새 C·L·VCL 총 5,760행 비교, 집계 CSV·JSON·출처 해시 및 서버에서 실행한 `build.py` |
| `experiments/`, B팀 안내 문서 | 실험 계획, 공통 규칙 검토, GPU 서버 준비·검사·실행·완료 기록과 파일 안내 |

기존 V 원본 결과와 공격 PNG는 재사용하며 수정하지 않았다. CSV·메타데이터·고정 입력의 원본 내용도 유지한다. 부품 실험의 `vcl`은 출력층·입력 임베딩을 bf16으로 유지하는 V+C+L이다.

## 제외할 자료

- 모델 가중치·체크포인트·캐시·작업 후보·임시 로그는 기존 ignore 규칙을 따른다.
- 과거 로컬 진단 `experiments/local_validation.txt`, `experiments/preparation_checks.json`은 로컬 경로가 있어 ignore에 추가했다. 파일 원본은 로컬에 보관한다.
- 부품 실험의 응답별 토큰 ID·전처리 상세 파일, 실행 로그·종료 코드 파일과 준비 검사의 원본 상세 로그는 서버에 보관한다. 제출 CSV에는 답변 전문이 있으며 메타데이터·완료 확인·비교 보고서를 함께 포함한다.

## 확인 근거와 검토 시 유의점

실제 모델 실행과 수치 집계·결과 확인은 GPU 서버에서 수행했다. 공개 자료 정리에서는 파일 목록·문서·Git 변경 범위만 살폈고 로컬 테스트·모델 실행은 하지 않았다. 준비·실행 중의 상태 기록은 당시 이력이며 최종 상태는 [완료 확인](sequence_completion_20261017.md)을 따른다.

512와 5,000 결과는 고정 공격 PNG 세트도 달라 별도 프로토콜로 비교한다. 보고서의 상한 도달 수를 ASR·전이율로 표시하지 않으며 의미적 답변 품질을 평가했다고 주장하지 않는다. RTN은 bf16 저장의 모의 양자화이고 bnb는 실제 압축 방식이다.

커밋할 때 저장소 규칙의 `[B]` 표시를 사용하고, main 반영은 PR과 CODEOWNER 승인을 거친다.
