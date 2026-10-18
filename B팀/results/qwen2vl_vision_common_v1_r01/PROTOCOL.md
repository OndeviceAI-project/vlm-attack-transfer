# B팀 공식 공통 설정 v1 실험

- 저장소 main 기준 커밋: 3884c76bd14182e3161aac43922b6e55bf8df1f5.
- 공통/settings.py v1은 수정하지 않고 import. Qwen2-VL-2B-Instruct, eval 30장, 448×448 RGB PNG.
- 최대 새 출력 512토큰, greedy, Qwen EOS 두 개 모두 사용.
- 공격 예산 300회(공통 최대 허용 500회와 구분), EPS=8/255, STEP=1/255, target=128, refresh=20, w_unc=.05, w_div=0.
- 공통 코드의 0부터 시작하는 반복문 검사 위치 유지: 실제 갱신 1,21,…,281회 및 마지막300회 검사. 512토큰을 EOS 없이 출력한 검사가 연속2회이면 조기종료.
- 이미지생성 README의 권장사항대로 관측된 출력 길이가 가장 긴 후보를 보존(상한 도달 여부 우선, 동률은 먼저 관측한 후보). 실패한 사진도 남김. clean이512이면 제외 대상 여부를 나중에 판단할 수 있도록 원본·로그 보존.
- 저장한 PNG를 다시 읽어 픽셀변화 상한, 공식 프로세서와 CPU float32·GPU bf16 전처리 차이0, 저장본 출력 토큰 수 일치를 확인.
- 기존에 확인한 결정적 계산 설정 적용: seed2026, deterministic_algorithms=True, cudnn.deterministic=True, benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8.
- 새로 고정한 공격 PNG30장을 ref, RTN V8/6/5/4/3, bnb V8/4에서 공통 사용. 비전 Linear만 양자화, census 확인.
- CSV: model,case,image_id,input,n_tokens,hit_cap,latency_s,text. 8조건×30장×clean/noise/adv=720행. image_id12자리, EOS도 토큰 수에 포함. hit_cap은 EOS없이512토큰 도달 여부.
- 성공률·전이율 분석 및 GitHub 업로드는 수행하지 않음.
- 기존500회·5000토큰 실험의 파일은 재사용·변경하지 않음. 기존clean/noise60개와 캐시 모델만 재사용.

공격 이미지: B팀/adv/eval-official-v1-i300-cap512-20261015/
CSV: B팀/results/eval-official-v1-i300-cap512-20261015.csv
메타데이터: 위 CSV와 같은 이름의 .json
실행기록·설정·실행소스: B팀/results/eval-official-v1-i300-cap512-20261015/

실행: bash run_official_v1.sh
확인: bash check_official_v1.sh
중단 후 같은 설정으로 재실행하면 완료한 이미지는 SHA-256을 확인한 뒤 건너뛰고 CSV는 완료 행을 이어서 기록한다. 계산 중이던 한 이미지는 처음부터 다시 계산한다.


이 파일은 실행 당시 프로토콜 기록이다. 예전 실행 ID와 원격 절대 경로는 출처 확인을 위해 보존했다. 현재 이름 매핑은 `../../reports/qwen2vl_vision_common_v1_r01/run_manifest.json`을 참고한다.
