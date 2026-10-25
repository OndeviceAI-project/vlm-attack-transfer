# 512 → 5,000토큰 순차 실험 완료 확인

확인 시각: 2026-10-17 15:00 KST. 실제 GPU 서버에서 완료 상태와 결과 파일을 읽어 확인했다. 로컬 모델 테스트·재측정은 실행하지 않았다.

**전체 4,320행이 정상 완료됐고 종료 코드는 0이다.**

이후 기존 V 결과 1,440행을 포함한 [512·5,000토큰 V/C/L/VCL 비교 요약](../reports/quantization_components_comparison_v1/README.md)을 작성했다. 원시 측정 CSV는 수정하지 않았다.

| 출력 상한 | 연결부 C | 언어모델 L | VCL | 합계 |
|---|---:|---:|---:|---:|
| 512 | 720/720 | 720/720 | 720/720 | 2,160/2,160 |
| 5,000 | 720/720 | 720/720 | 720/720 | 2,160/2,160 |
| 전체 | 1,440 | 1,440 | 1,440 | 4,320/4,320 |

- 시작: 2026-10-17 01:18:04 KST.
- 512토큰 세 실험 완료: 02:25:58 KST.
- 전체 측정·보고서 생성 완료: 12:25:36 KST.
- 전체 소요: 약 11시간 7분 33초.

각 실험은 ref + RTN8·6·5·4·3 + bnb8·4의 8조건이다. 조건별 90행, 실험별 720행을 확인했다.

## 결과 무결성 확인

여섯 실험 모두 다음 확인을 통과했다.

- 고정 평가 이미지 30장 × 모델 조건 8개 × clean/noise/adv의 정확한 조합. 누락·중복 없음.
- CSV SHA256가 완료 메타데이터 및 보고서 manifest와 일치.
- 메타데이터 파일 SHA256도 보고서 manifest와 일치.
- 토큰 수가 해당 상한 이내이고 생성 시간이 음수가 아니며 EOS·상한 플래그에 모순 없음.
- 조건별 실제 양자화 census가 예상 V/C/L/H 개수 및 양자화 대상과 일치.
- 입력 임베딩·출력층의 bf16 및 공유 상태 유지.
- 완료 보고서와 run manifest 존재, 완료 상태 일치.

GPU는 확인 시 계산 프로세스 없이 사용률 0%, 메모리 16 MiB인 유휴 상태였다. tmux 창은 작업 종료 후 남도록 설정한 상태이며 pane_dead=1은 정상 완료와 일치한다.

## 서버 기록 위치

체크아웃: `/home/multimodal/ondevice-ai/experiments/vlm-components-v1-20261017`

- 측정 CSV·메타데이터: `B팀/results/qwen2vl_<connector/language/vcl>_<common/extended>_components_v1_both_r01.csv` 및 `.json`.
- 보고서: `B팀/reports/<실험 ID>/README.md`.
- 전체 실행 상태·검증: `B팀/results/components-common-then-extended-both-r01/`의 `status.json`, `exit.code`, `completion_check.json`.

원본 측정 CSV와 기존 공격 PNG는 이번 확인에서 수정하지 않았다. 이 기록은 완료·자료 무결성 확인이며 ASR·전이율 분석 결과는 아니다.

## 접속 복구

이 PC의 Tailscale 서비스는 실행 중이었지만 내부 상태가 NoState였다. 설치된 Tailscale 앱을 실행한 후 Running·SelfOnline=true로 전환되고 서버도 온라인으로 확인돼 SSH 접속을 복구했다. 서버 재부팅이나 실험 설정 변경은 하지 않았다.
