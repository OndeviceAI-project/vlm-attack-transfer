# 512 → 5,000토큰 순차 실행 기록

**완료 확인:** 2026-10-17 12:25 KST에 전체 4,320행과 보고서 생성이 정상 종료됐다. 15:00 KST에 서버에서 누락·중복·CSV 및 메타데이터 해시·census를 확인했다. [완료 확인 기록](sequence_completion_20261017.md)을 따른다. 아래는 시작 당시 기록이다.

실행 시작: **2026-10-17 01:18 KST**. 사용자 요청에 따라 GPU 서버의 tmux에서 본측정을 시작했다. 첫 확인에서 512토큰 연결부 측정 CSV의 행 누적과 실제 GPU 추론 프로세스를 확인했다. 이 문서는 시작 기록이며 이후 진행률은 서버에서 읽는다.

## 실행 순서와 조건

1. common_v1 (상한 512): C → L → VCL
2. 앞의 세 측정·보고서가 모두 성공하면 extended_v1 (상한 5,000): C → L → VCL

각 부품은 ref + RTN8·6·5·4·3 + bnb8·4의 8조건, eval 30장 × clean/noise/adv의 720행이다. 프로토콜별 2,160행, 전체 예상 4,320행이다. 각 프로토콜에 맞는 기존 V 공격 PNG를 고정 재사용한다.

전체 순서를 하나의 GPU 잠금으로 보호한다. 오류가 나면 다음 단계로 넘어가지 않고 해당 단계·종료 코드·로그를 남긴다. SSH 접속을 종료해도 tmux 작업은 계속 실행된다.

## 서버 경로와 상태 확인

- 체크아웃: /home/multimodal/ondevice-ai/experiments/vlm-components-v1-20261017
- tmux: bteam-components-512-then-5000-both-r01
- 전체 상태: B팀/results/components-common-then-extended-both-r01/status.json
- 전체 로그: 같은 폴더의 sequence.log
- 시작 기록: 같은 폴더의 launch.json
- 종료 코드: 작업 종료 후 같은 폴더의 exit.code
- 프로토콜별 로그: B팀/results/components-common_v1-both-r01/pipeline.log 및 components-extended_v1-both-r01/pipeline.log

아래 명령은 읽기만 하며 새 측정을 시작하지 않는다.

~~~bash
cd /home/multimodal/ondevice-ai/experiments/vlm-components-v1-20261017
bash B팀/scripts/run_component_sequence.sh --status r01
tmux capture-pane -p -t bteam-components-512-then-5000-both-r01 -S -30
~~~

필요하면 tmux에 직접 연결한다.

~~~bash
tmux attach -t bteam-components-512-then-5000-both-r01
~~~

## 다음 확인 시 볼 항목

사용자가 다음 날 확인을 요청하면 서버에 접속해 전체 상태, 여섯 실험별 완료 행 수·오류, tmux 생존 상태와 GPU 프로세스를 확인한다. 완료됐다면 실험별 720행·중복/누락·CSV 해시·완료 메타데이터·보고서를 확인한다. 진행 중이면 현재 부품·프로토콜과 저장된 행 수를 보고한다.

이 요청은 사용자가 확인을 요청했을 때의 점검이다. 별도 예약 알림은 만들지 않았다. 원본 측정 CSV를 손으로 수정하거나 진행 중인 코드를 바꾸지 않는다.

시작에 사용한 명령:

~~~bash
bash B팀/scripts/run_component_sequence.sh --start r01
~~~

같은 실행을 다시 시작하려면 먼저 기존 tmux·GPU·실행 상태를 확인해야 한다. 실행 도우미는 같은 이름의 tmux 세션이 있으면 중복 시작을 거부한다.

