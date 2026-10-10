# data/

## image_list.csv — 두 팀 공통 사진 목록

COCO val2017에서 id가 작은 순서로 고른 50장(예비 측정에 쓴 목록) 중, 공격 설정 조정에 쓴 3장(39769, 139, 285)을 뺀 48장이다.

| split | 수 | 용도 |
|---|---|---|
| pilot | 3 | 예비 실험에 쓴 사진. **공식 실험에서 제외** |
| dev | 5 | 코드 확인, 공격 설정 조정. 결과를 보고서에 쓰지 않는다 |
| eval | 43 (후보) | 유효성 검사(docs/03 1단계) 뒤 order가 빠른 30장을 `eval_v1.txt`로 확정 |

열: `split, order, coco_id, url, note`. 파일 이름은 COCO 공식 이름과 같은 12자리(`000000000872.png`).

## eval_v1.txt — 확정된 평가 사진 30장

두 모델 모두 기준 모델(ref)에서 원본 사진의 답이 상한에 닿지 않은 id 중 order 순 30개. 한 줄에 id 하나. **2단계(공격 사진) 전에** 두 팀이 같이 만든다. 아직 없음.

## sets/<모델>/ — 사진 세트 (PNG, 커밋함)

```
sets/smolvlm/clean/   512×512 원본 리사이즈
sets/smolvlm/noise/   ±8 무작위 노이즈 대조군
sets/smolvlm/adv/     공격 사진 + attack_log.csv + SHA256SUMS  (고정 뒤 수정 금지)
sets/qwen2vl/...      448×448, 같은 구조
```

## coco_cache/ — 내려받은 COCO 원본 (커밋하지 않음)

`image_list.csv`의 url로 받는다. 지워도 다시 받으면 된다.
