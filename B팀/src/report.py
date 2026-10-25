"""Build descriptive tables from saved rows only; no model inference or ASR."""
import argparse,csv,json,hashlib,statistics as st
from pathlib import Path
B=Path(__file__).resolve().parents[1]
def write_summary(path, summary):
 """Refresh the generated block without replacing the experiment description."""
 start='<!-- measurements:start -->';end='<!-- measurements:end -->'
 previous=path.read_text(encoding='utf-8') if path.exists() else ''
 if start in previous or end in previous:
  if previous.count(start)!=1 or previous.count(end)!=1 or previous.index(start)>previous.index(end):
   raise ValueError(f'Invalid measurement block markers: {path}')
  before,remaining=previous.split(start,1);_,after=remaining.split(end,1)
  block=summary.replace('# 측정값 요약','## 측정값 요약',1).rstrip()
  updated=before+start+'\n'+block+'\n'+end+after
 elif previous and not previous.startswith('# 측정값 요약\n'):
  raise ValueError(f'Add measurement block markers before rebuilding this report: {path}')
 else:
  updated=summary
 path.write_text(updated,encoding='utf-8')

def build(label,image_set=None):
 if image_set is None:
  image_set=json.loads((B/'reports'/label/'run_manifest.json').read_text(encoding='utf-8'))['image_set_id']
 csv.field_size_limit(10000000);p=B/'results'/(label+'.csv');meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
 with p.open(encoding='utf-8',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['csv_sha256']
 if not meta.get('completed'):raise ValueError('Build reports only after measurement completes')
 cases=meta['config']['cases']
 selected=json.loads((B/'adv'/image_set/'selection.json').read_text(encoding='utf-8'))['images'];assert len(selected)==30
 ids={r['image_id'] for r in selected};expected={(c,i,k) for c in cases for i in ids for k in ['clean','noise','adv']}
 if len(rows)!=len(expected) or {(r['case'],r['image_id'],r['input']) for r in rows}!=expected:
  raise ValueError('Result rows do not match the recorded cases and image cohort')
 out=B/'reports'/label;out.mkdir(parents=True,exist_ok=True)
 component=meta['config'].get('quantization_profile')
 if component and not (out/'README.md').exists():
  cfg=meta['config'];env=cfg.get('environment',cfg)
  cap=cfg.get('max_new_tokens',cfg.get('gen_kwargs',{}).get('max_new_tokens'))
  intro=[f'# Qwen2-VL 부품 양자화 — {component["component"]}', '',
   f'실험 ID: `{label}`. 양자화 대상은 **{component["targets"]}의 Linear weight**이며 출력층·입력 임베딩은 bf16과 공유 상태를 유지한다.', '',
   f'고정 공격 이미지 세트 `{image_set}`를 재사용했다. 사진 30장 × clean·noise·adv × 모델 조건 {len(cases)}개 = {len(rows)}행이다. 출력 상한은 {cap}토큰이다.', '',
   f'조건: {", ".join(cases)}. RTN group 64; bnb 4비트는 NF4이다. RTN은 모의 양자화여서 저장 용량을 줄이지 않는다.', '',
   f'GPU: {env.get("gpu","메타데이터 참조")}. 패키지·모델 revision·census·보호된 가중치 확인 결과는 결과 JSON에 기록한다.', '',
   '공격 성공률·전이율과 정상 답변 품질은 별도 분석 대상이다.', '',
   '<!-- measurements:start -->', '<!-- measurements:end -->', '']
  (out/'README.md').write_text('\n'.join(intro),encoding='utf-8')
 labels=['본 측정 원본' if c=='ref' else c for c in cases]
 matrix=['### 사진별 같은 공격 이미지의 출력 토큰 수','','각 행은 같은 공격 PNG 한 장을 비교한다. `채택 당시 원본`은 이미지 채택 과정에서 기록한 원본 모델의 토큰 수이고, 나머지는 최종 평가에서 동일 PNG를 넣어 얻은 토큰 수다. 모든 값의 단위는 토큰이다.','','| '+' | '.join(['사진 ID','채택 당시 원본']+labels)+' |','|---|'+'---:|'*(len(cases)+1)]
 lookup={(r['case'],r['image_id'],r['input']):r for r in rows}
 for im in selected:
  iid=im['image_id'];chosen=im.get('saved_tokens',im.get('initial_tokens'))
  vals=[iid,str(chosen)]+[lookup[(c,iid,'adv')]['n_tokens'] for c in cases]
  matrix.append('| '+' | '.join(vals)+' |')
 summary=['# 측정값 요약','','성공·실패, ASR, 전이율은 계산하지 않았다. 평균·중앙값·범위·생성 시간과 출력 상한 도달 장수만 기술한다.','','### 모델·입력별 요약','', '| 모델 | 입력 | 평균 토큰 | 중앙값 | 최소 | 최대 | 상한 도달 장수 | 평균 생성 초 |','|---|---|---:|---:|---:|---:|---:|---:|']
 stats=[]
 for c in cases:
  for k in ['clean','noise','adv']:
   rs=[r for r in rows if r['case']==c and r['input']==k];assert len(rs)==30
   n=[int(r['n_tokens']) for r in rs];limit='hit_cap' if 'hit_cap' in fields else 'output_limit_reached'
   v=dict(case=c,input=k,mean_tokens=st.mean(n),median_tokens=st.median(n),min_tokens=min(n),max_tokens=max(n),output_limit_count=sum(r[limit]=='True' for r in rs),mean_seconds=st.mean(float(r['latency_s']) for r in rs))
   stats.append(v);summary.append(f"| {c} | {k} | {v['mean_tokens']:.1f} | {v['median_tokens']:.1f} | {min(n)} | {max(n)} | {v['output_limit_count']}/30 | {v['mean_seconds']:.2f} |")
 summary+=['','RTN은 양자화된 값을 부동소수점 가중치로 사용하는 모의 양자화다. 이 표는 모델 품질·메모리 절감·토큰당 속도 우열을 평가하지 않는다.','']+matrix+['',f'원본: ../../results/{label}.csv',f'환경·설정: ../../results/{label}.json','']
 write_summary(out/'README.md','\n'.join(summary))
 (out/'measurements-summary.json').write_text(json.dumps(dict(experiment=label,csv_sha256=meta['csv_sha256'],attack_success_assessed=False,groups=stats),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 manifest_path=out/'run_manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  if not manifest.get('published_record'):
   manifest.update(csv_sha256=meta['csv_sha256'],metadata_sha256=hashlib.sha256(p.with_suffix('.json').read_bytes()).hexdigest(),completed=True)
   manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('REPORT_BUILT',label,f'rows={len(rows)} images={len(ids)}')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_name',nargs='+');a=p.parse_args()
 for name in a.run_name:
  if '/' in name or name.startswith('.'):p.error('Use a run name, not a path')
  build(name)
