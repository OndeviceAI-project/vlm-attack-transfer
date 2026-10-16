"""Build descriptive tables from saved rows only; no model inference or ASR."""
import argparse,csv,json,hashlib,statistics as st
from pathlib import Path
B=Path(__file__).resolve().parents[1]
CASES=['ref','rtn-V-8','rtn-V-6','rtn-V-5','rtn-V-4','rtn-V-3','bnb-V-8','bnb-V-4']
def build(label,image_set=None):
 if image_set is None:
  image_set=json.loads((B/'reports'/label/'run_manifest.json').read_text())['image_set_id']
 csv.field_size_limit(10000000);p=B/'results'/(label+'.csv');meta=json.loads(p.with_suffix('.json').read_text())
 with p.open() as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['csv_sha256']
 assert len(rows)==720 and len({(r['case'],r['image_id'],r['input']) for r in rows})==720
 selected=json.loads((B/'adv'/image_set/'selection.json').read_text())['images'];assert len(selected)==30
 out=B/'reports'/label;out.mkdir(parents=True,exist_ok=True)
 matrix=['# 사진별 같은 공격 이미지의 출력 토큰 수','','원본 모델 열은 이미지 생성 당시 채택한 PNG의 저장본 검사 토큰 수, 나머지는 동일 PNG의 모델별 본 측정값이다. 모든 값의 단위는 토큰. 출력 상한 도달은 성공 판정이 아니다.','','| 사진 ID | 채택 당시 원본 | 본 측정 원본 | RTN8 | RTN6 | RTN5 | RTN4 | RTN3 | bnb8 | bnb4 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 lookup={(r['case'],r['image_id'],r['input']):r for r in rows}
 for im in selected:
  iid=im['image_id'];chosen=im.get('saved_tokens',im.get('initial_tokens'))
  vals=[iid,str(chosen)]+[lookup[(c,iid,'adv')]['n_tokens'] for c in CASES]
  matrix.append('| '+' | '.join(vals)+' |')
 (out/'image-model-tokens.md').write_text('\n'.join(matrix)+'\n')
 summary=['# 측정값 요약','','성공·실패, ASR, 전이율은 계산하지 않았다. 평균·중앙값·범위·생성 시간과 출력 상한 도달 장수만 기술한다.','', '| 모델 | 입력 | 평균 토큰 | 중앙값 | 최소 | 최대 | 상한 도달 장수 | 평균 생성 초 |','|---|---|---:|---:|---:|---:|---:|---:|']
 stats=[]
 for c in CASES:
  for k in ['clean','noise','adv']:
   rs=[r for r in rows if r['case']==c and r['input']==k];assert len(rs)==30
   n=[int(r['n_tokens']) for r in rs];limit='hit_cap' if 'hit_cap' in fields else 'output_limit_reached'
   v=dict(case=c,input=k,mean_tokens=st.mean(n),median_tokens=st.median(n),min_tokens=min(n),max_tokens=max(n),output_limit_count=sum(r[limit]=='True' for r in rs),mean_seconds=st.mean(float(r['latency_s']) for r in rs))
   stats.append(v);summary.append(f"| {c} | {k} | {v['mean_tokens']:.1f} | {v['median_tokens']:.1f} | {min(n)} | {max(n)} | {v['output_limit_count']}/30 | {v['mean_seconds']:.2f} |")
 summary+=['','RTN은 양자화된 값을 부동소수점 가중치로 사용하는 모의 양자화다. 이 표는 모델 품질·메모리 절감·토큰당 속도 우열을 평가하지 않는다.','',f'원본: ../../results/{label}.csv',f'환경·설정: ../../results/{label}.json','']
 (out/'README.md').write_text('\n'.join(summary))
 (out/'measurements-summary.json').write_text(json.dumps(dict(experiment=label,csv_sha256=meta['csv_sha256'],attack_success_assessed=False,groups=stats),ensure_ascii=False,indent=2)+'\n')
 print('REPORT_BUILT',label,'rows=720 images=30')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_name',nargs='+');a=p.parse_args()
 for name in a.run_name:
  if '/' in name or name.startswith('.'):p.error('Use a run name, not a path')
  build(name)
