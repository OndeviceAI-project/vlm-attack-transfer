"""Reproduce saved-result comparison on the GPU server; no inference."""
import csv, hashlib, json, statistics as st
from pathlib import Path
OUT=Path(__file__).resolve().parent
B=OUT.parents[1]
PARTS=[('V','vision'),('C','connector'),('L','language'),('VCL','vcl')]
COLS=['ref','RTN8','RTN6','RTN5','RTN4','RTN3','bnb8','bnb4']
PROTOS={'common':(512,'qwen2vl_attack_common_v1_i300_cap512_s01'),
        'extended':(5000,'qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01')}
csv.field_size_limit(10_000_000)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def case(part,col):
    return 'ref' if col=='ref' else f"{'rtn' if col.startswith('RTN') else 'bnb'}-{part}-{col[3:]}"
def csv_out(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def load(proto,part,component):
    cap,image_set=PROTOS[proto]
    name=f'qwen2vl_vision_{proto}_v1_r01' if part=='V' else f'qwen2vl_{component}_{proto}_components_v1_both_r01'
    p=B/'results'/(name+'.csv'); mp=p.with_suffix('.json')
    meta=read(mp); manifest_path=B/'reports'/name/'run_manifest.json'; manifest=read(manifest_path)
    digest=sha(p)
    assert meta['completed'] and meta['row_count']==720
    assert digest==meta['csv_sha256']==manifest['csv_sha256']
    assert sha(mp)==manifest['metadata_sha256']
    assert manifest['max_new_tokens']==cap and manifest['image_set_id']==image_set
    with p.open(encoding='utf8',newline='') as f:
        r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    expected_fields=['model','case','image_id','input','n_tokens']+(['hit_cap'] if cap==512 else ['ended_eos','output_limit_reached'])+['latency_s','text']
    assert fields==expected_fields
    selection_path=B/'adv'/image_set/'selection.json'; selected=read(selection_path)
    ids={r['image_id'] for r in selected['images']}
    expected={(case(part,col),iid,k) for col in COLS for iid in ids for k in ('clean','noise','adv')}
    assert len(ids)==30 and len(rows)==720 and {(r['case'],r['image_id'],r['input']) for r in rows}==expected
    cfg=meta['config'];env=cfg.get('environment',cfg);model=env['model']
    assert model['revision']==selected['source_revision'] and model['model_id']==selected['source_model']
    gen=cfg.get('gen_kwargs',cfg.get('generation_kwargs'))
    assert gen['max_new_tokens']==cap and gen['do_sample'] is False and gen['num_beams']==1
    hashes=cfg['inputs_sha256'] if cap==512 else meta['inputs_sha256']
    for im in selected['images']:
        for kind in ('clean','noise','adv'):
            assert hashes[im['image_id']+'/'+kind]==(im['sha256'] if kind=='adv' else im[kind+'_sha256'])
    flag='hit_cap' if cap==512 else 'output_limit_reached'
    for r in rows:
        n=int(r['n_tokens'])
        assert r['model']=='qwen2vl' and 1<=n<=cap and float(r['latency_s'])>=0
        assert r[flag] in ('True','False')
        if r[flag]=='True': assert n==cap
        if cap==5000:
            assert r['ended_eos'] in ('True','False')
            assert (r[flag]=='True')==(n==cap and r['ended_eos']=='False')
    source=dict(protocol=proto,part=part,experiment_id=name,csv_path=str(p.relative_to(B)),csv_sha256=digest,
        metadata_sha256=sha(mp),report_manifest_sha256=sha(manifest_path),image_set_id=image_set,
        selection_sha256=sha(selection_path),model_id=model['model_id'],model_revision=model['revision'],
        packages=env['packages'],generation=gen,rows=720,cohort_and_input_hashes_verified=True)
    return dict(name=name,meta=meta,rows=rows,ids=ids,flag=flag,cap=cap,source=source,
        lookup={(r['case'],r['image_id'],r['input']):r for r in rows})

def build():
    runs={};groups=[];sources=[];agreements=[];anomalies=[];timing=[]
    for proto in PROTOS:
        baseline=None
        for part,component in PARTS:
            run=load(proto,part,component);runs[proto,part]=run;sources.append(run['source'])
            refs={(r['image_id'],r['input']):r for r in run['rows'] if r['case']=='ref'}
            if baseline is None: baseline=refs;base_source=run['source']
            assert run['source']['packages']==base_source['packages']
            agreement=dict(protocol=proto,part=part,
                token_mismatches_vs_V=sum(r['n_tokens']!=baseline[k]['n_tokens'] for k,r in refs.items()),
                flag_mismatches_vs_V=sum(r[run['flag']]!=baseline[k][run['flag']] for k,r in refs.items()),
                text_mismatches_vs_V=sum(r['text']!=baseline[k]['text'] for k,r in refs.items()))
            assert agreement['token_mismatches_vs_V']==agreement['flag_mismatches_vs_V']==0
            agreements.append(agreement)
            for col in COLS:
                c=case(part,col);info=run['meta']['cases'][c];metrics={}
                for kind in ('clean','noise','adv'):
                    rs=[r for r in run['rows'] if r['case']==c and r['input']==kind]
                    lengths=[int(r['n_tokens']) for r in rs]
                    s=dict(protocol=proto,cap=run['cap'],part=part,method=col,case=c,input=kind,n_images=len(rs),
                        mean_tokens=st.mean(lengths),median_tokens=st.median(lengths),min_tokens=min(lengths),max_tokens=max(lengths),
                        cap_count=sum(r[run['flag']]=='True' for r in rs),
                        mean_latency_s=st.mean(float(r['latency_s']) for r in rs),
                        model_footprint_bytes=info.get('model_footprint_bytes'),peak_allocated_bytes=info.get('peak_allocated_bytes'),
                        source_experiment=run['name'])
                    assert s['n_images']==30
                    groups.append(s);metrics[kind]=s
                    if kind!='adv':
                        for r in rs:
                            if r[run['flag']]=='True':
                                iid=r['image_id']
                                anomalies.append(dict(protocol=proto,cap=run['cap'],part=part,method=col,image_id=iid,input=kind,
                                    **{k+'_tokens':int(run['lookup'][c,iid,k]['n_tokens']) for k in ('clean','noise','adv')}))
                timing.append(dict(protocol=proto,cap=run['cap'],part=part,method=col,
                    clean_mean_latency_s=metrics['clean']['mean_latency_s'],
                    noise_mean_latency_s=metrics['noise']['mean_latency_s'],adv_mean_latency_s=metrics['adv']['mean_latency_s'],
                    adv_mean_tokens=metrics['adv']['mean_tokens'],model_footprint_bytes=info.get('model_footprint_bytes'),
                    peak_allocated_bytes=info.get('peak_allocated_bytes')))
    index={(g['protocol'],g['part'],g['method'],g['input']):g for g in groups}
    def g(proto,part,col,kind='adv'): return index[proto,part,col,kind]
    assert len(groups)==192 and len(timing)==64
    csv_out(OUT/'group_stats.csv',groups);csv_out(OUT/'timing_memory.csv',timing)
    dump(OUT/'summary.json',dict(schema='components-comparison-v1',rows_analyzed=5760,new_component_rows=4320,
        previous_vision_rows=1440,groups=groups,baseline_agreement=agreements,control_anomalies=anomalies,
        attack_success_rate_computed=False,transfer_rate_computed=False))
    dump(OUT/'sources.json',dict(build_script_sha256=sha(Path(__file__)),source_runs=sources,source_csvs_modified=False))
    lines=['# V·C·L·VCL 양자화 결과 비교','',
        '기존 V 1,440행과 새 C·L·VCL 4,320행, 총 5,760행의 저장된 측정값을 비교했다. 집계와 해시·입력 대조는 GPU 서버에서 수행했고 모델을 다시 실행하지 않았다.','',
        '**각 칸은 공격 이미지 30장의 평균 생성 토큰 수 (상한 도달 장수/30)다.** 상한 도달은 EOS 없이 상한까지 생성한 경우다. 여기서는 이를 공격 성공률·전이율로 부르지 않는다.','',
        'V=비전, C=연결부, L=언어모델 본체, VCL=V+C+L. 출력층·입력 임베딩은 bf16 유지다. 원본 열은 각 실행에서 따로 측정한 ref이며 같은 프로토콜의 네 실행에서 이미지별 ref 토큰 수·상한 플래그가 모두 일치했다.','']
    for proto,(cap,image_set) in PROTOS.items():
        lines += [f'## {cap:,}토큰','',f'고정 공격 세트: {image_set}.','',
            '| 대상 | 원본 bf16 | RTN8 | RTN6 | RTN5 | RTN4 | RTN3 | bnb8 | bnb4 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for part,_ in PARTS:
            cells=[f"{g(proto,part,c)['mean_tokens']:,.1f} ({g(proto,part,c)['cap_count']}/30)" for c in COLS]
            lines.append('| '+part+' | '+' | '.join(cells)+' |')
        lines.append('')
    a=g('common','V','RTN3');b=g('extended','V','RTN3');d=g('extended','VCL','RTN4');e=g('extended','VCL','RTN3')
    lines += ['## 핵심 관찰','',
        f"1. **이 고정 공격 세트에서는 V RTN3의 긴 출력이 가장 적었다.** 512 조건 평균 {a['mean_tokens']:.1f}, 5,000 조건 평균 {b['mean_tokens']:.1f}토큰이며 모두 상한 도달 0/30이다.",
        f"2. **여러 부품을 함께 양자화하거나 비트를 낮춰도 항상 더 짧아지지는 않는다.** 5,000 조건의 VCL은 RTN4 {d['mean_tokens']:.1f}토큰·{d['cap_count']}/30에서 RTN3 {e['mean_tokens']:.1f}토큰·{e['cap_count']}/30으로 오히려 늘었다. 512 조건의 L도 RTN4 138.6 → RTN3 192.5토큰이었다.",
        '3. **bnb4는 V·L·VCL에서 두 프로토콜 모두 bnb8보다 공격 입력의 평균 길이가 짧았다.** C의 512 조건은 두 방법 모두 약 197.5토큰·9/30으로 거의 같았다.',
        '4. **512와 5,000에서 모든 순위가 유지되지는 않았다.** RTN8의 C/L 상한 도달 장수는 512에서 10/11, 5,000에서 16/14였다. 두 프로토콜은 공격 PNG도 다르므로 이를 출력 상한 하나의 효과로 단정하지 않는다.','',
        '## 원본·노이즈 입력 확인','',
        '범위는 원본 포함 8조건 각각에서 30장으로 구한 평균의 최솟값~최댓값이다. 상한 건수 분모 240은 같은 30장 × 8조건의 응답 수다.','',
        '| 상한 | 대상 | clean 평균 범위 | noise 평균 범위 | clean 상한/240 | noise 상한/240 |',
        '|---|---|---:|---:|---:|---:|']
    for proto,(cap,_) in PROTOS.items():
        for part,_ in PARTS:
            cl=[g(proto,part,c,'clean') for c in COLS];no=[g(proto,part,c,'noise') for c in COLS]
            lines.append(f"| {cap:,} | {part} | {min(x['mean_tokens'] for x in cl):.1f}~{max(x['mean_tokens'] for x in cl):.1f} | {min(x['mean_tokens'] for x in no):.1f}~{max(x['mean_tokens'] for x in no):.1f} | {sum(x['cap_count'] for x in cl)}/240 | {sum(x['cap_count'] for x in no)}/240 |")
    lines += ['','**clean은 모든 조건에서 상한 도달 0이었다. noise는 L RTN3에서 한 장이 상한에 도달했고 나머지는 모두 0이었다.**','']
    for x in anomalies:
        median=g(x['protocol'],x['part'],x['method'],'noise')['median_tokens']
        lines.append(f"- {x['cap']:,} 조건, {x['part']} {x['method']}, 사진 {x['image_id']}: clean {x['clean_tokens']}, noise {x['noise_tokens']:,}, adv {x['adv_tokens']:,}토큰. 이 조건의 noise 중앙값은 {median:.0f}토큰이다.")
    lines += ['','L RTN3의 noise 평균 상승은 한 장의 긴 출력 영향을 크게 받는다. 생성 길이가 짧아졌다는 사실만으로 정상 답변의 의미적 품질이 유지되거나 다른 공격에도 안전하다고 판단하지 않는다.','',
        '## 속도·메모리 보조 자료','',
        '평균 생성 시간과 GPU peak allocation은 [timing_memory.csv](timing_memory.csv)에 있다. 생성 시간은 답변 길이·초기 이미지 처리 비용을 포함하므로 이 값만으로 속도 순위를 정하지 않는다.','',
        '아래는 5,000 조건 메타데이터의 모델 footprint(MiB)다. peak GPU 메모리 및 실제 체크포인트 파일 크기와 다르다. RTN은 bf16 저장의 모의 양자화이므로 저장 용량이 줄지 않는다.','',
        '| 대상 | 원본 | RTN3 | bnb8 | bnb4 |','|---|---:|---:|---:|---:|']
    for part,_ in PARTS:
        cells=[]
        for col in ('ref','RTN3','bnb8','bnb4'):
            v=runs['extended',part]['meta']['cases'][case(part,col)].get('model_footprint_bytes')
            cells.append(f'{v/2**20:,.1f}' if v is not None else '미기록')
        lines.append('| '+part+' | '+' | '.join(cells)+' |')
    lines += ['','V의 과거 기록에 없는 항목은 추정하지 않았다. 별도로 관리되는 양자화 상태 등 모든 비용이 footprint에 포함된다고 보장하지 않으므로 실제 실행 자원은 peak allocation과 함께 확인한다.','',
        '## 자료와 재현','',
        '- [전체 조건·입력별 평균·중앙값·범위·상한 장수](group_stats.csv)',
        '- [생성 시간·모델 footprint·GPU peak allocation](timing_memory.csv)',
        '- [기계 판독 요약·ref 일치·대조군 예외](summary.json)',
        '- [원본 해시·환경·공격 세트 출처](sources.json)',
        '- [서버에서 실행한 집계 코드](build.py)','',
        '같은 프로토콜 안에서 고정 PNG·clean/noise 해시·모델 revision·패키지·생성 조건을 대조했다. 모든 값은 고정 평가 사진 30장에 대한 관측이며 표는 소수점 한 자리로 반올림했다. 원시 측정 CSV는 수정하지 않았다.','',
        '| 대상 | 512 원본 CSV | 5,000 원본 CSV |','|---|---|---|']
    for part,_ in PARTS:
        links=[f"[{runs[p,part]['name']}](../../results/{runs[p,part]['name']}.csv)" for p in PROTOS]
        lines.append('| '+part+' | '+' | '.join(links)+' |')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for s in sources: assert sha(B/s['csv_path'])==s['csv_sha256']
    print(json.dumps(dict(state='built',rows_analyzed=5760,summary_groups=192,
        baseline_agreement=agreements,control_anomalies=anomalies),ensure_ascii=True))
if __name__=='__main__': build()
