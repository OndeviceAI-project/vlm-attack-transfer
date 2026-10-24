"""Run versioned common/extended protocols; --check never starts inference."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import argparse,csv,gc,importlib.metadata,json,signal,sys,time,traceback,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import runtime as rt
from runtime import (ROOT,REPO,settings,sha256,write_json,save_csv,fingerprint,
    model_metadata,source_label,image_ids,input_path,read_image,load_case,load_processor,generate,check_preprocessing)
import attack
import evaluate
from attack import generate_one,verify_record
HERE=Path(__file__).resolve().parent
BTEAM=HERE.parent
BASE=BTEAM
LABEL=os.environ.get('BTEAM_RUN_NAME','unset')
RUN=BTEAM/'results'/LABEL
CSV=BTEAM/'results'/(LABEL+'.csv')
ADV=BTEAM/'adv'/os.environ.get('BTEAM_IMAGE_SET','unset')
STATUS=RUN/'status.json'

def now():return datetime.now(timezone.utc).isoformat()

def write_status(**kw):
 old=json.loads(STATUS.read_text()) if STATUS.exists() else {}
 old.update(updated_at=now(),**kw);rt.write_json(STATUS,old)

def validate_protocol():
 s=rt.settings
 expected=dict(SETTINGS_VERSION='v1',MAX_NEW_TOKENS=512,ATTACK_ITERS=300,ATTACK_MAX_ITERS=500,EPS=8,STEP=1,TARGET_LEN=128,REFRESH=20,W_UNC=.05,W_DIV=0,QUANT_GROUP=64,NOISE_SEED=2026)
 for k,v in expected.items():
  if getattr(s,k)!=v:raise ValueError(f'Official setting mismatch: {k}')
 assert s.GEN_KWARGS==dict(do_sample=False,num_beams=1,max_new_tokens=512,repetition_penalty=1.,temperature=None,top_p=None,top_k=None)
 assert s.PROMPT=='Describe this image in one sentence.'
 assert s.CASES_PHASE1==['ref','rtn-V-8','rtn-V-6','rtn-V-5','rtn-V-4','rtn-V-3','bnb-V-8','bnb-V-4']
 ids=rt.image_ids('eval');assert len(ids)==30
 manifest=json.loads((rt.ROOT/'data/manifest.json').read_text())
 refs={(r['split'],r['input'],r['image_id']):r['sha256'] for r in manifest['records']}
 hashes={}
 for iid in ids:
  imgs={}
  for kind in ('clean','noise'):
   p=rt.input_path('eval',kind,iid);img=rt.read_image(p);sha=rt.sha256(p)
   assert sha==refs['eval',kind,iid]
   hashes[iid+'/'+kind]=sha;imgs[kind]=np.asarray(img).astype(np.int16)
  assert int(np.abs(imgs['noise']-imgs['clean']).max())<=8
 assert rt.CSV_FIELDS==['model','case','image_id','input','n_tokens','hit_cap','latency_s','text']
 return ids,hashes

def verify_finished(ids):
 job,hashes=evaluate.verify_inputs('eval',ids,ADV)
 assert len(job['images'])==30
 for im in job['images']:
  assert im['status']=='complete' and 0<=im['iters_run']<=300 and im['linf']<=8 and not im['numerical_failure']
  p=ADV/(im['image_id']+'.png');assert rt.sha256(p)==im['sha256']
  assert np.abs(np.asarray(rt.read_image(p)).astype(np.int16)-np.asarray(rt.read_image(rt.input_path('eval','clean',im['image_id']))).astype(np.int16)).max()<=8
 with CSV.open(newline='',encoding='utf8') as f:
  r=csv.DictReader(f);assert r.fieldnames==rt.CSV_FIELDS;rows=list(r)
 keys={(r['case'],r['image_id'],r['input']) for r in rows}
 expected={(c,i,k) for c in rt.settings.CASES_PHASE1 for i in ids for k in ('clean','noise','adv')}
 assert len(rows)==720 and keys==expected
 for r in rows:
  assert r['model']=='qwen2vl' and len(r['image_id'])==12 and 1<=int(r['n_tokens'])<=512
  assert r['hit_cap'] in ('True','False') and float(r['latency_s'])>=0
  if r['hit_cap']=='True':assert int(r['n_tokens'])==512
 meta=json.loads(CSV.with_suffix('.json').read_text())
 assert meta['completed'] and meta['row_count']==720 and meta['csv_sha256']==rt.sha256(CSV)
 assert len(meta['cases'])==8
 print('OFFICIAL_VERIFIED images=30 rows=720 schema=common-v1',flush=True)
 return meta

def configure_gpu():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(2026);np.random.seed(2026)

def plan():
    if (settings.EPS,settings.STEP,settings.TARGET_LEN,settings.REFRESH,settings.W_UNC,settings.W_DIV)!=(8,1,128,20,.05,0):
        raise ValueError('Common attack objective settings changed')
    ids=image_ids('eval');manifest=json.loads((ROOT/'data/manifest.json').read_text())
    entries={(r['split'],r['input'],r['image_id']):r for r in manifest['records']}
    hashes={}
    for iid in ids:
        clean=None
        for kind in ['clean','noise']:
            path=input_path('eval',kind,iid)
            actual=sha256(path)
            if actual!=entries[('eval',kind,iid)]['sha256']:raise ValueError(f'Input changed: {path}')
            image=read_image(path);arr=np.asarray(image).astype(np.int16)
            if kind=='clean':clean=arr
            elif np.abs(arr-clean).max()>8:raise ValueError('Random noise exceeded the pixel bound')
            hashes[f'{iid}/{kind}']=actual
    model=model_metadata();snap=Path(model['snapshot_path'])
    index=json.loads((snap/'model.safetensors.index.json').read_text())
    shards=sorted(set(index['weight_map'].values()))
    sizes={name:(snap/name).stat().st_size for name in shards}
    if sizes.get('model-00001-of-00002.safetensors')!=3988609112 or sizes.get('model-00002-of-00002.safetensors')!=429441656:
        raise ValueError('Cached model shard sizes do not match the verified model')
    code_paths=[HERE/n for n in ['runtime.py','data.py','attack.py','evaluate.py','pipeline.py','report.py']]+[BTEAM/'scripts/worker.sh',BTEAM/'scripts/run.sh']+[REPO/'공통/settings.py',REPO/'공통/quant_rtn.py']
    if len(settings.CASES_PHASE1)!=8:raise ValueError('Expected eight model cases')
    return dict(protocol='b-measurement-i500-cap5000-tol20-det-v1',split='eval',image_ids=ids,
        cases=settings.CASES_PHASE1,expected_rows=720,iters=500,checkpoint_iterations=list(range(20,501,20)),
        max_new_tokens=5000,tolerance=.20,eps=8,step=1,target_len=128,target_refresh=20,w_unc=.05,w_div=0.,
        prompt=settings.PROMPT,generation_kwargs={**settings.GEN_KWARGS,'max_new_tokens':5000},
        seed=2026,deterministic_algorithms=True,cudnn_deterministic=True,cudnn_benchmark=False,
        cublas_workspace_config=':4096:8',early_stop_on_length=False,attack_success_assessed=False,
        selection='descending initial tokens; ties use earlier iteration; recheck relative difference <= .20',
        model=model,input_sha256=hashes,model_shard_sizes=sizes,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','bitsandbytes','accelerate','pillow','numpy']},
        code_sha256={source_label(p):sha256(p) for p in code_paths})

def common_run():
 ids,inputs=validate_protocol()
 torch.manual_seed(rt.settings.NOISE_SEED)
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.benchmark=False
 RUN.mkdir(parents=True,exist_ok=True)
 protocol=dict(label=LABEL,image_set_id=ADV.name,settings_version=rt.settings.SETTINGS_VERSION,settings_sha256=rt.sha256(rt.REPO/'공통/settings.py'),image_ids=ids,inputs_sha256=inputs,attack_iterations_budget=300,output_cap=512,generation_kwargs=rt.settings.GEN_KWARGS,environment=rt.environment(),determinism=dict(seed=2026,algorithms=True,cudnn=True,cudnn_benchmark=False,cublas=os.environ['CUBLAS_WORKSPACE_CONFIG']),candidate_checks='after updates 1,21,...,281 and at final update 300; matches common script loop convention',selection='best observed (output-limit flag, token count); ties retain earlier candidate, saved PNG reloaded and checked',early_stop='two consecutive checks at 512 without EOS',analysis='measurements only; no ASR/transfer-rate analysis',csv_fields=rt.CSV_FIELDS)
 fp=rt.fingerprint(protocol)
 pp=RUN/'protocol.json'
 if pp.exists():
  if rt.fingerprint(json.loads(pp.read_text()))!=fp:raise ValueError('Existing official run configuration changed; use a new run name')
 else:rt.write_json(pp,protocol)
 snapshots=RUN/'source'
 for f in [Path(__file__),HERE/'attack.py',HERE/'evaluate.py',HERE/'runtime.py',HERE/'data.py',HERE/'report.py',BASE/'scripts/worker.sh',rt.REPO/'공통/settings.py',rt.REPO/'공통/quant_rtn.py',rt.REPO/'공통/image_list.csv',rt.REPO/'README.md',rt.REPO/'이미지생성/README.md',rt.REPO/'이미지생성/attack_smolvlm.py']:
  dst=snapshots/rt.source_label(f);dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   assert rt.sha256(dst)==rt.sha256(f),'Source snapshot changed'
  else:shutil.copyfile(f,dst)
 prior=json.loads(STATUS.read_text()) if STATUS.exists() else {}
 if prior.get('state')=='complete':verify_finished(ids);return
 write_status(state='running',stage='generation',pid=os.getpid(),started_at=prior.get('started_at',now()),fingerprint=fp,error=None,expected_images=30,expected_rows=720,adv_dir=str(ADV),csv=str(CSV))
 def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  print('OFFICIAL_START',now(),'cap=512 iter_budget=300 settings=v1',flush=True)
  attack.common_main(['--split','eval','--out',str(ADV)])
  gc.collect();torch.cuda.empty_cache()
  export_selection('common_v1')
  write_status(stage='measurement',generated_images=30)
  evaluate.common_main(['--split','eval','--adv-dir',str(ADV),'--out',str(CSV)])
  write_status(stage='verification')
  meta=verify_finished(ids)
  meta['execution_protocol']=protocol
  rt.write_json(CSV.with_suffix('.json'),meta)
  write_status(state='complete',stage='complete',generated_images=30,measurement_rows=720,csv_sha256=rt.sha256(CSV),completed_at=now(),error=None)
  print('OFFICIAL_COMPLETE images=30 rows=720 analysis=not_run',now(),flush=True)
 except BaseException as e:
  write_status(state='interrupted' if isinstance(e,KeyboardInterrupt) else 'error',error=f'{type(e).__name__}: {e}')
  raise

class Pipeline:

    def __init__(self, cfg):
        self.run = RUN
        self.advdir = ADV
        self.csv_path = CSV
        self.cfg = cfg
        self.ids = cfg['image_ids']
        self.id = fingerprint(cfg)
        RUN.mkdir(parents=True, exist_ok=True)
        self.path = RUN / 'status.json'
        if self.path.exists():
            self.status = json.loads(self.path.read_text())
            if self.status['fingerprint'] != self.id:
                if self.status.get('state') == 'ready' and self.status.get('generated_images', 0) == 0 and (self.status.get('measurement_rows', 0) == 0):
                    self.status.update(fingerprint=self.id, config=cfg)
                else:
                    raise ValueError('Configuration changed; use a separate run folder')
        else:
            self.status = dict(fingerprint=self.id, config=cfg, state='ready', stage='not_started', generated_images=0, measurement_rows=0, expected_images=30, expected_rows=720, created_at=now(), completed=False)
        self.update()

    def update(self, **values):
        self.status.update(values, updated_at=now())
        write_json(self.path, self.status)

    def attack_progress(self, iid, iteration):
        self.update(state='running', stage='generation', current_image=iid, current_iteration=iteration)

    def generate(self):
        out = ADV
        out.mkdir(parents=True, exist_ok=True)
        adopted = {}
        for iid in self.ids:
            p = out / iid / 'record.json'
            if p.exists():
                record = json.loads(p.read_text())
                if record.get('status') == 'adopted':
                    verify_record(record)
                    adopted[iid] = record
        self.update(state='running', stage='generation', generated_images=len(adopted))
        pending = [iid for iid in self.ids if iid not in adopted]
        if pending:
            processor = load_processor()
            model, info = load_case('ref', attack=True)
            self.status['reference_model'] = info
            generate(model, processor, read_image(input_path('eval', 'clean', pending[0])), {'max_new_tokens': 1})
            for iid in pending:
                self.attack_progress(iid, 0)
                adopted[iid] = generate_one(model, processor, iid, out, 'eval', self.attack_progress)
                self.update(generated_images=len(adopted))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        ordered = [adopted[iid] for iid in self.ids]
        manifest = dict(completed=True, config=self.cfg, images=ordered, model_revision=self.cfg['model']['revision'])
        write_json(out / 'adopted.json', manifest)
        (out / 'SHA256SUMS').write_text(''.join((f'{r['selected']['sha256']}  {r['image_id']}.png\n' for r in ordered)))
        self.update(generated_images=30, stage='generation_complete')
        return {r['image_id']: r for r in ordered}
    measure = evaluate.measure_extended




def export_selection(protocol):
    """Make a small reusable manifest after generation, retaining the raw records."""
    source=json.loads((ADV/('job.json' if protocol=='common_v1' else 'adopted.json')).read_text())
    images=[]
    for r in source['images']:
        iid=r['image_id']
        im=dict(image_id=iid,path=iid+'.png',sha256=sha256(ADV/(iid+'.png')),
            clean_sha256=sha256(input_path('eval','clean',iid)),noise_sha256=sha256(input_path('eval','noise',iid)))
        if protocol=='common_v1':
            im.update(iterations_run=r['iters_run'],linf=r['linf'],start_tokens=r['start_len'],
                best_tokens=r['best_len'],saved_tokens=r['saved_len'],seconds=r['seconds'])
        else:
            chosen=r['selected']
            im.update(iterations_run=r['iterations_run'],linf=chosen['linf'],selected_iteration=chosen['iteration'],
                initial_tokens=chosen['initial_n_tokens'],recheck_tokens=chosen['recheck_n_tokens'],
                relative_difference=chosen['relative_token_difference'],seconds=r['image_seconds'])
        images.append(im)
    value=dict(experiment=LABEL,image_set_id=ADV.name,split='eval',attack_protocol=protocol,
        source_model=model_metadata()['model_id'],source_revision=model_metadata()['revision'],
        settings_version=settings.SETTINGS_VERSION,generation_target='unquantized bf16 reference model',images=images)
    path=ADV/'selection.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Image-set manifest changed; refusing replacement')
    write_json(path,value)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('protocol',choices=['common_v1','extended_v1'])
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true')
    actions.add_argument('--run',action='store_true')
    actions.add_argument('--status',action='store_true')
    parser.add_argument('--measure-only',action='store_true',help='Reuse an immutable published image set')
    args=parser.parse_args()
    import re
    for value in [LABEL,ADV.name]:
        if value=='unset' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}',value):
            parser.error('Set BTEAM_RUN_NAME and BTEAM_IMAGE_SET to explicit safe names')
    if args.status:
        value=json.loads(STATUS.read_text()) if STATUS.exists() else {'state':'not_started'}
        print(json.dumps(value,ensure_ascii=False,indent=2));return
    ids,_=validate_protocol()
    cfg=plan() if args.protocol=='extended_v1' else None
    if cfg is not None:cfg.update(image_set_id=ADV.name)
    adopted=None
    if args.measure_only:
        if not (ADV/'selection.json').exists():raise ValueError('Measure-only requires a published selection.json')
        selection=json.loads((ADV/'selection.json').read_text())
        if selection['attack_protocol']!=args.protocol:raise ValueError('Protocol differs from the published image set')
        _,hashes=evaluate.verify_inputs('eval',ids,ADV)
        adopted={im['image_id']:dict(selected=dict(sha256=im['sha256'])) for im in selection['images']}
        if cfg is not None:
            cfg.update(mode='measure_only',image_set_id=ADV.name,image_set_sha256=sha256(ADV/'selection.json'))
    if args.check:
        print('PREFLIGHT_OK',args.protocol,'images=30 inputs=60 rows=720 inference_started=False');return
    record_path=BTEAM/'reports'/LABEL/'run_manifest.json'
    if record_path.exists() and json.loads(record_path.read_text()).get('published_record'):
        raise ValueError('This is a preserved historical run; choose a new run name')
    if args.measure_only:
        # Re-evaluation must have its own output identity; published results stay immutable.
        if CSV.exists() and not STATUS.exists():raise ValueError('Use a new run name for measurement')
    elif ((ADV/'selection.json').exists() and not STATUS.exists()) or (CSV.exists() and not STATUS.exists()):
        raise ValueError('Published files are immutable; use a new run/image-set name or --measure-only')
    record=dict(experiment_id=LABEL,protocol=args.protocol,image_set_id=ADV.name,published_record=False,code_sha256={f.name:sha256(f) for f in HERE.glob('*.py')})
    if record_path.exists():
        if json.loads(record_path.read_text())!=record:raise ValueError('Existing run metadata changed; choose a new run name')
    else:write_json(record_path,record)
    configure_gpu()
    if args.protocol=='common_v1':
        if args.measure_only:
            RUN.mkdir(parents=True,exist_ok=True)
            write_status(state='running',stage='measurement',image_set_id=ADV.name,pid=os.getpid())
            try:
                evaluate.common_main(['--split','eval','--adv-dir',str(ADV),'--out',str(CSV)])
                write_status(state='complete',stage='complete',measurement_rows=720,csv_sha256=sha256(CSV))
            except BaseException as error:
                write_status(state='error',error=str(error));raise
        else:common_run()
        return
    pipeline=Pipeline(cfg)
    if pipeline.status.get('completed'):
        if sha256(CSV)!=pipeline.status['csv_sha256']:raise ValueError('Completed CSV changed')
        print('ALREADY_COMPLETE');return
    def interrupted(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        pipeline.update(state='running',started_at=now(),pid=os.getpid(),error=None)
        if adopted is None:
            adopted=pipeline.generate();export_selection('extended_v1')
        pipeline.measure(adopted)
    except BaseException as error:
        pipeline.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error))
        raise

if __name__=='__main__':main()
