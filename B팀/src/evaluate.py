"""Evaluate unchanged clean/noise/adv PNGs; one atomic, resumable CSV per run."""
import argparse
from runtime import model_metadata
import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from case_config import COMPONENTS, BACKENDS, DEFAULT_BACKEND, component_cases, profile
from runtime import (ROOT, REPO, settings, CSV_FIELDS, image_ids, input_path,
    load_processor, read_image, check_preprocessing, load_case, generate,
    write_json, save_csv, sha256, environment, fingerprint, verify_fixed_pixels)

def now():
    return datetime.now(timezone.utc).isoformat()

def verify_inputs(split, ids, advdir):
    job = published_job(split, ids, advdir) if not (advdir / 'job.json').exists() else json.loads((advdir / 'job.json').read_text())
    if not job.get('completed') or job['config']['split'] != split:
        raise ValueError('A completed attack set from the same split is required')
    if job['config']['environment']['model']['revision'] != model_metadata()['revision']:
        raise ValueError('Attack model revision does not match evaluation')
    if job['config']['environment']['settings_version'] != settings.SETTINGS_VERSION:
        raise ValueError('Attack settings version does not match evaluation')
    images = {r['image_id']: r for r in job['images']}
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    reference = {(r['split'], r['input'], r['image_id']): r for r in manifest['records']}
    fixed = json.loads((Path(__file__).resolve().parents[1] / 'results/input-manifest.json').read_text())
    frozen = {(r['split'], r['input'], r['image_id']): r['sha256'] for r in fixed['records']}
    hashes = {}
    for iid in ids:
        if iid not in images: raise ValueError(f'Missing attack image: {iid}')
        for kind in ['clean', 'noise', 'adv']:
            path = advdir / (iid + '.png') if kind == 'adv' else input_path(split, kind, iid)
            actual = sha256(path)
            expected = images[iid]['sha256'] if kind == 'adv' else reference[(split, kind, iid)]['sha256']
            if actual != expected: raise ValueError(f'Changed fixed input: {path}')
            if kind != 'adv' and actual != frozen[(split, kind, iid)]:
                raise ValueError(f'Server input differs from the frozen input manifest: {path}')
            hashes[f'{iid}/{kind}'] = actual
        if images[iid]['clean_sha256'] != hashes[f'{iid}/clean']:
            raise ValueError('Attack clean image differs from evaluation clean image')
        if images[iid].get('noise_sha256', hashes[f'{iid}/noise']) != hashes[f'{iid}/noise']:
            raise ValueError('Attack noise image differs from evaluation noise image')
        linf = verify_fixed_pixels(iid, read_image(input_path(split, 'clean', iid)),
            read_image(input_path(split, 'noise', iid)), read_image(advdir / (iid + '.png')))
        if linf != images[iid]['linf']:
            raise ValueError(f'Actual attack PNG differs from the recorded epsilon: {iid}')
    return job, hashes

def common_main(argv=None, progress=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'eval'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cases', nargs='+', choices=settings.CASES_PHASE1)
    parser.add_argument('--component', choices=COMPONENTS)
    parser.add_argument('--backend', choices=BACKENDS)
    parser.add_argument('--adv-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.backend is None:
        args.backend = DEFAULT_BACKEND if args.component else 'rtn'
    ids = image_ids(args.split, args.limit)
    if args.component and args.cases:
        parser.error('Use --component or --cases, not both')
    if not args.component and args.backend != 'rtn':
        parser.error('--backend requires --component')
    cases = component_cases(args.component, args.backend) if args.component else (args.cases or settings.CASES_PHASE1)
    if len(set(cases)) != len(cases) or 'ref' not in cases:
        parser.error('Unique cases including ref are required')
    if args.split == 'eval' and not args.component and cases != settings.CASES_PHASE1:
        parser.error('Eval requires all eight common cases in order')
    if args.out.suffix != '.csv': parser.error('Output must be .csv')
    advdir = args.adv_dir.resolve()
    selection_path = advdir / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text(encoding='utf-8'))['attack_protocol'] != 'common_v1':
        raise ValueError('Common measurements require the common_v1 attack image set')
    attack, hashes = verify_inputs(args.split, ids, advdir)
    processor = load_processor()
    checks = {}
    for iid in ids:
        for kind in ['clean', 'noise', 'adv']:
            path = advdir / (iid + '.png') if kind == 'adv' else input_path(args.split, kind, iid)
            checks[f'{iid}/{kind}'] = check_preprocessing(processor, read_image(path))
    config = dict(split=args.split, image_ids=ids, cases=cases, inputs_sha256=hashes,
        attack_fingerprint=attack['fingerprint'], environment=environment(), gen_kwargs=settings.GEN_KWARGS)
    if args.component:
        config['quantization_profile'] = profile(args.component, args.backend)
    run_id = fingerprint(config)
    meta_path = args.out.with_suffix('.json')
    expected = {(case, iid, kind) for case in cases for iid in ids for kind in ['clean', 'noise', 'adv']}
    rows = []
    done = set()
    if args.out.exists():
        if not meta_path.exists(): raise ValueError('Existing CSV lacks run metadata')
        meta = json.loads(meta_path.read_text())
        if meta['fingerprint'] != run_id: raise ValueError('Use a new CSV for a changed run configuration')
        if meta.get('completed') and meta.get('csv_sha256') != sha256(args.out): raise ValueError('Completed CSV has been changed')
        with args.out.open(newline='', encoding='utf8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS: raise ValueError('CSV schema mismatch')
            rows = list(reader)
        for row in rows:
            key = row['case'], row['image_id'], row['input']
            if key not in expected or key in done or row['model'] != 'qwen2vl':
                raise ValueError('Unexpected or duplicate existing row')
            if any(v is None for v in row.values()): raise ValueError('Incomplete existing CSV row')
            done.add(key)
    else:
        if meta_path.exists(): raise ValueError('Metadata exists without the CSV; use a new output name')
        meta = dict(fingerprint=run_id, config=config, preprocessing=checks, cases={}, completed=False)
        write_json(meta_path, meta)
        save_csv(args.out, rows)
    for case in cases:
        keys = {(case, iid, kind) for iid in ids for kind in ['clean', 'noise', 'adv']}
        if keys.issubset(done): continue
        gc.collect()
        torch.cuda.empty_cache()
        model, info = load_case(case)
        meta['cases'][case] = info
        write_json(meta_path, meta)
        generate(model, processor, read_image(input_path(args.split, 'clean', ids[0])), dict(max_new_tokens=1))
        torch.cuda.reset_peak_memory_stats()
        for iid in ids:
            for kind in ['clean', 'noise', 'adv']:
                key = case, iid, kind
                if key in done: continue
                path = advdir / (iid + '.png') if kind == 'adv' else input_path(args.split, kind, iid)
                result, _ = generate(model, processor, read_image(path))
                row = dict(model='qwen2vl', case=case, image_id=iid, input=kind,
                    **{k: result[k] for k in ['n_tokens', 'hit_cap', 'latency_s', 'text']})
                rows.append(row)
                done.add(key)
                save_csv(args.out, rows)
                if progress is not None:
                    progress(len(rows))
                print(f'{case} {iid} {kind} n={row["n_tokens"]} cap={row["hit_cap"]}', flush=True)
        meta['cases'][case]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(meta_path, meta)
        del model
    if done != expected or len(rows) != len(expected): raise ValueError('Result rows are incomplete')
    meta.update(completed=True, row_count=len(rows), csv_sha256=sha256(args.out))
    write_json(meta_path, meta)
    print(f'EVALUATION_COMPLETE rows={len(rows)} out={args.out}', flush=True)


def validate_measurement_rows(rows,ids,cases):
    expected={(case,iid,kind) for case in cases for iid in ids for kind in ['clean','noise','adv']}
    seen=set()
    for r in rows:
        key=r['case'],r['image_id'],r['input']
        if key not in expected or key in seen or r['model']!='qwen2vl':raise ValueError('Unexpected or duplicate result row')
        if set(r)!=set(EVAL_FIELDS) or any(v is None for v in r.values()):raise ValueError('Invalid result schema')
        length=int(r['n_tokens'])
        if not 1<=length<=5000:raise ValueError('Invalid token count')
        seen.add(key)
    return seen,expected

EVAL_FIELDS=['model','case','image_id','input','n_tokens','ended_eos','output_limit_reached','latency_s','text']

def measure_extended(self,adopted):
        expected_rows=len(self.ids)*len(self.cfg['cases'])*3
        split=self.cfg.get('split','eval')
        paths={};hashes={}
        for iid in self.ids:
            for kind in ['clean','noise','adv']:
                path=self.advdir/(iid+'.png') if kind=='adv' else input_path(split,kind,iid)
                expected=adopted[iid]['selected']['sha256'] if kind=='adv' else self.cfg['input_sha256'][f'{iid}/{kind}']
                if sha256(path)!=expected:raise ValueError('A fixed input changed before measurement')
                paths[(iid,kind)]=path;hashes[f'{iid}/{kind}']=expected
        csv_path=self.csv_path;meta_path=self.csv_path.with_suffix('.json');response_dir=self.run/'responses'
        response_dir.mkdir(exist_ok=True)
        if csv_path.exists():
            if not meta_path.exists():raise ValueError('Existing measurement CSV lacks metadata')
            meta=json.loads(meta_path.read_text())
            if meta['fingerprint']!=self.id or meta['inputs_sha256']!=hashes:raise ValueError('Measurement configuration changed')
            if meta.get('completed') and sha256(csv_path)!=meta['csv_sha256']:raise ValueError('Completed measurement CSV changed')
            with csv_path.open(newline='',encoding='utf8') as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames!=EVAL_FIELDS:raise ValueError('CSV schema mismatch')
                rows=list(reader)
            for existing in rows:
                response_path=response_dir/f'{existing["case"]}-{existing["image_id"]}-{existing["input"]}.json'
                if not response_path.exists():raise ValueError('Existing CSV row lacks its saved response')
                saved=json.loads(response_path.read_text())
                if saved['fingerprint']!=self.id or {k:str(v) for k,v in saved['row'].items()}!=existing:
                    raise ValueError('Existing CSV row differs from the saved response')
        else:
            meta=dict(fingerprint=self.id,config=self.cfg,inputs_sha256=hashes,cases={},completed=False)
            rows=[];write_json(meta_path,meta);save_csv(csv_path,rows,EVAL_FIELDS)
        done,expected=validate_measurement_rows(rows,self.ids,self.cfg['cases'])
        processor=load_processor()
        for case in self.cfg['cases']:
            case_keys={(case,iid,kind) for iid in self.ids for kind in ['clean','noise','adv']}
            if case_keys.issubset(done):continue
            self.update(state='running',stage='measurement',current_case=case,measurement_rows=len(rows))
            gc.collect();torch.cuda.empty_cache();model,info=load_case(case)
            meta['cases'][case]=info;write_json(meta_path,meta)
            generate(model,processor,read_image(input_path(split,'clean',self.ids[0])),{'max_new_tokens':1})
            torch.cuda.reset_peak_memory_stats()
            for iid in self.ids:
                for kind in ['clean','noise','adv']:
                    key=(case,iid,kind)
                    if key in done:continue
                    self.update(current_image=iid,current_input=kind,measurement_rows=len(rows))
                    path=paths[(iid,kind)];saved_path=response_dir/f'{case}-{iid}-{kind}.json'
                    if sha256(path)!=hashes[f'{iid}/{kind}']:raise ValueError('Input changed during measurement')
                    if saved_path.exists():
                        saved=json.loads(saved_path.read_text())
                        if saved['fingerprint']!=self.id or saved['input_sha256']!=sha256(path):raise ValueError('Changed stored response')
                        row=saved['row']
                        if len(saved['tokens'])!=int(row['n_tokens']):raise ValueError('Saved response token count changed')
                        if any(saved['preprocessing'].values()):raise ValueError('Saved response preprocessing mismatch')
                    else:
                        image=read_image(path);pre=check_preprocessing(processor,image)
                        result,tokens=generate(model,processor,image,{'max_new_tokens':5000})
                        row=dict(model='qwen2vl',case=case,image_id=iid,input=kind,
                            n_tokens=result['n_tokens'],ended_eos=result['ended_eos'],output_limit_reached=result['hit_cap'],
                            latency_s=result['latency_s'],text=result['text'])
                        write_json(saved_path,dict(fingerprint=self.id,input_sha256=sha256(path),row=row,tokens=tokens,preprocessing=pre))
                    validate_measurement_rows(rows+[row],self.ids,self.cfg['cases'])
                    rows.append(row);done.add(key);save_csv(csv_path,rows,EVAL_FIELDS)
                    self.update(measurement_rows=len(rows))
                    print(f'MEASUREMENT {len(rows)}/{expected_rows} {case} {iid} {kind} tokens={row["n_tokens"]}',flush=True)
            meta['cases'][case]['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            write_json(meta_path,meta);del model;gc.collect();torch.cuda.empty_cache()
        done,expected=validate_measurement_rows(rows,self.ids,self.cfg['cases'])
        if done!=expected or len(rows)!=expected_rows:raise ValueError('Expected complete paired measurements')
        meta.update(completed=True,row_count=expected_rows,csv_sha256=sha256(csv_path));write_json(meta_path,meta)
        self.update(state='complete',stage='complete',completed=True,measurement_rows=expected_rows,completed_at=now(),csv_sha256=sha256(csv_path))
        print(f'PIPELINE_COMPLETE images={len(self.ids)} rows={expected_rows}',flush=True)


def published_job(split, ids, advdir):
    """Adapt a published immutable image set without changing its files."""
    selected=json.loads((advdir/'selection.json').read_text())
    if selected['split']!=split or selected['source_model']!=settings.MODELS['qwen2vl']['model_id']:
        raise ValueError('Published image set model/split mismatch')
    images=[]
    for item in selected['images']:
        images.append(dict(item,clean_sha256=item['clean_sha256'],
            iters_run=item['iterations_run'],status='complete',numerical_failure=False))
    if {r['image_id'] for r in images}!=set(ids):
        raise ValueError('Published image set does not match the full requested cohort')
    return dict(completed=True,fingerprint=fingerprint(selected),images=images,
        config=dict(split=split,environment=dict(model=dict(revision=selected['source_revision']),
            settings_version=selected['settings_version'])))
