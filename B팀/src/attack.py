"""Official settings v1 Qwen eval generation; retains all 30 PNGs, no ASR analysis."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import numpy as np
from runtime import (settings, extend_answer, pixel_values, generate, eos_ids,
    read_image, input_path, check_preprocessing, raw_pixels, to_image, write_json,
    sha256, image_ids, load_processor, environment, fingerprint, load_case, save_csv, encode)

def objective(model, processor, base, pixels, target, eos):
    extended = extend_answer(base, target)
    extended['pixel_values'] = pixel_values(processor, pixels).to(torch.bfloat16)
    extra = {'logits_to_keep': target.numel() + 1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    out = model(**extended, use_cache=False, **extra)
    logits = out.logits[0, -target.numel()-1:-1].float()
    if logits.shape[0] != target.numel(): raise ValueError('Incorrect answer-logit alignment')
    loss_eos = logits[:, eos].max(dim=-1).values.mean()
    logp = F.log_softmax(logits, dim=-1)
    loss_unc = -logp.mean() - math.log(logp.shape[-1])
    return loss_eos + settings.W_UNC * loss_unc, loss_eos, loss_unc

@torch.no_grad()
def target_tokens(model, processor, image):
    _, ids = generate(model, processor, image, dict(max_new_tokens=settings.TARGET_LEN,
        min_new_tokens=settings.TARGET_LEN, bad_words_ids=[[e] for e in eos_ids(model)]))
    if len(ids) != settings.TARGET_LEN or any(i in eos_ids(model) for i in ids):
        raise ValueError('Long target must contain 128 non-EOS tokens')
    return torch.tensor(ids, device='cuda', dtype=torch.long)

def score(result):
    return int(result['hit_cap']), result['n_tokens']

def attack_one(model, processor, iid, args, outdir):
    clean_path = input_path(args.split, 'clean', iid)
    clean_image = read_image(clean_path)
    preflight = check_preprocessing(processor, clean_image)
    clean, _ = generate(model, processor, clean_image)
    x = raw_pixels(clean_image, 'cuda')
    delta = torch.zeros_like(x, requires_grad=True)
    best_delta = delta.detach().clone()
    best = dict(clean)
    eos = eos_ids(model)
    base = encode(processor, clean_image).to('cuda')
    target = None
    hits = 0
    iterations = 0
    numerical_failure = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f'[{iid}] clean={clean["n_tokens"]} eos={clean["ended_eos"]}', flush=True)
    if clean['n_tokens'] < settings.MAX_NEW_TOKENS:
        for it in range(args.iters):
            if it % settings.REFRESH == 0:
                target = target_tokens(model, processor, to_image(x + delta))
            delta.grad = None
            loss, le, lu = objective(model, processor, base,
                (x + delta).clamp(0, 255), target, eos)
            loss.backward()
            if not torch.isfinite(loss) or delta.grad is None or not torch.isfinite(delta.grad).all():
                numerical_failure = True
                raise FloatingPointError(f'{iid}: non-finite gradient/loss at update {it+1}')
            loss_values = [float(loss.detach()), float(le.detach()), float(lu.detach())]
            with torch.no_grad():
                delta.sub_(settings.STEP * delta.grad.sign())
                delta.clamp_(-settings.EPS, settings.EPS)
                delta.copy_((x + delta).clamp(0, 255) - x)
            iterations = it + 1
            del loss, le, lu
            if it % settings.REFRESH == 0 or iterations == args.iters:
                natural, _ = generate(model, processor, to_image(x + delta))
                if score(natural) > score(best):
                    best = natural
                    best_delta = delta.detach().clone()
                hits = hits + 1 if natural['hit_cap'] else 0
                print(f'[{iid}] iter={iterations} len={natural["n_tokens"]} cap={natural["hit_cap"]} loss={loss_values[0]:.4f} seconds={time.perf_counter()-started:.1f}', flush=True)
                if hits >= 2:
                    break
    png = outdir / (iid + '.png')
    temporary = outdir / (iid + '.png.tmp')
    to_image(x + best_delta).save(temporary, format='PNG')
    os.replace(temporary, png)
    saved_image = read_image(png)
    linf = int(np.abs(np.asarray(saved_image).astype(np.int16) - np.asarray(clean_image).astype(np.int16)).max())
    if linf > settings.EPS: raise ValueError(f'Saved PNG exceeds epsilon: {linf}')
    saved_preflight = check_preprocessing(processor, saved_image)
    saved, _ = generate(model, processor, saved_image)
    if (saved['n_tokens'], saved['hit_cap']) != (best['n_tokens'], best['hit_cap']):
        raise ValueError('Saved PNG generation differs from the selected best image')
    record = dict(image_id=iid, status='complete', iters_run=iterations,
        seconds=time.perf_counter()-started, start_len=clean['n_tokens'],
        best_len=best['n_tokens'], saved_len=saved['n_tokens'], linf=linf,
        numerical_failure=numerical_failure, clean=clean, saved=saved,
        clean_sha256=sha256(clean_path), sha256=sha256(png),
        preprocessing=preflight, saved_preprocessing=saved_preflight,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(outdir / (iid + '.json'), record)
    print(f'[{iid}] SAVED len={saved["n_tokens"]} linf={linf}', flush=True)
    return record

def common_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'eval'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--iters', type=int, default=settings.ATTACK_ITERS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.iters <= settings.ATTACK_MAX_ITERS:
        parser.error('Iteration count must be 1..500')
    if args.split == 'eval' and args.iters < settings.ATTACK_ITERS:
        parser.error('Eval generation requires at least the common 300 iterations')
    if settings.W_DIV != 0: raise NotImplementedError('This protocol uses W_DIV=0')
    ids = image_ids(args.split, args.limit)
    processor = load_processor()
    env = environment()
    config = dict(split=args.split, image_ids=ids, iters=args.iters, eps=settings.EPS,
        step=settings.STEP, target_len=settings.TARGET_LEN, refresh=settings.REFRESH,
        w_unc=settings.W_UNC, w_div=settings.W_DIV, gen_kwargs=settings.GEN_KWARGS,
        environment=env, clean_sha256={i: sha256(input_path(args.split, 'clean', i)) for i in ids})
    run_id = fingerprint(config)
    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    job_path = outdir / 'job.json'
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if job['fingerprint'] != run_id: raise ValueError('Use a new output folder for a different configuration/code version')
    else:
        job = dict(fingerprint=run_id, config=config, completed=False, images=[])
        write_json(job_path, job)
    model = None
    records = []
    for iid in ids:
        record_path = outdir / (iid + '.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['status'] == 'complete' and record['sha256'] == sha256(outdir / (iid + '.png')):
                records.append(record)
                continue
            raise ValueError(f'Changed completed attack image: {iid}')
        if model is None:
            torch.manual_seed(settings.NOISE_SEED)
            model, info = load_case('ref', attack=True)
            job['model_case'] = info
            write_json(job_path, job)
        record = attack_one(model, processor, iid, args, outdir)
        records.append(record)
        job['images'] = records
        write_json(job_path, job)
    fields = ['image_id', 'iters_run', 'seconds', 'start_len', 'best_len', 'saved_len', 'linf', 'numerical_failure']
    save_csv(outdir / 'attack_log.csv', [{k: r[k] for k in fields} for r in records], fields)
    (outdir / 'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {r["image_id"]}.png\n' for r in records))
    job.update(completed=True, images=records)
    write_json(job_path, job)
    print(f'ATTACK_COMPLETE images={len(records)} out={outdir}', flush=True)


def measured(model, processor, path, cap):
    image = read_image(path)
    pre = check_preprocessing(processor, image)
    result, tokens = generate(model, processor, image, {'max_new_tokens':cap})
    result['output_limit_reached'] = result.pop('hit_cap')
    result['tokens'] = tokens
    result['preprocessing'] = pre
    return result

def csv_row(iid, stage, iteration, result, path, rank='', difference='', within=''):
    return dict(image_id=iid,stage=stage,iteration=iteration,rank=rank,
        **{k:result[k] for k in ['n_tokens','ended_eos','output_limit_reached','latency_s','text']},
        relative_token_difference=difference,within_tolerance=within,png_sha256=sha256(path))

FIELDS = ['image_id','stage','iteration','rank','n_tokens','ended_eos',
    'output_limit_reached','latency_s','relative_token_difference','within_tolerance',
    'png_sha256','text']

"""Checkpointed measurement-only attack generation for the agreed protocol."""
import copy,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from runtime import (settings,input_path,read_image,raw_pixels,to_image,encode,
    eos_ids,sha256,write_json,save_csv)

def save_state(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path)

def capture_rng():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(value):
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])

def capture_rope(model):
    return {name:module.rope_deltas.detach().cpu().clone()
        for name,module in model.named_modules()
        if isinstance(getattr(module,'rope_deltas',None),torch.Tensor)}

def restore_rope(model,values):
    for name,value in values.items():model.get_submodule(name).rope_deltas=value.to('cuda')

def ranked_candidates(candidates):
    return sorted(candidates,key=lambda c:(-c['measurement']['n_tokens'],c['iteration']))

def relative_difference(first,again):
    if first<1:raise ValueError('Initial token count must be positive')
    return abs(again-first)/first

def verify_record(record):
    if record['iterations_run']!=500 or len(record['candidates'])!=25:
        raise ValueError('Expected 500 updates and 25 checkpoints')
    if [c['iteration'] for c in record['candidates']]!=list(range(20,501,20)):
        raise ValueError('Unexpected checkpoint sequence')
    for candidate in record['candidates']:
        if sha256(candidate['path'])!=candidate['sha256'] or candidate['linf']>8:
            raise ValueError('Checkpoint image changed or exceeded pixel bound')
    selected=record['selected']
    if not selected:raise ValueError('No candidate passed the 20% recheck; manual review required')
    if sha256(selected['path'])!=selected['sha256']:
        raise ValueError('Selected image changed')
    if relative_difference(selected['initial_n_tokens'],selected['recheck_n_tokens'])>.20:
        raise ValueError('Selected image did not meet recheck tolerance')

def generate_one(model,processor,iid,out,split,progress):
    folder=out/iid;folder.mkdir(parents=True,exist_ok=True)
    record_path=folder/'record.json';state_path=folder/'checkpoint.pt'
    clean_path=input_path(split,'clean',iid);noise_path=input_path(split,'noise',iid)
    clean_image=read_image(clean_path);x=raw_pixels(clean_image,'cuda')
    base=encode(processor,clean_image).to('cuda');eos=eos_ids(model)
    active_start=time.perf_counter();prior_seconds=0.
    state=None
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['image_id']!=iid or state['clean_sha256']!=sha256(clean_path):
            raise ValueError('Resume state input does not match')
        record=state['record'];rows=state['rows'];prior_seconds=state['active_seconds']
        delta=state['delta'].to('cuda').requires_grad_(True)
        target=state['target'].to('cuda') if state['target'] is not None else None
        restore_rng(state['rng']);restore_rope(model,state.get('rope',{}))
        for c in record['candidates']:
            if sha256(c['path'])!=c['sha256']:raise ValueError('Changed resume candidate')
        print(f'RESUME {iid} iteration={record["iterations_run"]}',flush=True)
    else:
        torch.manual_seed(settings.NOISE_SEED)
        delta=torch.zeros_like(x,requires_grad=True);target=None
        record=dict(image_id=iid,status='generating',iterations_run=0,baselines={},
            candidates=[],rechecks=[],selected=None,clean_sha256=sha256(clean_path),
            noise_sha256=sha256(noise_path))
        rows=[]
        for kind,path in [('clean',clean_path),('noise',noise_path)]:
            result=measured(model,processor,path,5000)
            record['baselines'][kind]=result
            rows.append(csv_row(iid,kind,0,result,path))
            print(f'BASELINE {iid} {kind} tokens={result["n_tokens"]}',flush=True)
    def commit():
        active=prior_seconds+time.perf_counter()-active_start
        save_state(state_path,dict(image_id=iid,clean_sha256=sha256(clean_path),
            record=copy.deepcopy(record),rows=copy.deepcopy(rows),active_seconds=active,
            delta=delta.detach().cpu().clone(),target=target.detach().cpu().clone() if target is not None else None,
            rng=capture_rng(),rope=capture_rope(model)))
        write_json(record_path,record);save_csv(folder/'measurements.csv',rows,FIELDS)
    if state is None:commit()
    start_attack=time.perf_counter();previous_attack=record.get('attack_seconds_so_far',0.)
    for it in range(record['iterations_run'],500):
        if it%settings.REFRESH==0:target=target_tokens(model,processor,to_image(x+delta))
        delta.grad=None
        loss,le,lu=objective(model,processor,base,(x+delta).clamp(0,255),target,eos)
        loss.backward()
        if not torch.isfinite(loss) or delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f'Nonfinite image gradient for {iid} at iteration {it+1}')
        value=float(loss.detach())
        with torch.no_grad():
            delta.sub_(settings.STEP*delta.grad.sign());delta.clamp_(-settings.EPS,settings.EPS)
            delta.copy_((x+delta).clamp(0,255)-x)
        del loss,le,lu
        iteration=it+1
        if iteration%20:continue
        path=folder/f'iter-{iteration:04d}.png'
        image=to_image(x+delta)
        if path.exists() and path.read_bytes():
            if not np.array_equal(np.asarray(read_image(path)),np.asarray(image)):
                raise ValueError('An uncommitted checkpoint PNG differs on replay; manual review required')
        else:
            temporary=path.with_name(path.name+'.tmp')
            image.save(temporary,format='PNG');os.replace(temporary,path)
        linf=int(np.abs(np.asarray(read_image(path)).astype(np.int16)-np.asarray(clean_image).astype(np.int16)).max())
        if linf>8:raise ValueError('Pixel bound exceeded')
        result=measured(model,processor,path,5000)
        record['candidates'].append(dict(iteration=iteration,path=str(path),sha256=sha256(path),linf=linf,
            objective=value,measurement=result))
        rows.append(csv_row(iid,'checkpoint',iteration,result,path))
        record['iterations_run']=iteration
        record['attack_seconds_so_far']=previous_attack+time.perf_counter()-start_attack
        commit();progress(iid,iteration)
        print(f'CHECKPOINT {iid} update={iteration}/500 tokens={result["n_tokens"]}',flush=True)
    record['attack_seconds']=record.get('attack_seconds_so_far',0.)
    ranked=ranked_candidates(record['candidates'])
    if not record['selected']:
        for rank,candidate in enumerate(ranked,1):
            if rank<=len(record['rechecks']):continue
            path=Path(candidate['path'])
            if sha256(path)!=candidate['sha256']:raise ValueError('Candidate changed before recheck')
            again=measured(model,processor,path,5000)
            first=candidate['measurement']['n_tokens'];difference=relative_difference(first,again['n_tokens'])
            within=difference<=.20
            record['rechecks'].append(dict(rank=rank,iteration=candidate['iteration'],initial_n_tokens=first,
                relative_token_difference=difference,within_tolerance=within,measurement=again))
            rows.append(csv_row(iid,'recheck',candidate['iteration'],again,path,rank,difference,within))
            if within:
                selected=out/(iid+'.png')
                if selected.exists() and sha256(selected)!=candidate['sha256']:
                    raise ValueError('Existing selected image differs; manual review required')
                temporary=selected.with_name(selected.name+'.tmp')
                shutil.copyfile(path,temporary);os.replace(temporary,selected)
                record['selected']=dict(path=str(selected),sha256=sha256(selected),iteration=candidate['iteration'],
                    rank=rank,initial_n_tokens=first,recheck_n_tokens=again['n_tokens'],
                    relative_token_difference=difference,linf=candidate['linf'])
            commit()
            print(f'RECHECK {iid} rank={rank} first={first} again={again["n_tokens"]} difference={difference:.4f} adopted={within}',flush=True)
            if within:break
    if not record['selected']:
        record['status']='awaiting_review';commit()
        raise RuntimeError(f'No candidate met the 20% rule for {iid}; records retained')
    verify_record(record)
    record.update(status='adopted',image_seconds=prior_seconds+time.perf_counter()-active_start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    commit()
    print(f'IMAGE_COMPLETE {iid} iteration={record["selected"]["iteration"]} tokens={record["selected"]["recheck_n_tokens"]}',flush=True)
    return record
