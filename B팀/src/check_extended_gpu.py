"""GPU-server readiness check; dev inference only, never starts the eval suite."""
import argparse
import csv
import gc
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-adv-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # fcntl is intentionally server-only; no local validation fallback.
    import fcntl
    import torch
    import runtime as rt
    import evaluate
    import pipeline
    from case_config import COMPONENTS, PROTOCOLS, component_cases
    from components import make_plan, server_blockers
    from transformers.utils import logging as hf_logging

    logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
    warnings.filterwarnings('once', message='MatMul8bitLt:.*')
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    blockers = server_blockers(rt.ROOT)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run this check on the bf16 CUDA experiment server')
    plan = make_plan('extended_v1', 'both')
    cap = PROTOCOLS['extended_v1']['cap']
    assert cap == 5000 and plan['total_expected_rows'] == 2160
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol='extended_v1', max_new_tokens=cap, split='dev',
        eval_inference_started=False, main_experiment_started=False, completed=False)

    with (rt.ROOT / '.qwen-dev.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline.configure_gpu()
        pipeline.validate_protocol()
        ids = rt.image_ids('eval')
        advdir = Path(plan['bundle']['image_set_id'])
        advdir = Path(__file__).resolve().parents[1] / 'adv' / advdir
        evaluate.verify_inputs('eval', ids, advdir)
        dev_ids = rt.image_ids('dev')
        job, dev_hashes = evaluate.verify_inputs('dev', dev_ids, args.dev_adv_dir.resolve())
        processor = rt.load_processor()
        preprocessing = {}
        for iid in ids:
            for kind in ('clean', 'noise', 'adv'):
                path = advdir / (iid + '.png') if kind == 'adv' else rt.input_path('eval', kind, iid)
                preprocessing[f'{iid}/{kind}'] = rt.check_preprocessing(processor, rt.read_image(path))
        summary.update(environment=rt.environment(), plan=plan, preprocessing=preprocessing)
        rt.write_json(out / 'readiness.json', summary)
        print('EXTENDED_INPUTS_PASS images=30 inputs=90 preprocessing_max_abs=0', flush=True)
        iid = dev_ids[0]
        samples = {kind: rt.read_image(rt.input_path('dev', kind, iid)) for kind in ('clean', 'noise')}
        checks = []
        try:
            for component in COMPONENTS:
                for case in component_cases(component):
                    model = None
                    try:
                        model, info = rt.load_case(case)
                        generated = {}
                        for kind, image in samples.items():
                            result, tokens = rt.generate(model, processor, image, {'max_new_tokens': cap})
                            assert 1 <= len(tokens) <= cap
                            ended = any(token in info['eos_token_ids'] for token in tokens)
                            assert result['ended_eos'] == ended
                            assert result['hit_cap'] == (len(tokens) == cap and not ended)
                            generated[kind] = dict(result, tokens=tokens)
                        checks.append(dict(component=component, case=case, model=info, generation=generated))
                        summary['model_checks'] = checks
                        rt.write_json(out / 'readiness.json', summary)
                        print(f'EXTENDED_MODEL_PASS {len(checks)}/24 {component} {case} cap={cap}', flush=True)
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()

            # Exercise the production extended writer with real dev clean/noise/adv.
            # The controlled interruption happens after one saved row; resume must
            # retain it, write the other two rows, and skip all rows on a third call.
            run = out / 'dev-pipeline'
            run.mkdir(exist_ok=True)
            cfg = dict(split='dev', image_ids=[iid], cases=['ref'], max_new_tokens=cap,
                input_sha256=dev_hashes, environment=summary['environment'],
                dev_attack_sha256=rt.sha256(args.dev_adv_dir / (iid + '.png')))
            status = dict(fingerprint=rt.fingerprint(cfg))
            interrupted = {'pending': not (run / 'responses.csv').exists()}

            class ProbeInterruption(Exception):
                pass

            def update(**values):
                status.update(values)
                rt.write_json(run / 'status.json', status)
                if interrupted['pending'] and values.get('measurement_rows') == 1:
                    interrupted['pending'] = False
                    raise ProbeInterruption('Readiness check: resume after first saved dev row')

            runner = SimpleNamespace(ids=[iid], advdir=args.dev_adv_dir.resolve(), run=run,
                csv_path=run / 'responses.csv', cfg=cfg, id=rt.fingerprint(cfg), update=update)
            adopted = {r['image_id']: dict(selected=dict(sha256=r['sha256'])) for r in job['images']}
            try:
                evaluate.measure_extended(runner, adopted)
            except ProbeInterruption:
                gc.collect()
                torch.cuda.empty_cache()
                print('EXTENDED_DEV_INTERRUPTION_PASS saved_rows=1', flush=True)
            first = run / 'responses' / f'ref-{iid}-clean.json'
            first_hash = rt.sha256(first)
            evaluate.measure_extended(runner, adopted)
            assert rt.sha256(first) == first_hash
            csv_hash = rt.sha256(runner.csv_path)
            evaluate.measure_extended(runner, adopted)
            assert rt.sha256(runner.csv_path) == csv_hash
            with runner.csv_path.open(newline='', encoding='utf8') as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames == evaluate.EVAL_FIELDS
                rows = list(reader)
            assert len(rows) == 3 and {r['input'] for r in rows} == {'clean', 'noise', 'adv'}
            metadata = json.loads(runner.csv_path.with_suffix('.json').read_text())
            assert metadata['completed'] and metadata['row_count'] == 3
            assert metadata['csv_sha256'] == csv_hash
            summary.update(completed=True, model_conditions_passed=len(checks),
                dev_generation_count=sum(len(c['generation']) for c in checks),
                dev_pipeline=dict(rows=rows, resume_preserved_first_response=True,
                    completed_rerun_preserved_csv=True, csv_sha256=csv_hash))
            rt.write_json(out / 'readiness.json', summary)
            print('EXTENDED_GPU_READY models=24 generations=48 dev_pipeline_rows=3 cap=5000 eval_inference_started=False', flush=True)
        except BaseException as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            rt.write_json(out / 'readiness.json', summary)
            raise


if __name__ == '__main__':
    main()
