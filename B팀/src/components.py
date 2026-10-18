"""Prepare/check/run C, L and VCL measurements using fixed published PNGs.

Default --plan needs only Python's standard library and never starts inference.
--check checks the experiment server without loading a model. Only --run measures.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from case_config import (BACKENDS, DEFAULT_BACKEND, COMPONENTS, MODEL_ID, MODEL_REVISION, PROTOCOLS,
                         component_cases, expected_census, profile, run_name)

BTEAM = Path(__file__).resolve().parents[1]
REPO = BTEAM.parent
INPUTS = ('clean', 'noise', 'adv')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_bundle(protocol):
    """Check the full fixed cohort, PNG bytes and cross-linked input hashes."""
    spec = PROTOCOLS[protocol]
    directory = BTEAM / 'adv' / spec['image_set']
    selection_path = directory / 'selection.json'
    selected = read_json(selection_path)
    expected = dict(split='eval', source_model=MODEL_ID, source_revision=MODEL_REVISION,
                    settings_version='v1', attack_protocol=protocol,
                    image_set_id=spec['image_set'])
    for key, value in expected.items():
        if selected.get(key) != value:
            raise ValueError(f'Fixed image-set mismatch: {key}')
    with (REPO / '공통' / 'image_list.csv').open(encoding='utf-8', newline='') as stream:
        ids = [f"{int(row['coco_id']):012d}" for row in csv.DictReader(stream)
               if row['split'] == 'eval']
    images = selected['images']
    if len(ids) != 30 or len(set(ids)) != 30 or len(images) != 30:
        raise ValueError('Expected exactly 30 distinct eval images')
    if len({im['image_id'] for im in images}) != 30 or {im['image_id'] for im in images} != set(ids):
        raise ValueError('Attack set differs from the complete eval cohort')
    sums = {}
    for line in (directory / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split()
        if name in sums:
            raise ValueError('Duplicate SHA256SUMS entry')
        sums[name] = digest
    if set(sums) != {iid + '.png' for iid in ids}:
        raise ValueError('Attack hash list differs from the complete eval cohort')
    manifest_path = BTEAM / 'results' / 'input-manifest.json'
    manifest = read_json(manifest_path)
    refs = {(r['split'], r['input'], r['image_id']): r['sha256']
            for r in manifest['records']}
    for im in images:
        iid = im['image_id']
        if im['path'] != iid + '.png' or not 0 <= im['linf'] <= 8:
            raise ValueError(f'Invalid saved attack record: {iid}')
        path = directory / im['path']
        if sha(path) != im['sha256'] or sums[path.name] != im['sha256']:
            raise ValueError(f'Fixed attack PNG changed: {iid}')
        with path.open('rb') as stream:
            header = stream.read(24)
        if (header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR'
                or struct.unpack('>II', header[16:24]) != (448, 448)):
            raise ValueError(f'Expected a 448x448 PNG: {iid}')
        for kind in ('clean', 'noise'):
            if im[kind + '_sha256'] != refs['eval', kind, iid]:
                raise ValueError(f'Fixed input manifest differs from the attack source: {iid}/{kind}')
    return dict(image_set_id=directory.name, selection_sha256=sha(selection_path),
                input_manifest_sha256=sha(manifest_path), image_ids=ids,
                verified_attack_pngs=len(images))


def make_plan(protocol='common_v1', backend=DEFAULT_BACKEND, repeat='r01', components=None):
    bundle = validate_bundle(protocol)
    experiments = []
    for component in components or COMPONENTS:
        cases = component_cases(component, backend)
        name = run_name(component, protocol, backend, repeat)
        experiments.append(dict(experiment_id=name, component=component,
            quantization_profile=profile(component, backend), cases=cases,
            expected_census={case: expected_census(case) for case in cases},
            n_images=30, inputs=list(INPUTS), expected_rows=30 * len(cases) * 3,
            output_csv=f'B팀/results/{name}.csv'))
    return dict(setup_version='components_v1', model_id=MODEL_ID,
        model_revision=MODEL_REVISION, protocol=protocol,
        max_new_tokens=PROTOCOLS[protocol]['cap'], backend=backend,
        prompt='Describe this image in one sentence.',
        generation=dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
            max_new_tokens=PROTOCOLS[protocol]['cap'], temperature=None, top_p=None, top_k=None,
            eos_token_ids=[151645,151643]),
        image_size=[448, 448], seed=2026, attack_regeneration=False,
        bundle=bundle, experiments=experiments,
        total_expected_rows=sum(e['expected_rows'] for e in experiments),
        inference_started=False)


def pipeline_command(plan, experiment, action):
    return [sys.executable, str(BTEAM / 'src' / 'pipeline.py'), plan['protocol'], action,
            '--measure-only', '--component', experiment['component'],
            '--backend', plan['backend']]


def run_environment(plan, experiment, root):
    return {**os.environ, 'ONDEVICE_AI_ROOT': str(root), 'ONDEVICE_AI_REPO': str(REPO),
            'BTEAM_RUN_NAME': experiment['experiment_id'],
            'BTEAM_IMAGE_SET': plan['bundle']['image_set_id'],
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUTF8': '1'}


def server_blockers(root):
    blockers = []
    required = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
                'transformers': '5.19.0', 'bitsandbytes': '0.50.0',
                'accelerate': '1.15.0', 'pillow': '11.2.1', 'numpy': '2.2.6'}
    for package, version in required.items():
        try:
            actual = importlib.metadata.version(package)
            if actual != version:
                blockers.append(f'Package {package}: expected {version}, found {actual}')
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f'Missing package: {package}=={version}')
    model_path = root / 'model.json'
    if not model_path.exists():
        blockers.append(f'Missing model metadata: {model_path}')
    else:
        model = read_json(model_path)
        if model.get('model_id') != MODEL_ID or model.get('revision') != MODEL_REVISION:
            blockers.append('Cached model ID/revision differs from the fixed attack source')
        snapshot = Path(model.get('snapshot_path', ''))
        index_path = snapshot / 'model.safetensors.index.json'
        if not index_path.is_file():
            blockers.append(f'Missing model cache: {snapshot}')
        else:
            index = read_json(index_path)
            expected = {'model-00001-of-00002.safetensors': 3988609112,
                        'model-00002-of-00002.safetensors': 429441656}
            if set(index['weight_map'].values()) != set(expected):
                blockers.append('Model shard index differs from the verified revision')
            for name, size in expected.items():
                path = snapshot / name
                if not path.is_file() or path.stat().st_size != size:
                    blockers.append(f'Missing/changed cached model shard: {path}')
    if not (root / 'data' / 'manifest.json').is_file():
        blockers.append(f'Missing server input manifest: {root / "data" / "manifest.json"}')
    return blockers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', nargs='?', choices=PROTOCOLS, default='common_v1')
    parser.add_argument('--backend', choices=BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument('--run-number', default='r01')
    parser.add_argument('--component', choices=['all', *COMPONENTS], default='all')
    parser.add_argument('--root', type=Path,
                        default=Path(os.environ.get('ONDEVICE_AI_ROOT', Path.home() / 'ondevice-ai')))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan', action='store_true')
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        components = None if args.component == 'all' else [args.component]
        plan = make_plan(args.protocol, args.backend, args.run_number, components)
        if not (args.check or args.run):
            print(json.dumps(dict(state='plan_verified', runtime_checked=False, **plan),
                             ensure_ascii=False, indent=2))
            return 0
        root = args.root.resolve()
        blockers = server_blockers(root)
        if blockers:
            print(json.dumps(dict(state='server_not_ready', blockers=blockers,
                                 inference_started=False), ensure_ascii=False, indent=2))
            return 2
        # This launches only --check: no model loading, generation, or result writes.
        for experiment in plan['experiments']:
            subprocess.run(pipeline_command(plan, experiment, '--check'), check=True,
                           cwd=BTEAM, env=run_environment(plan, experiment, root))
        if args.check:
            print('COMPONENT_SUITE_READY inference_started=False', flush=True)
            return 0
        for experiment in plan['experiments']:
            print('COMPONENT_START', experiment['experiment_id'], flush=True)
            env = run_environment(plan, experiment, root)
            subprocess.run(pipeline_command(plan, experiment, '--run'),
                           check=True, cwd=BTEAM, env=env)
            subprocess.run([sys.executable, str(BTEAM / 'src' / 'report.py'),
                            experiment['experiment_id']], check=True, cwd=BTEAM, env=env)
        print('COMPONENT_SUITE_COMPLETE', flush=True)
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'COMPONENT_SETUP_ERROR: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
