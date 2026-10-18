"""Component setup contracts and synthetic orchestration; no torch/GPU required."""
import ast
import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import case_config as cases
import components
import report


def extract_functions(filename, names, namespace):
    """Execute actual orchestration functions with explicit fake model dependencies."""
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class PlanTests(unittest.TestCase):
    def test_complete_cohorts_both_protocols_and_all_backends(self):
        for protocol, cap in [('common_v1', 512), ('extended_v1', 5000)]:
            for backend, rows in [('rtn', 540), ('bnb', 270), ('both', 720)]:
                plan = components.make_plan(protocol, backend)
                self.assertEqual(plan['max_new_tokens'], cap)
                self.assertEqual(plan['bundle']['verified_attack_pngs'], 30)
                self.assertEqual(plan['total_expected_rows'], rows * 3)
                self.assertFalse(plan['attack_regeneration'])
                self.assertEqual([e['component'] for e in plan['experiments']], ['connector', 'language', 'vcl'])
                self.assertEqual({e['expected_rows'] for e in plan['experiments']}, {rows})
                self.assertEqual(len({e['experiment_id'] for e in plan['experiments']}), 3)

    def test_head_cases_and_unknown_bit_widths_are_rejected(self):
        for case in ['rtn-H-4', 'rtn-VCLH-4', 'bnb-C-3', 'rtn-L-7', 'rtn-C-2', 'rtn-VC-4']:
            with self.subTest(case=case), self.assertRaises(ValueError):
                cases.parse_case(case)
        self.assertEqual(cases.parse_case('rtn-V-6')['bits'], 6)
        self.assertEqual(cases.parse_case('rtn-L-6')['bits'], 6)

    def test_defaults_match_all_seven_vision_quantization_conditions(self):
        for component,target in [('connector','C'),('language','L'),('vcl','VCL')]:
            expected=['ref']+[f'rtn-{target}-{bit}' for bit in [8,6,5,4,3]]+[f'bnb-{target}-{bit}' for bit in [8,4]]
            self.assertEqual(cases.component_cases(component),expected)
        plan=components.make_plan()
        self.assertEqual(plan['backend'],'both')
        self.assertEqual(plan['total_expected_rows'],2160)
        self.assertEqual({e['expected_rows'] for e in plan['experiments']},{720})

    def test_check_never_dispatches_measurement_or_report(self):
        with patch.object(components, 'server_blockers', return_value=[]), \
             patch.object(components.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(components.main(['--check']), 0)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertIn('--check', call.args[0])
            self.assertNotIn('--run', call.args[0])
            self.assertIn('--measure-only', call.args[0])

    def test_failed_preflight_stops_before_any_measurement(self):
        error = subprocess.CalledProcessError(2, 'synthetic-check')
        with patch.object(components, 'server_blockers', return_value=[]), \
             patch.object(components.subprocess, 'run', side_effect=error) as run, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(components.main(['--run']), 2)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn('--run', run.call_args.args[0])

    def test_suite_dispatches_each_run_with_own_name_and_fixed_set(self):
        with patch.object(components, 'server_blockers', return_value=[]), \
             patch.object(components.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(components.main(['extended_v1', '--run', '--backend', 'bnb']), 0)
        self.assertEqual(run.call_count, 9)  # Three checks, then three measurement/report pairs.
        measurements = [c for c in run.call_args_list if '--run' in c.args[0]]
        self.assertEqual(len(measurements), 3)
        self.assertEqual(len({c.kwargs['env']['BTEAM_RUN_NAME'] for c in measurements}), 3)
        self.assertEqual({c.kwargs['env']['BTEAM_IMAGE_SET'] for c in measurements},
                         {'qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01'})


class FixedPixelTests(unittest.TestCase):
    def checker(self):
        return extract_functions('runtime.py', {'verify_fixed_pixels'}, dict(
            np=np, settings=SimpleNamespace(NOISE_SEED=2026, EPS=8)))['verify_fixed_pixels']

    def test_bounded_noise_with_wrong_pattern_is_rejected(self):
        clean = np.full((3, 3, 3), 128, dtype=np.uint8)
        signs = np.random.default_rng([2026, 632]).choice([-1, 1], size=clean.shape)
        noise = (clean.astype(np.int16) + 8 * signs).astype(np.uint8)
        self.assertIsNone(self.checker()('000000000632', clean, noise))
        noise[1, 1, 1] = clean[1, 1, 1]
        self.assertEqual(int(np.abs(noise.astype(np.int16) - clean).max()), 8)
        with self.assertRaisesRegex(ValueError, 'seed/pattern'):
            self.checker()('000000000632', clean, noise)

    def test_saved_attack_pixels_are_checked_independently_of_metadata(self):
        clean = np.full((3, 3, 3), 128, dtype=np.uint8)
        signs = np.random.default_rng([2026, 632]).choice([-1, 1], size=clean.shape)
        noise = (clean.astype(np.int16) + 8 * signs).astype(np.uint8)
        adv = clean + 8
        self.assertEqual(self.checker()('000000000632', clean, noise, adv), 8)
        adv[1, 1, 1] = 137
        with self.assertRaisesRegex(ValueError, 'Actual attack PNG exceeded epsilon'):
            self.checker()('000000000632', clean, noise, adv)


class FakeWeight:
    dtype = 'bfloat16'
    _version = 0
    def data_ptr(self):
        return id(self)


class FakeModel:
    def __init__(self):
        self.embedding = SimpleNamespace(weight=FakeWeight())
        self.head = SimpleNamespace(weight=self.embedding.weight)
        self.targets = ''
    def get_input_embeddings(self): return self.embedding
    def get_output_embeddings(self): return self.head
    def get_memory_footprint(self): return 1000
    def to(self, device): return self
    def eval(self): return self
    def requires_grad_(self, value): return self


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, damage=False, drift=False):
        parts = {'V': 'model.visual.blocks', 'C': 'model.visual.merger',
                 'L': 'model.language_model', 'H': 'lm_head'}
        calls = []
        def load(path, **kwargs):
            calls.append(kwargs)
            model = FakeModel()
            cfg = kwargs.get('quantization_config')
            if cfg is not None:
                model.targets = ''.join(p for p, prefix in parts.items()
                                        if prefix not in cfg.llm_int8_skip_modules)
            return model
        def apply(model, parts, targets, bits, group):
            self.assertEqual(group, 64)
            model.targets = targets
            calls.append(dict(rtn_targets=targets, bits=bits))
            if damage:
                model.head.weight = FakeWeight()
            return {p: 1.0 for p in targets}
        def census(model, parts):
            totals = {'V': 128, 'C': 2, 'L': 196, 'H': 1}
            if drift: totals['L'] = 195
            return {p: [n if p in model.targets else 0, n] for p, n in totals.items()}
        def check(count, targets, names):
            for p, (n, total) in count.items():
                self.assertEqual(n, total if p in targets else 0)
        namespace = dict(parse_case=cases.parse_case, expected_census=cases.expected_census,
            torch=SimpleNamespace(bfloat16='bfloat16', cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True)),
            settings=SimpleNamespace(QUANT_GROUP=64, PART_NAMES={p:p for p in parts}),
            SPEC={'parts': parts}, model_metadata=lambda: {'snapshot_path': 'synthetic'},
            BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load),
            quant_rtn=SimpleNamespace(apply_rtn=apply, census=census, check_census=check),
            eos_ids=lambda model: [151645, 151643])
        return extract_functions('runtime.py', {'protected_weights', 'load_case'}, namespace), calls

    def test_every_component_backend_selects_only_requested_parts(self):
        for component, target in [('connector', 'C'), ('language', 'L'), ('vcl', 'VCL')]:
            for case in cases.component_cases(component, 'both'):
                with self.subTest(case=case):
                    ns, calls = self.make_runtime()
                    model, info = ns['load_case'](case)
                    self.assertEqual(model.targets, '' if case == 'ref' else target)
                    self.assertIs(model.head.weight, model.embedding.weight)
                    self.assertEqual(info['census']['H'], [0, 1])
                    self.assertTrue(info['protected_weights']['embedding_head_tied'])
                    if case.startswith('bnb'):
                        skip = calls[0]['quantization_config'].llm_int8_skip_modules
                        self.assertIn('lm_head', skip)
                        for part in 'VCL':
                            prefix = ns['SPEC']['parts'][part]
                            self.assertEqual(prefix in skip, part not in target)

    def test_changed_head_tie_or_model_structure_stops_loading(self):
        for options in [dict(damage=True), dict(drift=True)]:
            ns, _ = self.make_runtime(**options)
            with self.assertRaises(ValueError): ns['load_case']('rtn-VCL-4')

    def test_attack_generation_rejects_quantized_case_before_loading(self):
        ns, calls = self.make_runtime()
        with self.assertRaises(ValueError): ns['load_case']('rtn-C-4', attack=True)
        self.assertEqual(calls, [])


class ReportTests(unittest.TestCase):
    def test_component_reports_use_actual_cases_and_keep_manual_description(self):
        for backend in ['rtn', 'bnb', 'both']:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(dir=SRC.parent) as name:
                root = Path(name).resolve(); self.assertTrue(root.is_relative_to(SRC.parent))
                label = 'synthetic'; image_set = 'fixed'
                (root/'results').mkdir(); (root/'adv'/image_set).mkdir(parents=True)
                output = root/'reports'/label; output.mkdir(parents=True)
                selected = [{'image_id': f'{i:012d}', 'saved_tokens': 512} for i in range(30)]
                (root/'adv'/image_set/'selection.json').write_text(json.dumps({'images': selected}), encoding='utf-8')
                plan = cases.component_cases('vcl', backend)
                rows = [dict(model='qwen2vl',case=c,image_id=im['image_id'],input=k,
                             n_tokens=20,hit_cap=False,latency_s=.1,text='synthetic')
                        for c in plan for im in selected for k in components.INPUTS]
                path = root/'results'/(label+'.csv')
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                meta=dict(completed=True,csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    config={'cases':plan,'quantization_profile':cases.profile('vcl',backend),
                            'gen_kwargs':{'max_new_tokens':512}})
                path.with_suffix('.json').write_text(json.dumps(meta),encoding='utf-8')
                (output/'run_manifest.json').write_text(json.dumps(dict(image_set_id=image_set,published_record=False)),encoding='utf-8')
                if backend=='rtn':
                    (output/'README.md').write_text('# Manual purpose\n\n<!-- measurements:start -->\nold\n<!-- measurements:end -->\n',encoding='utf-8')
                with patch.object(report,'B',root), contextlib.redirect_stdout(io.StringIO()):
                    report.build(label)
                text=(output/'README.md').read_text(encoding='utf-8')
                if backend=='rtn':self.assertTrue(text.startswith('# Manual purpose'))
                else:self.assertIn('VCL',text)
                self.assertIn(plan[-1],text);self.assertNotIn('rtn-V-6',text)
                summary=json.loads((output/'measurements-summary.json').read_text(encoding='utf-8'))
                self.assertEqual(len(summary['groups']),len(plan)*3)


class MeasurementTests(unittest.TestCase):
    def test_extended_component_interruption_and_dynamic_completion(self):
        with tempfile.TemporaryDirectory(dir=SRC.parent) as name:
            root=Path(name).resolve();self.assertTrue(root.is_relative_to(SRC.parent))
            ids=[f'{i:012d}' for i in range(30)]
            for kind in ['clean','noise','adv']:
                (root/kind).mkdir()
                for iid in ids:(root/kind/(iid+'.png')).write_bytes((kind+iid).encode())
            fields=['model','case','image_id','input','n_tokens','ended_eos','output_limit_reached','latency_s','text']
            hashes={f'{iid}/{kind}':components.sha(root/kind/(iid+'.png'))
                    for iid in ids for kind in ['clean','noise']}
            adopted={iid:{'selected':{'sha256':components.sha(root/'adv'/(iid+'.png'))}} for iid in ids}
            status={};counter={'n':0,'interrupt':True};measured=[]
            def write_json(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
            def save_csv(path,rows,fields):
                with path.open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            def generate(model,processor,image,overrides):
                if overrides['max_new_tokens']==1:return {},[]
                counter['n']+=1
                if counter['interrupt'] and counter['n']==5:raise InterruptedError('synthetic interruption')
                measured.append((model,image))
                return dict(n_tokens=2,ended_eos=True,hit_cap=False,latency_s=.1,text='synthetic'),[1,151645]
            run=root/'run';run.mkdir()
            runner=SimpleNamespace(ids=ids,advdir=root/'adv',csv_path=root/'extended.csv',run=run,
                cfg={'cases':cases.component_cases('vcl'),'input_sha256':hashes},id='synthetic',
                update=lambda **kw:status.update(kw))
            namespace=dict(input_path=lambda split,kind,iid:root/kind/(iid+'.png'),
                sha256=components.sha,write_json=write_json,save_csv=save_csv,EVAL_FIELDS=fields,
                json=json,csv=csv,gc=gc,read_image=lambda path:path,check_preprocessing=lambda *a:{'cpu':0,'gpu':0},
                load_processor=lambda:None,load_case=lambda case:(case,{}),generate=generate,now=lambda:'synthetic',
                torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None,
                    reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0)))
            namespace=extract_functions('evaluate.py',{'measure_extended','validate_measurement_rows'},namespace)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InterruptedError):namespace['measure_extended'](runner,adopted)
                counter['interrupt']=False
                namespace['measure_extended'](runner,adopted)
                before=len(measured);namespace['measure_extended'](runner,adopted)
            self.assertEqual(len(measured),before)
            self.assertEqual(status['measurement_rows'],720)
            self.assertTrue(status['completed'])
            meta=json.loads(runner.csv_path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['row_count'],720)
            self.assertEqual(meta['csv_sha256'],components.sha(runner.csv_path))
            self.assertEqual(len(list((run/'responses').glob('*.json'))),720)

    def test_common_component_interruption_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory(dir=SRC.parent) as name:
            root=Path(name).resolve();self.assertTrue(root.is_relative_to(SRC.parent))
            ids=[f'{i:012d}' for i in range(30)]; calls=[]; counter={'n':0,'interrupt':True}
            fields=['model','case','image_id','input','n_tokens','hit_cap','latency_s','text']
            def write_json(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
            def save_csv(path,rows):
                with path.open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            def generate(model,processor,image,overrides=None):
                if overrides:return {},[]  # Warm-up is outside the recorded cohort.
                counter['n']+=1
                if counter['interrupt'] and counter['n']==8:raise InterruptedError('synthetic interruption')
                calls.append((model,image))
                return dict(n_tokens=2,hit_cap=False,latency_s=.1,text='synthetic'),[1,151645]
            settings=SimpleNamespace(CASES_PHASE1=['ref'],GEN_KWARGS={'max_new_tokens':512})
            namespace=dict(argparse=argparse,Path=Path,settings=settings,COMPONENTS=cases.COMPONENTS,
                BACKENDS=cases.BACKENDS,DEFAULT_BACKEND=cases.DEFAULT_BACKEND,
                component_cases=cases.component_cases,profile=cases.profile,
                image_ids=lambda *a:ids,verify_inputs=lambda *a:({'fingerprint':'fixed'},{}),
                load_processor=lambda:None,input_path=lambda split,kind,iid:root/kind/(iid+'.png'),
                check_preprocessing=lambda *a:{'cpu':0,'gpu':0},read_image=lambda p:str(p),
                environment=lambda:{'synthetic':True},fingerprint=lambda cfg:hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                json=json,csv=csv,gc=gc,sha256=components.sha,write_json=write_json,save_csv=save_csv,
                CSV_FIELDS=fields,load_case=lambda case:(case,{}),generate=generate,
                torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None,
                    reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0)))
            main=extract_functions('evaluate.py',{'common_main'},namespace)['common_main']
            path=root/'measurement.csv';args=['--split','eval','--component','language',
                '--adv-dir',str(root/'adv'),'--out',str(path)]
            progress=[]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InterruptedError):main(args,progress=progress.append)
                counter['interrupt']=False
                main(args,progress=progress.append)
                before=len(calls);main(args)
            self.assertEqual(len(calls),before)
            self.assertEqual(progress[-1],720)
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertTrue(meta['completed']);self.assertEqual(meta['row_count'],720)
            with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),720)
            self.assertEqual(len({(r['case'],r['image_id'],r['input']) for r in rows}),720)
            self.assertEqual({r['case'] for r in rows},set(cases.component_cases('language')))
            self.assertEqual(meta['csv_sha256'],components.sha(path))


if __name__ == '__main__':
    unittest.main()
