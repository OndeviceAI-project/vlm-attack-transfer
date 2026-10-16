"""Validate saved deliverables on CPU; never load a model or regenerate results."""
import csv,hashlib,json,os,unittest
from pathlib import Path
import numpy as np
from PIL import Image
csv.field_size_limit(10_000_000)
B=Path(__file__).resolve().parents[1]
CASES=['ref','rtn-V-8','rtn-V-6','rtn-V-5','rtn-V-4','rtn-V-3','bnb-V-8','bnb-V-4']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
class ArtifactsTests(unittest.TestCase):
 def test_csv_schema_cohort_and_hashes(self):
  for f in (B/'reports').glob('*/run_manifest.json'):
   cfg=json.loads(f.read_text());run=cfg['experiment_id'];path=B/'results'/(run+'.csv')
   meta=json.loads(path.with_suffix('.json').read_text())
   self.assertEqual(sha(path),meta['csv_sha256']);self.assertEqual(sha(path),cfg['csv_sha256'])
   with path.open(newline='') as stream:reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
   expected=['model','case','image_id','input','n_tokens']+(['hit_cap'] if cfg['max_new_tokens']==512 else ['ended_eos','output_limit_reached'])+['latency_s','text']
   self.assertEqual(fields,expected);self.assertEqual(len(rows),720)
   selected=json.loads((B/'adv'/cfg['image_set_id']/'selection.json').read_text())['images']
   ids={r['image_id'] for r in selected}
   self.assertEqual({(r['case'],r['image_id'],r['input']) for r in rows},{(c,i,k) for c in CASES for i in ids for k in ['clean','noise','adv']})
   for r in rows:
    self.assertEqual(r['model'],'qwen2vl');self.assertEqual(len(r['image_id']),12)
    self.assertTrue(1<=int(r['n_tokens'])<=cfg['max_new_tokens']);self.assertGreaterEqual(float(r['latency_s']),0)
 def test_png_and_reference_input_hashes(self):
  root=Path(os.environ.get('ONDEVICE_AI_ROOT',Path.home()/'ondevice-ai'))
  for sel in (B/'adv').glob('*/selection.json'):
   data=json.loads(sel.read_text());self.assertEqual(len(data['images']),30)
   sums=dict(line.split()[::-1] for line in (sel.parent/'SHA256SUMS').read_text().splitlines())
   for im in data['images']:
    png=sel.parent/im['path'];self.assertEqual(sha(png),im['sha256']);self.assertEqual(sums[png.name],im['sha256'])
    with Image.open(png) as pic:self.assertEqual(pic.size,(448,448));arr=np.asarray(pic).astype(np.int16)
    clean=root/'data/qwen2vl/eval/clean'/png.name
    if clean.exists():
     self.assertEqual(sha(clean),im['clean_sha256'])
     with Image.open(clean) as pic:self.assertLessEqual(int(np.abs(arr-np.asarray(pic).astype(np.int16)).max()),8)
 def test_consolidated_responses_match_csv(self):
  run='qwen2vl_vision_extended_v1_r01'
  with (B/'results'/(run+'.csv')).open(newline='') as stream:rows=list(csv.DictReader(stream))
  expected={(r['case'],r['image_id'],r['input']):r for r in rows}
  lines=(B/'results'/run/'responses.jsonl').read_text().splitlines();self.assertEqual(len(lines),720)
  seen=set()
  for line in lines:
   entry=json.loads(line);rec=entry['record'];r=rec['row'];key=r['case'],r['image_id'],r['input']
   self.assertNotIn(key,seen);seen.add(key);self.assertEqual({k:str(v) for k,v in r.items()},expected[key])
   self.assertEqual(len(rec['tokens']),r['n_tokens']);self.assertFalse(any(rec['preprocessing'].values()))
if __name__=='__main__':unittest.main()
