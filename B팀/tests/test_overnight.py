"""CPU-only synthetic checks; no model load, GPU execution or real experiment."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import attack as state
from evaluate import EVAL_FIELDS,validate_measurement_rows

class FakeEncoding(dict):
    def to(self,*a,**kw):return self
class FakeModel:
    def named_modules(self):return []

class OvernightTests(unittest.TestCase):
    def test_selection_tie_and_tolerance(self):
        values=[dict(iteration=60,measurement={'n_tokens':5000}),dict(iteration=20,measurement={'n_tokens':5000}),dict(iteration=40,measurement={'n_tokens':3000})]
        self.assertEqual([x['iteration'] for x in state.ranked_candidates(values)],[20,60,40])
        self.assertLessEqual(state.relative_difference(5000,4000),.20)
        self.assertGreater(state.relative_difference(5000,3999),.20)
        self.assertLessEqual(state.relative_difference(100,120),.20)
    def test_measurement_cohort_duplicate_and_bad_length(self):
        ids=[f'{i:012d}' for i in range(30)];cases=['ref','rtn-V-8','rtn-V-6','rtn-V-5','rtn-V-4','rtn-V-3','bnb-V-8','bnb-V-4']
        rows=[dict(model='qwen2vl',case=c,image_id=i,input=k,n_tokens=20,ended_eos=True,output_limit_reached=False,latency_s=.1,text='synthetic') for c in cases for i in ids for k in ['clean','noise','adv']]
        seen,expected=validate_measurement_rows(rows,ids,cases)
        self.assertEqual(len(seen),720);self.assertEqual(seen,expected)
        with self.assertRaises(ValueError):validate_measurement_rows(rows+[rows[0]],ids,cases)
        broken=[{**rows[0],'n_tokens':5001}]
        with self.assertRaises(ValueError):validate_measurement_rows(broken,ids,cases)
    def test_snapshot_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            p=Path(name)/'state.pt'
            state.save_state(p,{'delta':torch.arange(8),'step':40})
            loaded=torch.load(p,map_location='cpu',weights_only=False)
            self.assertEqual(loaded['step'],40);self.assertTrue(torch.equal(loaded['delta'],torch.arange(8)))
            self.assertFalse(p.with_name('state.pt.tmp').exists())
    def test_interrupted_generation_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name,contextlib.ExitStack() as stack:
            root=Path(name);clean=root/'clean.png';noise=root/'noise.png'
            Image.fromarray(np.full((4,4,3),100,dtype=np.uint8)).save(clean)
            Image.fromarray(np.full((4,4,3),108,dtype=np.uint8)).save(noise)
            original_to=torch.Tensor.to
            def cpu_to(t,*a,**kw):
                a=list(a)
                if a and isinstance(a[0],str) and a[0]=='cuda':a[0]='cpu'
                if kw.get('device')=='cuda':kw['device']='cpu'
                return original_to(t,*a,**kw)
            def target(*a):return torch.randint(0,8,(2,))
            def objective(model,processor,base,pixels,target,eos):
                loss=((pixels-target.float().mean()*30)**2).mean()
                return loss,loss,loss*0
            def measured(model,processor,path,cap):
                n=20+int(np.asarray(Image.open(path))[0,0,0])%17
                return dict(n_tokens=n,ended_eos=True,output_limit_reached=False,latency_s=0.,text='synthetic',tokens=[1]*(n-1)+[2],preprocessing={'cpu':0,'gpu':0})
            patches=[patch.object(torch.Tensor,'to',cpu_to),patch.object(state,'input_path',lambda split,kind,iid:clean if kind=='clean' else noise),
                patch.object(state,'read_image',lambda p:Image.open(p).convert('RGB')),patch.object(state,'raw_pixels',lambda im,dev:torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()),
                patch.object(state,'encode',lambda *a:FakeEncoding()),patch.object(state,'eos_ids',lambda *a:[2,3]),patch.object(state,'target_tokens',target),
                patch.object(state,'objective',objective),patch.object(state,'measured',measured),patch.object(torch.cuda,'get_rng_state_all',lambda:[]),
                patch.object(torch.cuda,'set_rng_state_all',lambda values:None),patch.object(torch.cuda,'reset_peak_memory_stats',lambda:None),patch.object(torch.cuda,'max_memory_allocated',lambda:0)]
            for item in patches:stack.enter_context(item)
            def stop(iid,it):
                if it==40:raise InterruptedError('synthetic interruption')
            full=root/'full';resumed=root/'resumed';full.mkdir();resumed.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                a=state.generate_one(FakeModel(),None,'000000000001',full,'eval',lambda *a:None)
                with self.assertRaises(InterruptedError):state.generate_one(FakeModel(),None,'000000000001',resumed,'eval',stop)
                b=state.generate_one(FakeModel(),None,'000000000001',resumed,'eval',lambda *a:None)
            self.assertEqual([x['sha256'] for x in a['candidates']],[x['sha256'] for x in b['candidates']])
            self.assertEqual([x['measurement']['tokens'] for x in a['candidates']],[x['measurement']['tokens'] for x in b['candidates']])
            self.assertEqual(a['selected']['sha256'],b['selected']['sha256'])
            self.assertEqual(b['iterations_run'],500);self.assertEqual(len(b['candidates']),25)

if __name__=='__main__':unittest.main()
