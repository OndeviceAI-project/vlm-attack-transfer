"""Prepare dev/eval PNG inputs on the experiment server; downloads require --download."""
import argparse,csv,json,hashlib,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import ROOT,REPO,settings,write_json,sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
 expected=json.loads((Path(__file__).resolve().parents[1]/'results/input-manifest.json').read_text())
 refs={(r['split'],r['input'],r['image_id']):r['sha256'] for r in expected['records']}
 with (REPO/'공통/image_list.csv').open() as f:ids=[r for r in csv.DictReader(f) if r['split'] in ('dev','eval')]
 records=[]
 for r in ids:
  iid=f"{int(r['coco_id']):012d}";split=r['split'];paths={k:ROOT/'data/qwen2vl'/split/k/(iid+'.png') for k in ('clean','noise')}
  if all(path.exists() and sha256(path)==refs[split,k,iid] for k,path in paths.items()):
   records.extend(dict(image_id=iid,split=split,input=k,path=str(path.relative_to(ROOT)),sha256=sha256(path)) for k,path in paths.items());continue
  if not a.download:raise RuntimeError('Missing/changed input '+iid+'; use --download on the experiment server to prepare inputs')
  for path in paths.values():
   if path.exists():raise RuntimeError('Refusing to replace existing changed/partial inputs; use a separate data root')
  raw=ROOT/'data/coco-val2017'/(iid+'.jpg');raw.parent.mkdir(parents=True,exist_ok=True)
  if not raw.exists():
   temp=raw.with_suffix('.jpg.tmp');urllib.request.urlretrieve('https://images.cocodataset.org/val2017/'+iid+'.jpg',temp);temp.replace(raw)
  with Image.open(raw) as pic:clean=pic.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
  arr=np.asarray(clean).astype(np.int16);sign=np.random.default_rng([2026,int(iid)]).choice([-1,1],size=arr.shape)
  noise=Image.fromarray(np.clip(arr+8*sign,0,255).astype(np.uint8))
  for k,pic in [('clean',clean),('noise',noise)]:
   paths[k].parent.mkdir(parents=True,exist_ok=True);temp=paths[k].with_suffix('.png.tmp');pic.save(temp,format='PNG')
   if sha256(temp)!=refs[split,k,iid]:raise RuntimeError('PNG hash differs from recorded inputs; retain temporary file and inspect Pillow/source data')
   temp.replace(paths[k]);records.append(dict(image_id=iid,split=split,input=k,path=str(paths[k].relative_to(ROOT)),sha256=sha256(paths[k])))
 # Check-only mode does not write or download.
 if a.download:write_json(ROOT/'data/manifest.json',dict(settings_version=settings.SETTINGS_VERSION,resize=expected['resize'],records=records))
 print('INPUTS_VERIFIED',len(records),'PNG files')
if __name__=='__main__':main()
