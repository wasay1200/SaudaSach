import csv,hashlib,json,os,platform,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/'configs/study.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def stable_seed(*values):return int(hashlib.sha256(':'.join(map(str,values)).encode()).hexdigest()[:8],16)
def write_json(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def read_csv(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise ValueError('No records to write')
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seed_all(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True)
def assert_disjoint(rows):
 if not rows:raise ValueError('Empty data')
 for key in ['group','sha256','pcm_sha256']:
  assignments={}
  for r in rows:
   if not r.get(key):raise ValueError('Missing '+key)
   assignments.setdefault(r[key],set()).add(r['split'])
  if any(len(v)>1 for v in assignments.values()):raise ValueError('Cross-split overlap: '+key)
 for part in ['train','dev','test']:
  if {r['label'] for r in rows if r['split']==part}!={'0','1'}:raise ValueError('Missing class in '+part)

def run_provenance():
 import subprocess,importlib.metadata,sys
 try:revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 except subprocess.CalledProcessError:revision='uncommitted'
 versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','librosa','soundfile','torch','onnx','onnxruntime']}
 sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
 return {'code_revision':revision,'source_hashes':sources,'python':sys.version,'environment_versions':versions,'platform':platform.platform()}
