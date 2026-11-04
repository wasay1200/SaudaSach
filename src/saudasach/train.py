import json,time
from pathlib import Path
import joblib,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings
from .common import ROOT,CONFIG,read_csv,sha,write_json,assert_disjoint,seed_all,stable_seed,run_provenance
from .data import load_cache
from .metrics import metrics,threshold_from_dev
from .models import CompactCNN

def clip_scores(model,cache,ids,offsets):
 # Network accepts a batch of windows; aggregate each clip at the end.
 model.eval();outputs=[];winids=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids])
 with torch.no_grad():
  for b in range(0,len(winids),256):outputs.extend(model(torch.from_numpy(np.asarray(cache[winids[b:b+256]],dtype=np.float32))).numpy().tolist())
 out=[];pos=0
 for i in ids:
  n=offsets[i+1]-offsets[i];out.append(float(np.mean(outputs[pos:pos+n])));pos+=n
 return np.asarray(out)

class TrainingData(Dataset):
 def __init__(self,arrays,indices,labels,augmented,seed):self.arrays=arrays;self.indices=indices;self.labels=labels;self.augmented=augmented;self.seed=seed;self.epoch=0
 def __len__(self):return len(self.indices)
 def __getitem__(self,k):
  idx=self.indices[k];which='clean'
  if self.augmented:
   rng=np.random.default_rng(stable_seed(self.seed,self.epoch,int(idx)));bucket=int(rng.integers(4))
   if bucket==1:which='noise20' if rng.integers(2) else 'noise10'
   elif bucket==2:which='opus24' if rng.integers(2) else 'gsm13'
   elif bucket==3:which='noise10_opus24' if rng.integers(2) else 'noise10_gsm13'
  return np.asarray(self.arrays[which][idx],dtype=np.float32),np.float32(self.labels[idx])

def train_all():
 rows=read_csv(ROOT/'manifests/splits.csv');assert_disjoint(rows)
 offsets=np.load(ROOT/'data/cache/offsets.npy');labels=np.array([int(r['label']) for r in rows]);trainids=np.array([i for i,r in enumerate(rows) if r['split']=='train']);devids=np.array([i for i,r in enumerate(rows) if r['split']=='dev'])
 arrays={c:load_cache(c) for c in ['clean','noise20','noise10','opus24','gsm13','noise10_opus24','noise10_gsm13']}
 winlabels=np.repeat(labels,np.diff(offsets));trainwindows=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in trainids])
 stat=np.asarray(arrays['clean'][trainwindows],dtype=np.float32)
 mean=stat.mean(axis=(0,2))[:,None];std=np.maximum(stat.std(axis=(0,2))[:,None],1.0);del stat
 manifest_hash=sha(ROOT/'manifests/splits.csv');(ROOT/'artifacts').mkdir(exist_ok=True)
 for augmented in [False,True]:
  variant='augmented' if augmented else 'ordinary'
  for seed in CONFIG['seeds']:
   out=ROOT/'artifacts'/f'cnn_{variant}_seed{seed}';out.mkdir(exist_ok=True)
   if (out/'metadata.json').exists():
    saved=json.loads((out/'metadata.json').read_text())
    if saved['manifest_sha256']!=manifest_hash or saved['config_sha256']!=sha(ROOT/'configs/study.json'):raise ValueError('Stale training artifacts')
    continue
   seed_all(seed);model=CompactCNN(mean,std)
   ds=TrainingData(arrays,trainwindows,winlabels,augmented,seed)
   loader=DataLoader(ds,batch_size=CONFIG['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
   pos=sum(winlabels[trainwindows]);neg=len(trainwindows)-pos
   lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,dtype=torch.float32))
   opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=.001)
   history=[];best=-1;stale=0;started=time.time()
   for epoch in range(CONFIG['epochs']):
    model.train();ds.epoch=epoch;total=0
    for x,y in loader:
     opt.zero_grad();logits=model(x);loss=lossfn(logits,y);loss.backward();opt.step();total+=float(loss.detach())*len(y)
    vals=[]
    for cond in CONFIG['development_selection_conditions']:
     sc=clip_scores(model,arrays[cond],devids,offsets);vals.append(metrics(labels[devids],sc,0)['balanced_accuracy'])
    score=float(np.mean(vals));history.append({'epoch':epoch+1,'loss':total/len(ds),'dev_mean_balanced_accuracy_at_zero':score})
    print('TRAIN',variant,seed,'epoch',epoch+1,'loss',round(total/len(ds),4),'dev',round(score,4),'elapsed',round(time.time()-started),flush=True)
    if score>best+1e-6:best=score;stale=0;torch.save(model.state_dict(),out/'weights.pt')
    else:stale+=1
    write_json(out/'history.json',history)
    if stale>=CONFIG['patience']:break
   model.load_state_dict(torch.load(out/'weights.pt',weights_only=True,map_location='cpu'))
   pooled=np.concatenate([clip_scores(model,arrays[c],devids,offsets) for c in CONFIG['development_selection_conditions']]);yy=np.tile(labels[devids],len(CONFIG['development_selection_conditions']))
   threshold=threshold_from_dev(yy,pooled);width=float(max(.1,.1*(np.quantile(pooled,.75)-np.quantile(pooled,.25))))
   write_json(out/'metadata.json',{'architecture':'CompactCNN temporal log-mel CNN','variant':variant,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'threshold':threshold,'uncertainty_low':threshold-width,'uncertainty_high':threshold+width,'threshold_source':'pooled clean/noise10/opus24/gsm13 development clips only','best_dev_selection':best,'elapsed_seconds':time.time()-started,'manifest_sha256':manifest_hash,'config_sha256':sha(ROOT/'configs/study.json'),'training_window_count':len(ds),'pretrained_weights':False,'training_budget':'Same windows per epoch and epoch cap; early stopping may differ.','provenance':run_provenance()})
   print('SAVED',out.name,flush=True)
 # SVM reference and two directional generator-holdout references, no final-test access.
 sx=load_cache('clean','svm');dy=labels[devids]
 for name,exclude in [('svm_reference',None),('svm_without_tacotron','tacotron'),('svm_without_tts','tts')]:
  out=ROOT/'artifacts'/name;out.mkdir(exist_ok=True)
  if (out/'metadata.json').exists():continue
  ids=np.array([i for i in trainids if rows[i]['generator']!=exclude]);dev=np.array([i for i in devids if rows[i]['generator']!=exclude]);best=-1
  for c in [.01,.1,1,10]:
   pipe=make_pipeline(StandardScaler(),LinearSVC(C=c,class_weight='balanced',dual=False,max_iter=30000,random_state=42))
   with warnings.catch_warnings():
    warnings.simplefilter('error',ConvergenceWarning);pipe.fit(np.asarray(sx[ids]),labels[ids])
   score=metrics(labels[dev],pipe.decision_function(np.asarray(sx[dev])),0)['balanced_accuracy']
   if score>best:best=score;model=pipe;selected_c=c
  threshold=threshold_from_dev(labels[dev],model.decision_function(np.asarray(sx[dev])))
  joblib.dump(model,out/'model.joblib');write_json(out/'metadata.json',{'C':selected_c,'threshold':threshold,'excluded_generator':exclude,'dev_selection':best,'manifest_sha256':manifest_hash,'config_sha256':sha(ROOT/'configs/study.json'),'seed':42,'provenance':run_provenance()})
  print('SAVED',name,flush=True)
 # Freeze the deployable winner using development evidence, not final-test results.
 choices=[]
 for file in (ROOT/'artifacts').glob('cnn_*/metadata.json'):
  m=json.loads(file.read_text());choices.append((m['best_dev_selection'],file.parent.name))
 choices.sort(key=lambda x:(-x[0],x[1]))
 write_json(ROOT/'artifacts/selection.json',{'selected':choices[0][1],'criterion':'highest mean development balanced accuracy at zero during checkpoint selection; ties lexical','all_models':choices,'manifest_sha256':manifest_hash,'config_sha256':sha(ROOT/'configs/study.json'),'weight_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'artifacts').rglob('*')) if p.name in ['weights.pt','model.joblib']},'final_test_seen':False})
 print('All models trained; selection frozen.',flush=True)
