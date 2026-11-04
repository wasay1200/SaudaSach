"""Portable ONNX packaging, integrity checks and offline file inference."""
import json,shutil,time,platform
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from .common import ROOT,CONFIG,read_csv,write_json,sha
from .audio import load_audio,windows,logmel,SR
from .data import load_cache
from .models import CompactCNN

def export_bundle():
 selection=json.loads((ROOT/'artifacts/selection.json').read_text());chosen=ROOT/'artifacts'/selection['selected'];meta=json.loads((chosen/'metadata.json').read_text());model=CompactCNN();model.load_state_dict(torch.load(chosen/'weights.pt',weights_only=True,map_location='cpu'));model.eval();torch.set_num_threads(4)
 out=ROOT/'artifacts/deployment';out.mkdir(exist_ok=True);path=out/'model.onnx'
 torch.onnx.export(model,torch.zeros(1,64,197),str(path),input_names=['logmel'],output_names=['synthetic_logit'],dynamic_axes={'logmel':{0:'batch'},'synthetic_logit':{0:'batch'}},opset_version=17,dynamo=False)
 onnx.checker.check_model(onnx.load(path))
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;sess=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
 rows=read_csv(ROOT/'manifests/splits.csv');offsets=np.load(ROOT/'data/cache/offsets.npy');clean=load_cache('clean');dev=[i for i,r in enumerate(rows) if r['split']=='dev'][:10];errors=[]
 for i in dev:
  x=np.asarray(clean[offsets[i]:offsets[i+1]],dtype=np.float32)
  with torch.no_grad():a=model(torch.from_numpy(x)).numpy()
  b=sess.run(None,{'logmel':x})[0];errors.append(float(np.max(np.abs(a-b))))
 if max(errors)>1e-4:raise ValueError('ONNX parity failed')
 payload={'schema_version':1,'model_file':'model.onnx','sha256':sha(path),'source_model':selection['selected'],'labels':{'0':'genuine','1':'synthetic'},'feature_shape':[64,197],'frontend':CONFIG,'aggregation':'mean of per-window logits, 2-second windows, 0.5-second step with final full window','threshold':meta['threshold'],'uncertainty_low':meta['uncertainty_low'],'uncertainty_high':meta['uncertainty_high'],'score_type':'uncalibrated logit, not probability','onnx_max_abs_error':max(errors),'training_license_notice':'Research use only; see DATA_LICENSES.md. No raw recordings are redistributed.'}
 write_json(out/'metadata.json',payload);write_json(ROOT/'reports/export_parity.json',{'fixtures':10,'max_absolute_logit_error':max(errors),'tolerance':1e-4,'passed':True,'scope':'PyTorch versus ONNX on Python-generated log-mel tensors, not Android feature parity'})
 # Static QDQ quantization uses only training calibration features.
 from onnxruntime.quantization import CalibrationDataReader,QuantType,QuantFormat,quantize_static
 train=[i for i,r in enumerate(rows) if r['split']=='train'];calibration=[offsets[i] for i in train[:100]]
 class Reader(CalibrationDataReader):
  def __init__(self):self.it=iter(calibration)
  def get_next(self):
   i=next(self.it,None)
   return None if i is None else {'logmel':np.asarray(clean[i:i+1],dtype=np.float32)}
 quantize_static(str(path),str(out/'model_int8.onnx'),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,per_channel=True)
 q=ort.InferenceSession(str(out/'model_int8.onnx'),opts,providers=['CPUExecutionProvider']);q.run(None,{'logmel':np.asarray(clean[0:1],dtype=np.float32)})
 write_json(out/'quantization.json',{'file':'model_int8.onnx','sha256':sha(out/'model_int8.onnx'),'calibration_partition':'train','calibration_windows':len(calibration),'full_precision_bytes':path.stat().st_size,'int8_bytes':(out/'model_int8.onnx').stat().st_size,'storage_reduction':1-(out/'model_int8.onnx').stat().st_size/path.stat().st_size,'default_deployment':'full precision until matched evaluation and trade-offs are recorded'})
 print('Bundle exported:',chosen.name,'parity',max(errors),flush=True)

def load_bundle(directory):
 directory=Path(directory);m=json.loads((directory/'metadata.json').read_text())
 if m.get('schema_version')!=1 or m.get('feature_shape')!=[64,197] or m.get('labels')!={'0':'genuine','1':'synthetic'}:raise ValueError('Incompatible bundle schema')
 file=directory/m['model_file']
 if file.parent.resolve()!=directory.resolve() or sha(file)!=m['sha256']:raise ValueError('Invalid model path or checksum')
 opts=ort.SessionOptions();opts.intra_op_num_threads=2
 return m,ort.InferenceSession(str(file),opts,providers=['CPUExecutionProvider'])

def predict(path,bundle=ROOT/'artifacts/deployment'):
 started=time.perf_counter()
 if Path(path).stat().st_size>20*1024*1024:raise ValueError('File exceeds 20 MiB')
 import soundfile as sf
 if not 3<=sf.info(path).duration<=60:raise ValueError('Supported duration is 3 to 60 seconds')
 m,sess=load_bundle(bundle);y=load_audio(path)
 seconds=len(y)/SR
 if not 3<=seconds<=60:raise ValueError('Supported duration is 3 to 60 seconds')
 if Path(path).stat().st_size>20*1024*1024:raise ValueError('File exceeds 20 MiB')
 if np.sqrt(np.mean(y.astype(np.float64)**2))<1e-5:return {'outcome':'Inconclusive','reason':'Insufficient audio energy','analyzed_seconds':seconds}
 # Energy gate is conservative input screening, not a validated speech activity detector.
 active=librosa_activity_seconds(y)
 if active<2:return {'outcome':'Inconclusive','reason':'Insufficient active audio','analyzed_seconds':seconds}
 x=np.stack([logmel(w) for w in windows(y,.5)]);score=float(sess.run(None,{'logmel':x})[0].mean())
 outcome='Inconclusive' if m['uncertainty_low']<=score<=m['uncertainty_high'] else ('Synthetic speech suspected' if score>m['threshold'] else 'No strong synthetic evidence')
 return {'outcome':outcome,'evidence_logit':score,'score_is_probability':False,'analyzed_seconds':seconds,'windows':len(x),'model':m['source_model'],'elapsed_seconds_including_model_load':time.perf_counter()-started,'limitation':'Does not verify identity, truthfulness or absence of fraud; energy gate is not validated VAD.'}

def librosa_activity_seconds(y):
 import librosa
 intervals=librosa.effects.split(y,top_db=35,frame_length=400,hop_length=160)
 return float(sum(b-a for a,b in intervals)/SR)

def deployment_benchmark():
 if not json.loads((ROOT/'reports/final_test_access.json').read_text()).get('completed'):raise ValueError('Run frozen final evaluation first')
 import onnxruntime as ort
 from .metrics import metrics
 rows=read_csv(ROOT/'manifests/splits.csv');ids=np.array([i for i,r in enumerate(rows) if r['split']=='test']);y=np.array([int(rows[i]['label']) for i in ids]);offsets=np.load(ROOT/'data/cache/offsets.npy');out=ROOT/'artifacts/deployment';m,s=load_bundle(out)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;q=ort.InferenceSession(str(out/'model_int8.onnx'),opts,providers=['CPUExecutionProvider']);reports={}
 for cond in CONFIG['test_conditions']:
  cache=load_cache(cond);a=[];b=[]
  for i in ids:
   x=np.asarray(cache[offsets[i]:offsets[i+1]],dtype=np.float32)
   a.append(float(s.run(None,{'logmel':x})[0].mean()));b.append(float(q.run(None,{'logmel':x})[0].mean()))
  reports[cond]={'full_precision':metrics(y,a,m['threshold']),'int8':metrics(y,b,m['threshold'])}
 reduction=json.loads((out/'quantization.json').read_text())['storage_reduction'];eer_change=float(np.mean([reports[c]['int8']['eer']-reports[c]['full_precision']['eer'] for c in reports]))
 write_json(ROOT/'reports/quantization_evaluation.json',{'conditions':reports,'storage_reduction':reduction,'mean_eer_increase':eer_change,'target_achieved':bool(reduction>=.5 and eer_change<=.01),'threshold_policy':'Same validation-selected threshold; no quantized test retuning','note':'Final test used only for frozen variant comparison; default deployment remains full precision.'})
 dev=[i for i,r in enumerate(rows) if r['split']=='dev'];lat=[]
 # 50 warm preprocessing-plus-inference timings, model already loaded; selected dev files are >=3s.
 valid=[i for i in dev if 3<=float(rows[i]['seconds'])<=15]
 for k in range(53):
  i=valid[k%len(valid)];start=time.perf_counter();wav=load_audio(ROOT/'data/raw'/rows[i]['path'],max_seconds=15);x=np.stack([logmel(w) for w in windows(wav,.5)]);s.run(None,{'logmel':x})
  if k>=3:lat.append(time.perf_counter()-start)
 write_json(ROOT/'reports/host_benchmark.json',{'platform':platform.platform(),'processor':platform.processor(),'cpu_threads':2,'device':'Execution host CPU, NOT an Android phone','input_profile':'Up to 15 seconds, real development audio, rolling 2s/0.5s frontend','runs':50,'median_seconds':float(np.median(lat)),'p95_seconds':float(np.quantile(lat,.95)),'includes':'decoding/resampling, log-mel frontend and inference; excludes already-loaded model','physical_android_status':'Not performed; NFR-002/018/023 require later app/device work.'})
 print('Quantization comparison and host benchmark complete',flush=True)
