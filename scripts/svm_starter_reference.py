#!/usr/bin/env python3
"""SaudaSach CPU SVM baseline. Run --help; see README.md for the complete workflow."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class DataConflictError(ValueError):
    pass


SR = 16000
FEATURE_CONFIG = dict(sample_rate=SR, max_seconds=6.0, min_seconds=0.5,
                      n_mfcc=40, n_mels=64, n_fft=512, win_length=400,
                      hop_length=160, center=False, delta_width=9,
                      pooling='mean_and_std_mfcc_delta_delta2', peak_normalization=False)
FIELDS = ['path', 'label', 'generator', 'group', 'sha256', 'seconds', 'sample_rate']
EXTENSIONS = {'.wav', '.flac', '.ogg'}
CONDITIONS = ['clean', 'noise20', 'noise10', 'opus24', 'noise10_opus24']


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest is empty.')
    for r in rows:
        if r.get('label') not in ('0', '1'):
            raise ValueError('Each label must be 0 (genuine) or 1 (synthetic).')
    return rows


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audio(path, cfg=FEATURE_CONFIG):
    # Training and prediction use identical deterministic first-six-second preprocessing.
    with sf.SoundFile(path) as f:
        y = f.read(frames=math.ceil(cfg['max_seconds'] * f.samplerate),
                   dtype='float32', always_2d=True)
        original_sr = f.samplerate
    y = y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Empty or non-finite audio.')
    if original_sr != cfg['sample_rate']:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=cfg['sample_rate'])
    if len(y) < cfg['min_seconds'] * cfg['sample_rate']:
        raise ValueError('Less than 0.5 seconds of audio.')
    if np.sqrt(np.mean(y.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError('Silence or near-silence; no prediction is produced.')
    return y.astype(np.float32)


def features(y, cfg=FEATURE_CONFIG):
    m = librosa.feature.mfcc(y=y, sr=cfg['sample_rate'], n_mfcc=cfg['n_mfcc'],
                            n_mels=cfg['n_mels'], n_fft=cfg['n_fft'],
                            win_length=cfg['win_length'], hop_length=cfg['hop_length'],
                            center=cfg['center'])
    ds = [m, librosa.feature.delta(m, width=cfg['delta_width'], mode='nearest'),
          librosa.feature.delta(m, order=2, width=cfg['delta_width'], mode='nearest')]
    x = np.concatenate([v for d in ds for v in (d.mean(axis=1), d.std(axis=1))])
    if not np.isfinite(x).all():
        raise ValueError('Non-finite features.')
    return x.astype(np.float32)


def opus_roundtrip(y):
    if not shutil.which('ffmpeg'):
        raise ValueError('Opus conditions require FFmpeg with libopus. Omit --include-opus initially.')
    with tempfile.TemporaryDirectory(prefix='saudasach_') as tmp:
        tmp = Path(tmp); sf.write(tmp/'in.wav', y, SR, subtype='FLOAT')
        commands = [
            ['ffmpeg', '-v', 'error', '-y', '-i', str(tmp/'in.wav'), '-ar', '48000',
             '-ac', '1', '-c:a', 'libopus', '-b:a', '24k', str(tmp/'out.ogg')],
            ['ffmpeg', '-v', 'error', '-y', '-i', str(tmp/'out.ogg'), '-ar', str(SR),
             '-ac', '1', '-c:a', 'pcm_f32le', str(tmp/'decoded.wav')]]
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise ValueError('FFmpeg failed: ' + result.stderr.strip())
        result, _ = sf.read(tmp/'decoded.wav', dtype='float32')
        return result[:len(y)]


def degrade(y, condition, key, seed):
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition: ' + condition)
    z = y.copy()
    if condition.startswith('noise'):
        snr = 20 if condition == 'noise20' else 10
        stable_seed = int(hashlib.sha256(f'{seed}:{key}:{condition}'.encode()).hexdigest()[:16], 16)
        noise = np.random.default_rng(stable_seed).normal(size=len(y))
        noise *= np.sqrt(np.mean(y.astype(np.float64)**2) /
                         (np.mean(noise**2) * 10**(snr/10)))
        z = (z + noise).astype(np.float32)
    if 'opus24' in condition:
        z = opus_roundtrip(z)
    return z


def scan(args):
    import re
    root = Path(args.data).resolve()
    if not root.is_dir():
        raise ValueError('Dataset directory not found: ' + str(root))
    candidates, excluded, unknown = [], [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        parts = [re.sub(r'[^a-z0-9]', '', s.lower()) for s in relative.parts[:-1]]
        real = any(x in {'bonafide', 'bonafidepart1', 'bonafidepart2', 'genuine', 'real'} for x in parts)
        fake = any(x in {'spoofedtts', 'spoofedtacotron', 'tacotron', 'vits', 'vitstts', 'fake', 'synthetic'} for x in parts)
        if real == fake:
            unknown.append(str(relative)); continue
        match = next((re.fullmatch(r'speaker[_ -]?(\d+)', v, re.I)
                      for v in relative.parts[:-1] if re.fullmatch(r'speaker[_ -]?(\d+)', v, re.I)), None)
        group = f'speaker_{int(match.group(1)):02d}' if match else ''
        generator = 'genuine' if real else ('tacotron' if any('tacotron' in v for v in parts) else 'tts_or_other')
        candidates.append((path, relative, group, str(int(fake)), generator))
    if unknown:
        raise ValueError('Unrecognized label folders. Use the documented names or build a custom CSV. Examples: ' + ', '.join(unknown[:5]))
    if not candidates:
        raise ValueError('No WAV, FLAC or OGG files found. The GitHub repository itself does not contain the audio.')
    # Deterministic optional pilot subset, stratified by group and generator; never an official benchmark.
    if args.max_per_group:
        counts = {}; selected = []
        for item in sorted(candidates, key=lambda v: hashlib.sha256(str(v[1]).encode()).hexdigest()):
            key = (item[2], item[3], item[4]); n = counts.get(key, 0)
            if n < args.max_per_group:
                selected.append(item); counts[key] = n + 1
        candidates = selected
    rows = []; seen = {}; content_seen = {}
    for i, (path, relative, group, label, generator) in enumerate(candidates):
        try:
            info = sf.info(path)
            if info.duration < FEATURE_CONFIG['min_seconds']:
                raise ValueError('Too short for the baseline.')
            # Full-file raw-byte hash catches duplicate files; decoded content also catches header differences.
            rawhash = digest(path)
            yfull, rate = sf.read(path, dtype='float32', always_2d=True)
            if not np.isfinite(yfull).all() or not len(yfull):
                raise ValueError('Invalid decoded audio.')
            pcmhash = hashlib.sha256(str(rate).encode() + np.asarray(yfull, dtype='<f4').tobytes()).hexdigest()
            prior = seen.get(rawhash) or content_seen.get(pcmhash)
            if prior:
                if prior['label'] != label:
                    raise DataConflictError('Identical audio has conflicting labels: ' + str(relative))
                if prior['group'] != group:
                    raise DataConflictError('Identical audio has different groups. Reconcile identities: ' + str(relative))
                excluded.append(dict(path=str(relative), reason='Exact duplicate of '+prior['path'])); continue
            audio(path)  # Check the actual first-six-second representation too.
            row = dict(path=relative.as_posix(), label=label, generator=generator, group=group,
                       sha256=rawhash, seconds=round(info.duration, 5), sample_rate=info.samplerate)
            seen[rawhash] = row; content_seen[pcmhash] = row; rows.append(row)
        except DataConflictError:
            raise
        except Exception as exc:
            excluded.append(dict(path=relative.as_posix(), reason=str(exc)))
        if (i+1) % 100 == 0:
            print(f'Audited {i+1}/{len(candidates)} files', flush=True)
    if not rows:
        raise ValueError('No usable recordings remain.')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'manifest.csv', rows, FIELDS)
    write_csv(out/'excluded.csv', excluded, ['path', 'reason'])
    summary = dict(files=len(rows), excluded=len(excluded),
                   class_counts={k:sum(r['label']==k for r in rows) for k in ['0','1']},
                   groups=len({r['group'] for r in rows if r['group']}),
                   missing_groups=sum(not r['group'] for r in rows),
                   source_seconds=sum(float(r['seconds']) for r in rows),
                   max_per_group=args.max_per_group,
                   note='Automatically inferred folder labels/groups MUST be reviewed. No speech VAD or near-duplicate detection.')
    save_json(out/'audit.json', summary); print(json.dumps(summary, indent=2))
    print('Next: review manifest.csv, especially group and label; then run split.')


def validate(rows, require_split=True):
    required = ['path','label','group','sha256'] + (['split'] if require_split else [])
    for r in rows:
        if any(not r.get(k, '').strip() for k in required):
            raise ValueError('Missing path, group, hash or split. Review the manifest; file-random fallback is intentionally unavailable.')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest paths.')
    if require_split:
        if set(r['split'] for r in rows) != {'train','dev','test'}:
            raise ValueError('Exactly train, dev and test partitions are required.')
        for key in ['group','sha256']:
            partitions = {}
            for r in rows:partitions.setdefault(r[key], set()).add(r['split'])
            if any(len(x)>1 for x in partitions.values()):
                raise ValueError(f'Cross-partition {key} overlap detected.')
        for part in ['train','dev','test']:
            if set(r['label'] for r in rows if r['split']==part) != {'0','1'}:
                raise ValueError(f'{part} lacks one class; revise the split using metadata, not test performance.')


def split(args):
    if not args.groups_reviewed:
        raise ValueError('Review the group identities, then pass --groups-reviewed. Never treat each clip as an independent speaker.')
    rows = read_csv(args.manifest); validate(rows, False)
    groups = np.array([r['group'] for r in rows]); idx = np.arange(len(rows))
    if len(set(groups)) < 6:
        raise ValueError('At least six independent groups are needed for this starter split.')
    train_dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.15,
                                           random_state=args.seed).split(idx, groups=groups))
    tr, dv = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85,
                                    random_state=args.seed+1).split(train_dev, groups=groups[train_dev]))
    for name, ids in [('train',train_dev[tr]),('dev',train_dev[dv]),('test',test)]:
        for i in ids: rows[i]['split'] = name
    validate(rows)
    write_csv(args.out, rows, list(rows[0]))
    print('Saved',args.out)
    for part in ['train','dev','test']:
        rs=[r for r in rows if r['split']==part]
        print(part, 'files=',len(rs), 'groups=',sorted(set(r['group'] for r in rs)),
              'genuine=',sum(r['label']=='0' for r in rs), 'synthetic=',sum(r['label']=='1' for r in rs))
    print('Fractions apply to groups, not exact file counts. Keep this CSV frozen for all comparisons.')


def matrix(rows, root, conditions, seed, cfg=FEATURE_CONFIG):
    xs, ys, metadata = [], [], []
    for i, r in enumerate(rows):
        path = Path(root)/r['path']
        if digest(path) != r['sha256']:
            raise ValueError('Audio changed since audit: '+r['path'])
        y = audio(path, cfg)
        for condition in conditions:
            z = degrade(y, condition, r['path'], seed)
            xs.append(features(z,cfg)); ys.append(int(r['label']))
            metadata.append(dict(path=r['path'],group=r['group'],generator=r.get('generator','unknown'),condition=condition))
        if (i+1)%100==0:print(f'Features {i+1}/{len(rows)}',flush=True)
    return np.asarray(xs),np.asarray(ys),metadata


def metrics(y, scores):
    pred=(scores>=0).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    fpr,tpr,_=roc_curve(y,scores,drop_intermediate=False)
    # Interpolate the ROC crossing with FPR=FNR; EER is descriptive, not a deployed threshold.
    difference=fpr+tpr-1; j=np.flatnonzero(difference>=0)[0]
    if j==0:eer=float(fpr[0])
    else:
        alpha=-difference[j-1]/(difference[j]-difference[j-1])
        eer=float(fpr[j-1]+alpha*(fpr[j]-fpr[j-1]))
    return dict(n=len(y), accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                fake_f1=float(f1_score(y,pred,zero_division=0)),
                genuine_recall=float(cm[0,0]/cm[0].sum()), fake_recall=float(cm[1,1]/cm[1].sum()),
                false_positive_rate=float(cm[0,1]/cm[0].sum()), roc_auc=float(roc_auc_score(y,scores)),
                eer=eer, decision_threshold=0.0, confusion_matrix=cm.tolist(),
                classification_report=classification_report(y,pred,labels=[0,1],
                    target_names=['genuine','synthetic'],output_dict=True,zero_division=0))


def train(args):
    rows=read_csv(args.manifest);validate(rows)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Run output folder must be empty/new; use a new name to preserve prior results.')
    out.mkdir(parents=True,exist_ok=True)
    conditions=['clean'] + (['noise20','noise10'] if args.augment else [])
    if args.include_opus:
        if not args.augment:raise ValueError('--include-opus in train requires --augment.')
        conditions+=['opus24','noise10_opus24']
        opus_roundtrip(np.ones(SR,dtype=np.float32)*.01)
    tr=[r for r in rows if r['split']=='train']; dev=[r for r in rows if r['split']=='dev']
    started=time.perf_counter(); x,y,_=matrix(tr,args.data,conditions,args.seed)
    xv,yv,_=matrix(dev,args.data,['clean'],args.seed)
    candidates=[]; best=None; best_score=-1
    for c in [.01,.1,1.,10.]:
        model=make_pipeline(StandardScaler(),LinearSVC(C=c,class_weight='balanced',dual=False,
                                                      max_iter=20000,random_state=args.seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x,y)
        ba=float(balanced_accuracy_score(yv,model.predict(xv)));candidates.append(dict(C=c,dev_clean_balanced_accuracy=ba))
        if ba>best_score:best,best_score,best_c=model,ba,c
    bundle=dict(model=best,feature_config=FEATURE_CONFIG,classes={0:'genuine',1:'synthetic'},
                score_type='uncalibrated SVM margin; positive means synthetic',threshold=0.0,
                seed=args.seed,train_conditions=conditions,C=best_c,manifest_sha256=digest(args.manifest),
                train_sha256=[r['sha256'] for r in tr],train_groups=sorted(set(r['group'] for r in tr)),
                versions={v:importlib.metadata.version(v) for v in ['numpy','scipy','librosa','scikit-learn','soundfile','joblib']})
    joblib.dump(bundle,out/'model.joblib');write_csv(out/'split_manifest.csv',rows,list(rows[0]))
    save_json(out/'training.json', {k:v for k,v in bundle.items() if k!='model'} | dict(
        selection_candidates=candidates,training_rows=len(y),elapsed_seconds=time.perf_counter()-started,
        caution='One seed and development results only; no generalization or improvement established.'))
    print('Saved',out/'model.joblib','; selected C=',best_c,'on CLEAN DEVELOPMENT data.')
    evaluate_model(bundle,rows,args.data,out/'dev_results','dev',args.include_opus,args.seed)


def evaluate_model(bundle,rows,root,out,partition,include_opus,seed):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected=[r for r in rows if r['split']==partition]
    conditions=['clean','noise20','noise10']+(['opus24','noise10_opus24'] if include_opus else [])
    reports={}; all_predictions=[]
    for condition in conditions:
        x,y,meta=matrix(selected,root,[condition],seed,bundle['feature_config'])
        scores=bundle['model'].decision_function(x); report=metrics(y,scores);reports[condition]=report
        for r,target,score in zip(meta,y,scores):
            all_predictions.append(r|dict(label=int(target),margin=float(score),prediction=int(score>=0),split=partition))
        print(partition,condition,'balanced_accuracy=',round(report['balanced_accuracy'],4),
              'FPR=',round(report['false_positive_rate'],4),'EER=',round(report['eer'],4))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay(np.array(report['confusion_matrix']),display_labels=['genuine','synthetic']).plot(cmap='Blues',colorbar=False)
        plt.title(partition+' / '+condition);plt.tight_layout();plt.savefig(out/(condition+'_confusion.png'),dpi=150);plt.close()
    save_json(out/'metrics.json',dict(partition=partition,conditions=reports,
        units='Fractions 0..1; multiply by 100 for percentages.',
        limitation='Gaussian noise is a pilot corruption, not environmental noise. No confidence intervals in this starter.'))
    write_csv(out/'predictions.csv',all_predictions,list(all_predictions[0]))


def evaluate(args):
    rows=read_csv(args.manifest);validate(rows)
    bundle=joblib.load(args.model)  # Load ONLY a model file you created/trust.
    if digest(args.manifest)!=bundle['manifest_sha256']:
        raise ValueError('Manifest differs from the one used during training.')
    if args.partition=='test' and not args.final_test:
        raise ValueError('Final test is reserved. Add --final-test only after freezing model and study settings.')
    if args.include_opus:opus_roundtrip(np.ones(SR,dtype=np.float32)*.01)
    evaluate_model(bundle,rows,args.data,args.out,args.partition,args.include_opus,args.seed)


def predict(args):
    bundle=joblib.load(args.model)
    y=audio(args.audio,bundle['feature_config']);x=features(y,bundle['feature_config'])[None,:]
    score=float(bundle['model'].decision_function(x)[0])
    print(json.dumps(dict(prediction='synthetic_speech_suspected' if score>=0 else 'no_strong_synthetic_evidence',
                         svm_margin=score,analyzed_seconds=len(y)/SR,
                         note='Uncalibrated margin, not a probability. First six seconds only; no identity/truth verification.'),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan',help='Audit local files and infer labels/speaker folder IDs.');s.add_argument('--data',required=True);s.add_argument('--out',default='audit');s.add_argument('--max-per-group',type=int,default=0);s.set_defaults(func=scan)
    s=sub.add_parser('split',help='Freeze a group-disjoint train/dev/test manifest.');s.add_argument('--manifest',required=True);s.add_argument('--out',default='splits.csv');s.add_argument('--groups-reviewed',action='store_true');s.add_argument('--seed',type=int,default=42);s.set_defaults(func=split)
    s=sub.add_parser('train',help='Train a clean or augmented SVM; report DEVELOPMENT results only.');s.add_argument('--data',required=True);s.add_argument('--manifest',required=True);s.add_argument('--out',required=True);s.add_argument('--augment',action='store_true');s.add_argument('--include-opus',action='store_true');s.add_argument('--seed',type=int,default=42);s.set_defaults(func=train)
    s=sub.add_parser('evaluate',help='Evaluate the exact saved manifest and model.');s.add_argument('--data',required=True);s.add_argument('--manifest',required=True);s.add_argument('--model',required=True);s.add_argument('--out',required=True);s.add_argument('--partition',choices=['dev','test'],default='dev');s.add_argument('--final-test',action='store_true');s.add_argument('--include-opus',action='store_true');s.add_argument('--seed',type=int,default=42);s.set_defaults(func=evaluate)
    s=sub.add_parser('predict',help='Check an audio file with your trained model.');s.add_argument('--model',required=True);s.add_argument('--audio',required=True);s.set_defaults(func=predict)
    args=parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        parser.exit(1, f'ERROR: {exc}\n')

if __name__=='__main__':main()
