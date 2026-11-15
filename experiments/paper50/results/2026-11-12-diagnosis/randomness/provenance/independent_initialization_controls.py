"""Fixed data/initialization robustness matrix for the published Adam settings.

Independent diagnostics, not authors' data or code. Uses the audited frozen
scaled recurrence from the course repository; all inputs are regenerated from
explicit PCG64 seeds. Every configured run is retained, with no outcome selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy

FROZEN_SHA='cff71403539c260aa910a48a7332900c25d69b47be74ab464aac43d40c6fdc67'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(repo):
    path=Path(repo)/'experiments/paper50/run.py'
    if digest(path)!=FROZEN_SHA: raise ValueError('Audited recurrence changed')
    spec=importlib.util.spec_from_file_location('paper50_frozen',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def configurations(steps):
    values=[('independent_initialization',2,seed) for seed in range(20000,20005)]
    return [dict(family=family,data_seed=seed,initialization_seed=init,epsilon=eps,
                 steps=steps,run_id=f'{family}-data{seed:02d}-init{init:05d}-eps{eps:g}')
            for family,seed,init in values for eps in [0.,1e-8]]

def input_arrays(seed,init):
    rng=np.random.Generator(np.random.PCG64(seed))
    x=rng.standard_normal((50,50)); y=rng.choice([-1,1],50)
    w=np.random.Generator(np.random.PCG64(init)).standard_normal(50)
    return x,y,w

def simulate(task):
    repo,root,config,reference=task
    root=Path(root); core=module(repo)
    x,y,w=input_arrays(config['data_seed'],config['initialization_seed'])
    z=y[:,None]*x
    initial=w.copy(); m=np.zeros(50); v=np.zeros(50); scale=0.
    steps=config['steps']; eps=config['epsilon']
    targets=set(map(int,np.geomspace(1,steps,64))) | {0,steps}
    row_values=[]; ws=[]; ms=[]; vs=[]; scales=[]; saved=[]
    begin=time.perf_counter(); cpu=time.process_time(); completed=0
    reason='completed_fixed_steps'; error=None
    def record(t):
        margins=z@w; raw=float(margins.min()); qi=raw/float(np.abs(w).max()); q2=raw/float(np.linalg.norm(w))
        row=dict(step=t,margin_linf=qi,margin_l2=q2,relative_gap_linf=1-qi/reference['gamma_linf'],
                 relative_gap_l2=1-q2/reference['gamma_l2'],margin_raw=raw,norm_linf=float(np.abs(w).max()),
                 log10_min_sqrt_v=None,log10_max_sqrt_v=None,epsilon_below_coordinates=None,
                 epsilon_strong_coordinates=None,update_suppression_min=None)
        if t:
            logs=.5*np.log(v)-scale
            row['log10_min_sqrt_v']=float(logs.min()/math.log(10)); row['log10_max_sqrt_v']=float(logs.max()/math.log(10))
            if eps:
                ratios=logs-math.log(eps)
                row['epsilon_below_coordinates']=int(np.sum(ratios<0))
                row['epsilon_strong_coordinates']=int(np.sum(ratios<=math.log(.1)))
                row['update_suppression_min']=float(1/(1+np.exp(ratios.max())))
        row_values.append(row); saved.append(t); ws.append(w.copy());ms.append(m.copy());vs.append(v.copy());scales.append(scale)
    record(0)
    try:
        for t in range(steps):
            w,m,v,scale=core.scaled_adam_step(z,w,m,v,scale,.9,.99,(t+1)**(-.5),eps)
            completed=t+1
            if completed in targets:
                record(completed)
                if time.perf_counter()-begin>600 or time.process_time()-cpu>600:
                    reason='budget';break
    except (ArithmeticError,OverflowError,FloatingPointError) as exc:
        reason='numerical_failure';error=repr(exc)
    if saved[-1]!=completed:record(completed)
    folder=root/config['run_id'];folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'checkpoints.npz',steps=saved,weights=ws,m_scaled=ms,v_scaled=vs,log_scales=scales)
    with (folder/'trajectory.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row_values[0]));writer.writeheader();writer.writerows(row_values)
    result=dict(config=config,updates=completed,stop_reason=reason,error=error,
                endpoint=row_values[-1],initialization_array_sha256=core.array_sha(initial),
                wall_seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
                first_all_epsilon_below_recorded_checkpoint=next((r['step'] for r in row_values if r['epsilon_below_coordinates']==50),None),
                first_all_epsilon_strong_recorded_checkpoint=next((r['step'] for r in row_values if r['epsilon_strong_coordinates']==50),None),
                files={p.name:digest(p) for p in folder.iterdir() if p.is_file()})
    save_json(folder/'summary.json',result)
    print(json.dumps({'run':config['run_id'],'gap_linf':result['endpoint']['relative_gap_linf'],'reason':reason,'wall':result['wall_seconds']}),flush=True)
    return result

def verify(root,repo):
    root=Path(root); core=module(repo)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete matrix')
    if manifest['source_sha256']!=digest(__file__):raise ValueError('Runner source changed')
    expected=configurations(manifest['steps_per_run'])
    actual=sorted([r['config'] for r in manifest['results']],key=lambda x:x['run_id'])
    if actual!=sorted(expected,key=lambda x:x['run_id']):raise ValueError('Matrix changed')
    checkpoint_count=0;max_error=0.
    for r in manifest['results']:
        cfg=r['config'];folder=root/cfg['run_id']; ref=manifest['references'][str(cfg['data_seed'])]['reference']
        if r['updates']!=cfg['steps'] or r['stop_reason']!='completed_fixed_steps':raise ValueError('Incomplete run')
        for filename,expected_hash in r['files'].items():
            if digest(folder/filename)!=expected_hash:raise ValueError('File changed')
        x,y,w0=input_arrays(cfg['data_seed'],cfg['initialization_seed']);z=y[:,None]*x
        if core.array_sha(w0)!=r['initialization_array_sha256']:raise ValueError('Initialization changed')
        with (folder/'trajectory.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        with np.load(folder/'checkpoints.npz') as state:
            if state['steps'].tolist()!=[int(row['step']) for row in rows]:raise ValueError('Checkpoint indices differ')
            if not np.array_equal(state['weights'][0],w0):raise ValueError('Stored initialization differs')
            for j,row in enumerate(rows):
                w=state['weights'][j];raw=float((z@w).min()); qi=raw/float(np.abs(w).max());q2=raw/float(np.linalg.norm(w))
                for key,value in [('margin_linf',qi),('margin_l2',q2),('relative_gap_linf',1-qi/ref['gamma_linf']),('relative_gap_l2',1-q2/ref['gamma_l2'])]:
                    err=abs(value-float(row[key]));max_error=max(max_error,err)
                    if err>2e-10:raise ValueError('Margin differs')
                checkpoint_count+=1
    return dict(passed=True,runs=len(manifest['results']),checkpoints_recomputed=checkpoint_count,max_margin_recomputation_error=max_error)

def run(args):
    repo=str(args.repo.resolve());core=module(repo);root=args.output
    root.mkdir(parents=True,exist_ok=False)
    refs={}
    for seed in range(10):
        x,y,w0=input_arrays(seed,seed+10000)
        refs[str(seed)]=dict(reference=core.references(y[:,None]*x),X_array_sha256=core.array_sha(x),y_array_sha256=core.array_sha(y))
    configs=configurations(args.steps)
    header=dict(status='running',steps_per_run=args.steps,planned_conditions=configs,references=refs,
                source_sha256=digest(__file__),frozen_source_sha256=FROZEN_SHA,
                settings=dict(n=50,d=50,beta1=.9,beta2=.99,c=1,a=.5,index_offset=1,bias_correction=False,loss='full_batch_mean_logistic',epsilon_location='outside_sqrt',dtype='float64'),
                environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),thread_settings={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),
                design='Five fixed disjoint PCG64 initializations 20000-20004 on dataset2; both epsilons; supplements original matrix because original init seed2 reused data seed2')
    save_json(root/'manifest.json',header)
    tasks=[(repo,str(root),c,refs[str(c['data_seed'])]['reference']) for c in configs]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(simulate,tasks))
    header['results']=results
    header['status']='completed' if all(r['stop_reason']=='completed_fixed_steps' for r in results) else 'incomplete'
    save_json(root/'manifest.json',header)
    save_json(root/'verification.json',verify(root,repo))
    save_json(root/'endpoints.json',[dict(**r['config'],**r['endpoint']) for r in results])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['run','verify'])
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=1000000);ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args()
    if args.steps<1 or args.jobs<1:ap.error('steps/jobs must be positive')
    if args.command=='run':run(args)
    else:print(json.dumps(verify(args.output,args.repo),indent=2))
