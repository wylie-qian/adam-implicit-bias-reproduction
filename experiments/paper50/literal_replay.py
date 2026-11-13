"""Independent full-history literal Adam replay in native long double.

No moment rescaling is used. Requires NumPy and a C++17 compiler; x86 native
extended precision is checked explicitly. Outputs are always new directories.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / 'results/2026-11-12-multirate'
DEFAULT_OUTPUT = HERE / 'results/2026-11-13-literal-replay'
CONFIGS = [('0.5', 'zero_epsilon', 'zero', '0'), ('0.3', 'fixed_epsilon', 'fixed', '1e-8'),
           ('0.5', 'fixed_epsilon', 'fixed', '1e-8'), ('0.7', 'fixed_epsilon', 'fixed', '1e-8'),
           ('1', 'fixed_epsilon', 'fixed', '1e-8')]
FLAGS = ['-O3', '-std=c++17', '-fno-fast-math']

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def sources(): return {p.name:digest(p) for p in [HERE/'literal_replay.cpp', Path(__file__)]}

def validate_evidence(evidence):
    """Bind the frozen inputs, selected runs, and reference to their manifest."""
    manifest=json.loads((evidence/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['steps_per_condition']!=1000000:
        raise ValueError('Baseline evidence is incomplete')
    files={row['path']:row['sha256'] for row in manifest['files']}
    if len(files)!=len(manifest['files']): raise ValueError('Duplicate baseline evidence files')
    selected=['data.json','reference.json']+[f'a-{a}_{mode}.json' for a,mode,_,_ in CONFIGS]
    for name in selected:
        if name not in files or digest(evidence/name)!=files[name]:
            raise ValueError('Baseline evidence hash mismatch: '+name)
    for name in ['data.json','reference.json']:
        if digest(evidence/name)!=manifest['input_sha256'][name]:
            raise ValueError('Baseline input hash mismatch: '+name)
    data=json.loads((evidence/'data.json').read_text())
    reference=json.loads((evidence/'reference.json').read_text())
    for name,array in [('X',np.asarray(data['X'],dtype=np.float64)),
                       ('y',np.asarray(data['y'],dtype=data.get('y_dtype','int64'))),
                       ('w0',np.asarray(data['w0'],dtype=np.float64))]:
        array=np.ascontiguousarray(array)
        h=hashlib.sha256();h.update(str(array.dtype).encode());h.update(json.dumps(array.shape).encode());h.update(array.tobytes())
        if h.hexdigest()!=reference.get(name+'_array_sha256'):
            raise ValueError('Baseline reference/data identity mismatch: '+name)
    planned={c['run_id']:c for c in manifest['planned_configs']}
    for a,mode,short,eps in CONFIGS:
        run_id=f'a-{a}_{mode}'
        run=json.loads((evidence/(run_id+'.json')).read_text())
        cfg=run['config']
        if (cfg!=planned.get(run_id) or cfg['a']!=float(a) or cfg['mode']!=mode
            or cfg['epsilon']!=float(eps) or cfg['beta1']!=.9 or cfg['beta2']!=.99
            or cfg['c']!=1. or cfg['time_offset']!=1 or cfg['bias_correction']
            or run['status']!='completed' or run['completed_updates']!=1000000
            or run['requested_updates']!=1000000 or run['endpoint']!=run['trace'][-1]):
            raise ValueError('Baseline condition identity/completion mismatch: '+run_id)

def make_input(evidence):
    validate_evidence(evidence)
    data=json.loads((evidence/'data.json').read_text())
    run=json.loads((evidence/'a-0.5_zero_epsilon.json').read_text())
    steps=[r['step'] for r in run['trace']]
    z=np.asarray(data['y'], dtype=np.float64)[:,None]*np.asarray(data['X'], dtype=np.float64)
    w=np.asarray(data['w0'], dtype=np.float64)
    if z.shape!=(50,50) or w.shape!=(50,) or not np.isfinite(z).all() or not np.isfinite(w).all():
        raise ValueError('Expected frozen finite 50x50 input')
    if steps[0]!=0 or steps[-1]!=1000000 or steps!=sorted(set(steps)):
        raise ValueError('Unexpected checkpoint schedule')
    lines=[f'50 50 {len(steps)}', ' '.join(map(str,steps))]
    lines+=[' '.join(repr(float(x)) for x in row) for row in z]
    lines+=[' '.join(repr(float(x)) for x in w)]
    return '\n'.join(lines)+'\n'

def compare(evidence, output):
    if np.finfo(np.longdouble).nmant < 63 or np.finfo(np.longdouble).maxexp < 16384:
        raise RuntimeError('Verifier requires native extended precision/range')
    data=json.loads((evidence/'data.json').read_text())
    wrapped=json.loads((evidence/'reference.json').read_text()); ref=wrapped.get('reference',wrapped)
    z=np.asarray(data['y'])[:,None]*np.asarray(data['X'])
    reports=[]
    for a,mode,short,eps in CONFIGS:
        name=f'a-{a}_{short}'
        run=json.loads((evidence/f'a-{a}_{mode}.json').read_text())
        meta=json.loads((output/f'{name}_run.json').read_text())
        if meta['completed_steps']!=1000000 or meta['LDBL_MANT_DIG']<64 or meta['LDBL_MAX_EXP']<16384:
            raise ValueError('Incomplete replay or unsupported arithmetic')
        if meta['epsilon']!=float(eps) or meta['a']!=float(a): raise ValueError('Config mismatch')
        with gzip.open(output/f'{name}.csv.gz','rt') as f: rows=list(csv.DictReader(f))
        if len(rows)!=len(run['trace']) or len(rows)!=meta['checkpoints']: raise ValueError('Checkpoint count mismatch')
        metrics=[]
        for x,saved in zip(rows,run['trace']):
            if int(x['step'])!=saved['step']: raise ValueError('Checkpoint step mismatch')
            arrays=[np.array([x[f'{label}{i}'] for i in range(50)],dtype=np.longdouble) for label in ['w','m','v']]
            w,m,v=arrays
            if not all(np.isfinite(vv).all() for vv in arrays): raise ValueError('Nonfinite state')
            if saved['step'] and not (v>0).all(): raise ValueError('Nonpositive raw moment')
            b=np.longdouble(saved['log_scale_applied_on_last_update'])
            saved_arrays=[np.asarray(saved[k],dtype=np.longdouble) for k in ['weights','first_moment_scaled','second_moment_scaled']]
            # Convert reference states to raw units in extended precision.
            sw,sm,sv=saved_arrays[0],saved_arrays[1]*np.exp(-b),saved_arrays[2]*np.exp(-2*b)
            def relative_inf(u,vv):
                den=max(np.max(np.abs(u)),np.max(np.abs(vv)))
                return float(np.max(np.abs(u-vv))/den) if den else 0.
            wd=np.asarray(w,dtype=np.float64)
            gap=1-float(np.min(z@wd)/np.max(np.abs(wd)))/ref['gamma_linf']
            metrics.append(dict(step=saved['step'],max_weight_absolute_difference=float(np.max(np.abs(w-sw))),
                weight_relative_linf_difference=relative_inf(w,sw),first_moment_relative_linf_difference=relative_inf(m,sm),
                second_moment_relative_linf_difference=relative_inf(v,sv),relative_gap_difference=abs(gap-saved['relative_gap_linf']),
                relative_gap=gap,min_log10_raw_second_moment=float(np.log10(v.min())) if saved['step'] else None))
        keys=[k for k in metrics[0] if k.endswith('difference')]
        reports.append(dict(a=float(a),mode=mode,checkpoints=len(rows),raw_run=meta,
            maximum_differences={k:max(x[k] for x in metrics) for k in keys},endpoint=metrics[-1],trace=metrics))
    return dict(schema_version=1,comparison='Independent native long-double raw recurrence versus retained binary64 scaled recurrence',
        relative_norm_definition='||raw-replay minus raw-reference||_infinity / max(||raw-replay||_infinity, ||raw-reference||_infinity); zero for two zero vectors',
        arithmetic_difference='Replay uses long-double decimal beta constants, arithmetic, powers and scalar reductions. Baseline uses binary64 constants, NumPy/BLAS and joint moment scaling. Differences are reported without silently treating them as zero.',
        conditions=reports)

def input_hashes(evidence):
    names=['data.json','reference.json']+[f'a-{a}_{mode}.json' for a,mode,_,_ in CONFIGS]
    return {n:digest(evidence/n) for n in names}

def create_manifest(evidence,output,compiler_version,orchestration="literal_replay.py run"):
    return dict(schema_version=1,study_type='Full-history direct recurrence audit; a=.5 zero/fixed epsilon followed by remaining fixed-epsilon rates',
        parameters=dict(beta1='0.9L',beta2='0.99L',epsilon_placement='outside_sqrt',bias_correction=False,
                        loss='full_batch_mean_logistic',initial_moments='zero',rate='t^(-a), t=1..1000000'),
        configs=[dict(a=a,mode=mode,output_prefix=f'a-{a}_{short}',epsilon=eps) for a,mode,short,eps in CONFIGS],
        source_hashes=sources(),source_roles={'literal_replay.cpp':'Executed production recurrence; hash identifies exact source bytes.',
            'literal_replay.py':'Packaging, verification and future reproduction entrypoint; historical orchestration recorded separately.'},
        execution_provenance=orchestration,production_cpp_sha256=digest(HERE/'literal_replay.cpp'),
        input_hashes=input_hashes(evidence),compiler_version=compiler_version,
        compiler_flags=FLAGS,environment=dict(platform=platform.platform(),numpy=np.__version__),
        files={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'},
        limitations=['Native extended precision is about 19 decimal digits, not arbitrary precision.',
                     'Some negligible per-example logistic weights may underflow in the zero-epsilon run; counters and first steps are retained.',
                     'Full moment histories were replayed, but only 61 checkpoints per condition are retained.',
                     'This audits our own data and recurrence, not the unavailable author implementation.'])

def run(evidence,output,compiler):
    macro_text=subprocess.run([compiler,*FLAGS,'-dM','-E','-x','c++','-'],input='',text=True,capture_output=True,check=True).stdout
    macros={line.split()[1]:line.split()[2] for line in macro_text.splitlines() if len(line.split())>=3}
    if int(macros.get('__LDBL_MANT_DIG__','0'))<64 or int(macros.get('__LDBL_MAX_EXP__','0'))<16384:
        raise RuntimeError('Compiler must provide native extended precision/range')
    input_text=make_input(evidence)
    output.mkdir(parents=True,exist_ok=False)
    (output/'input.txt').write_text(input_text)
    version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    with tempfile.TemporaryDirectory(prefix='literal-adam-') as td:
        binary=Path(td)/'replay'
        subprocess.run([compiler,*FLAGS,str(HERE/'literal_replay.cpp'),'-o',str(binary)],check=True)
        for a,mode,short,eps in CONFIGS:
            stem=f'a-{a}_{short}'
            raw=output/f'{stem}.csv'
            completed=subprocess.run([str(binary),str(output/'input.txt'),str(raw),a,eps,'1000000'],text=True,capture_output=True,check=True)
            meta=json.loads(completed.stdout); write_json(output/f'{stem}_run.json',meta)
            with (output/f'{stem}.csv.gz').open('xb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g: g.write(raw.read_bytes())
            raw.unlink()
            print(json.dumps(dict(condition=stem,**meta)),flush=True)
    write_json(output/'comparison.json',compare(evidence,output))
    write_json(output/'manifest.json',create_manifest(evidence,output,version))
    return verify(evidence,output)

def verify(evidence,output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['source_hashes']!=sources() or manifest['input_hashes']!=input_hashes(evidence): raise ValueError('Source/input hash mismatch')
    if (output/'input.txt').read_text()!=make_input(evidence): raise ValueError('Frozen input mismatch')
    actual={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}
    if actual!=manifest['files']: raise ValueError('Output hash/file mismatch')
    recomputed=compare(evidence,output)
    if recomputed!=json.loads((output/'comparison.json').read_text()): raise ValueError('Comparison mismatch')
    return dict(verified=True,conditions=len(CONFIGS),retained_states=sum(x['checkpoints'] for x in recomputed['conditions']),
                scope='Checks file/source identities and recomputes comparisons; does not rerun million-step histories.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--evidence',type=Path,default=DEFAULT_EVIDENCE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--compiler',default=os.environ.get('CXX','g++'));a=p.parse_args()
    print(json.dumps(run(a.evidence,a.output,a.compiler) if a.command=='run' else verify(a.evidence,a.output),indent=2))
