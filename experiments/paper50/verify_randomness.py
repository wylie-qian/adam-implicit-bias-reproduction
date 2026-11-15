"""Build/verify the compact 46-run randomness audit, or run regression checks.

Every run retains six checkpoint indices chosen solely by its checkpoint count,
including initial and final states. One aggregate NPZ avoids per-run binary files.
Original full audit records describe checks performed before compaction; they are
not represented as newly verified histories after the omitted states are removed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from run import HERE, array_sha, sha, write_json
from randomness import input_arrays, configurations, FROZEN_SHA
from legacy_rng import ROOT_SEEDS, candidates

DEFAULT = HERE / 'results/2026-11-12-diagnosis/randomness'


def retained_indices(length):
    if length < 6:
        raise ValueError('At least six checkpoints required')
    return np.linspace(0, length - 1, 6, dtype=int).tolist()


def read_json(path):
    return json.loads(Path(path).read_text())


def parse_row(row):
    numeric = {}
    for key, value in row.items():
        if value == '': numeric[key] = None
        else:
            try: numeric[key] = float(value)
            except ValueError: numeric[key] = value
    numeric['step'] = int(numeric['step'])
    return numeric


def compact(original, independent, legacy, clean_source, output):
    full = {name: read_json(root/'manifest.json') for name, root in
            [('original', original), ('independent', independent), ('legacy', legacy)]}
    if any(m['status'] != 'completed' for m in full.values()):
        raise ValueError('All full studies must complete before compaction')
    if full['original']['source_sha256'] != sha(HERE/'randomness.py') or full['independent']['source_sha256'] != sha(clean_source) or full['legacy']['source_sha256'] != sha(HERE/'legacy_rng.py'):
        raise ValueError('Full-run source does not match preserved source')
    output.mkdir(parents=True, exist_ok=False)
    (output/'provenance').mkdir()
    shutil.copyfile(clean_source, output/'provenance/independent_initialization_controls.py')
    manifest = dict(schema_version=1, kind='compact_randomness_audit',
                    retention='Six linspace indices from each original checkpoint sequence, including initial/final; no outcome selection',
                    sources={'randomness.py':sha(HERE/'randomness.py'), 'legacy_rng.py':sha(HERE/'legacy_rng.py'),
                             'run.py':FROZEN_SHA, 'alignment.py':sha(HERE/'alignment.py'),
                             'provenance/independent_initialization_controls.py':sha(clean_source)},
                    full_audit_records={name:read_json(root/'verification.json') for name,root in [('original',original),('independent',independent),('legacy',legacy)]},
                    full_run_source_manifests={name:sha(root/'manifest.json') for name,root in [('original',original),('independent',independent),('legacy',legacy)]},
                    datasets={}, legacy_selections=[], runs=[])
    for seed, entry in full['original']['references'].items():
        manifest['datasets'][f'pcg64-{seed}'] = dict(generator='PCG64', seed=int(seed), **entry)
    tasks=[]
    for suite, root in [('original',original),('independent',independent)]:
        for r in full[suite]['results']:
            c=r['config']; collision=c['data_seed']==c['initialization_seed']
            family=('dataset' if c['family']=='dataset' else 'correlated_initialization' if collision else 'initialization' if suite=='original' else 'independent_initialization')
            tasks.append((dict(run_id=c['run_id'],family=family,data_key=f"pcg64-{c['data_seed']}",initialization_generator='PCG64',initialization_seed=c['initialization_seed'],epsilon=c['epsilon'],steps=c['steps'],source_suite=suite,seed_collision=collision),r,root/c['run_id'],'checkpoints.npz'))
    for case in full['legacy']['cases']:
        keys=[]
        for c in case['candidates']:
            key=f"mt19937-{case['root_seed']}-{c['index']}";keys.append(key)
            manifest['datasets'][key]=dict(generator='MT19937',root_seed=case['root_seed'],index=c['index'],**{k:v for k,v in c.items() if k!='index'})
        manifest['legacy_selections'].append(dict(root_seed=case['root_seed'],candidate_keys=keys,selected_index=case['selected_index']))
        for item in full['legacy']['results']:
            if item['root_seed']!=case['root_seed']:continue
            r=item['result'];c=r['config']
            tasks.append((dict(run_id=f"mt19937-root{case['root_seed']}-{c['run_id']}",family='legacy_rng',data_key=keys[case['selected_index']],initialization_generator='MT19937',initialization_seed=case['initialization_seed'],epsilon=c['epsilon'],steps=full['legacy']['steps_per_run'],source_suite='legacy',seed_collision=False),r,legacy/case['run_id']/c['run_id'],'states.npz'))
    arrays={k:[] for k in ['steps','weights','m_scaled','v_scaled','log_scales']}
    for info,r,folder,filename in tasks:
        hashes=r.get('files',r.get('hashes'))
        for name,digest in hashes.items():
            if sha(folder/name)!=digest:raise ValueError('Original evidence hash mismatch')
        with (folder/'trajectory.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        with np.load(folder/filename) as state:
            indices=retained_indices(len(state['steps']))
            for key in arrays:arrays[key].append(state[key][indices])
        info.update(state_index=len(manifest['runs']),original_recorded_checkpoints=len(rows),retained_indices=indices,
                    updates=r['updates'],stop_reason=r['stop_reason'],endpoint=r['endpoint'],
                    checkpoints=[parse_row(rows[i]) for i in indices],original_file_hashes=hashes,
                    wall_seconds=r['wall_seconds'],cpu_seconds=r['cpu_seconds'])
        manifest['runs'].append(info)
    np.savez_compressed(output/'compact_states.npz',**{k:np.stack(v) for k,v in arrays.items()})
    manifest['files']={'compact_states.npz':sha(output/'compact_states.npz')}
    write_json(output/'manifest.json',manifest)
    write_json(output/'verification.json',verify(output))
    write_json(output/'summary.json',summarize(manifest))


def certificate(z,ref):
    n,d=z.shape;u=np.array(ref['w_linf']);w=np.array(ref['w_l2_svm']);gamma=ref['gamma_linf']
    lp=ref['lp'];ineq=np.array(lp['inequality_marginals']);lower=np.array(lp['lower_marginals']);upper=np.array(lp['upper_marginals'])
    objective=np.r_[np.zeros(d),-1.];constraint=np.c_[-z,np.ones(n)]
    alpha=np.array(ref['qp']['alpha'])
    checks=[max(0.,gamma-(z@u).min(),np.abs(u).max()-1),abs(-gamma-(-lower.sum()+upper.sum())),
            np.abs(objective-constraint.T@ineq-np.r_[lower+upper,0.]).max(),max(0.,ineq.max(),-lower.min(),upper.max()),
            np.abs(ineq*(z@u-gamma)).max(),np.abs(lower*(u+1)).max(),np.abs(upper*(1-u)).max(),
            max(0.,1-(z@w).min()),np.abs(w-z.T@alpha).max(),max(0.,-alpha.min()),
            np.abs(alpha*(z@w-1)).max(),abs(.5*np.dot(w,w)-(alpha.sum()-.5*np.linalg.norm(z.T@alpha)**2)),
            abs(ref['gamma_l2']-1/np.linalg.norm(w)),
            abs(ref['direction_separation_l2']-np.linalg.norm(w/np.linalg.norm(w)-u/np.linalg.norm(u)))]
    if max(checks)>1e-7:raise ValueError('Reference certificate failed')
    return float(max(checks))


def expected_design():
    expected={(f'pcg64-{c["data_seed"]}',c['initialization_seed'],c['epsilon'],'PCG64') for c in configurations(1)}
    expected|={('pcg64-2',seed,eps,'PCG64') for seed in range(20000,20005) for eps in [0.,1e-8]}
    return expected


def verify(root):
    manifest=read_json(root/'manifest.json')
    for filename,digest in manifest['files'].items():
        if sha(root/filename)!=digest:raise ValueError('Compact evidence hash mismatch')
    for filename,digest in manifest['sources'].items():
        path=root/filename if filename.startswith('provenance/') else HERE/filename
        if sha(path)!=digest:raise ValueError('Preserved scientific source changed')
    arrays={}; residuals=[];legacy_arrays={seed:candidates(seed) for seed in ROOT_SEEDS}
    for key,entry in manifest['datasets'].items():
        if entry['generator']=='PCG64':x,y,_=input_arrays(entry['seed'],0)
        else:x,y=legacy_arrays[entry['root_seed']][entry['index']]
        if array_sha(x)!=entry['X_array_sha256'] or array_sha(y)!=entry['y_array_sha256']:raise ValueError('Input generation changed')
        arrays[key]=(x,y);residuals.append(certificate(y[:,None]*x,entry['reference']))
    expected=expected_design()
    for selection in manifest['legacy_selections']:
        keys=selection['candidate_keys']
        chosen=max(range(10),key=lambda i:(manifest['datasets'][keys[i]]['reference']['direction_separation_l2'],-i))
        if chosen!=selection['selected_index']:raise ValueError('Geometry selection differs')
        expected|={(keys[chosen],selection['root_seed']+10000,eps,'MT19937') for eps in [0.,1e-8]}
    actual={(r['data_key'],r['initialization_seed'],r['epsilon'],r['initialization_generator']) for r in manifest['runs']}
    if actual!=expected or len(manifest['runs'])!=46:raise ValueError('Fixed matrix differs')
    max_error=0.;checked=0
    with np.load(root/'compact_states.npz') as states:
        for i,r in enumerate(manifest['runs']):
            if r['state_index']!=i or r['retained_indices']!=retained_indices(r['original_recorded_checkpoints']):raise ValueError('Retention policy differs')
            if r['updates']!=r['steps'] or r['stop_reason']!='completed_fixed_steps':raise ValueError('Incomplete original run')
            x,y=arrays[r['data_key']];z=y[:,None]*x;ref=manifest['datasets'][r['data_key']]['reference']
            if r['initialization_generator']=='PCG64':w0=np.random.Generator(np.random.PCG64(r['initialization_seed'])).standard_normal(50)
            else:w0=np.random.RandomState(r['initialization_seed']).standard_normal(50)
            if not np.array_equal(states['weights'][i,0],w0):raise ValueError('Initial weights differ')
            if r['seed_collision'] != (r['data_key']==f"pcg64-{r['initialization_seed']}"):raise ValueError('Collision disclosure differs')
            if states['steps'][i].tolist()!=[c['step'] for c in r['checkpoints']]:raise ValueError('Checkpoint indices differ')
            for j,row in enumerate(r['checkpoints']):
                w=states['weights'][i,j];raw=float((z@w).min());qi=raw/np.abs(w).max();q2=raw/np.linalg.norm(w)
                for key,value in [('margin_linf',qi),('margin_l2',q2),('relative_gap_linf',1-qi/ref['gamma_linf']),('relative_gap_l2',1-q2/ref['gamma_l2'])]:
                    error=abs(value-row[key]);max_error=max(max_error,error)
                    if error>2e-10:raise ValueError('Retained margin differs')
                    if j==5 and abs(value-r['endpoint'][key])>2e-10:raise ValueError('Endpoint differs')
                if row['step'] and r['epsilon']:
                    ratio=.5*np.log(states['v_scaled'][i,j])-states['log_scales'][i,j]-math.log(r['epsilon'])
                    if int(np.sum(ratio<=math.log(.1)))!=row['epsilon_strong_coordinates']:raise ValueError('Epsilon diagnostic differs')
                checked+=1
    return dict(passed=True,runs=46,total_updates=sum(r['updates'] for r in manifest['runs']),datasets_regenerated=len(arrays),reference_certificates_checked=len(residuals),max_reference_residual=max(residuals),retained_checkpoints_recomputed=checked,max_margin_recomputation_error=max_error,
                note='Only retained checkpoints reverified here; full-history audits are separately recorded, not rerun')


def summarize(manifest):
    results=[dict(run_id=r['run_id'],family=r['family'],data_key=r['data_key'],initialization_seed=r['initialization_seed'],epsilon=r['epsilon'],relative_gap_linf=r['endpoint']['relative_gap_linf'],relative_gap_l2=r['endpoint']['relative_gap_l2'],seed_collision=r['seed_collision']) for r in manifest['runs']]
    families=[]
    for family in sorted({r['family'] for r in results}):
        for eps in [0.,1e-8]:
            values=[r['relative_gap_linf'] for r in results if r['family']==family and r['epsilon']==eps]
            families.append(dict(family=family,epsilon=eps,runs=len(values),relative_gap_linf_range=[min(values),max(values)],within_one_percent=sum(v<=.01 for v in values)))
    return dict(total_updates=sum(r['updates'] for r in manifest['runs']),all_outcomes_retained=True,families=families,results=results)


class ScientificInvariants(unittest.TestCase):
    def test_data_initialization_collision_is_real_and_disclosed(self):
        x,_,w0=input_arrays(2,2)
        np.testing.assert_array_equal(x[0],w0)
        for seed in range(20000,20005):self.assertFalse(np.array_equal(x[0],input_arrays(2,seed)[2]))
    def test_matrix_and_retention_are_fixed_before_outcomes(self):
        self.assertEqual(len(expected_design()),40)
        self.assertEqual(retained_indices(64),[0,12,25,37,50,63])
        self.assertEqual(retained_indices(6),list(range(6)))
    def test_compact_hash_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'compact_states.npz';p.write_bytes(b'original');digest=sha(p);p.write_bytes(b'changed')
            write_json(root/'manifest.json',{'files':{p.name:digest}})
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify(root)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['compact','verify','test']);ap.add_argument('--output',type=Path,default=DEFAULT)
    ap.add_argument('--original',type=Path);ap.add_argument('--independent',type=Path);ap.add_argument('--legacy',type=Path);ap.add_argument('--clean-source',type=Path)
    args=ap.parse_args()
    if args.command=='test':unittest.main(argv=[__file__],verbosity=2)
    elif args.command=='compact':
        if any(x is None for x in [args.original,args.independent,args.legacy,args.clean_source]):ap.error('compact requires all source directories and --clean-source')
        compact(args.original,args.independent,args.legacy,args.clean_source,args.output)
    else:print(json.dumps(verify(args.output),indent=2))
