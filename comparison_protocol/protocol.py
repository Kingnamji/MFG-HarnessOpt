"""Manifest-backed phase boundaries. Test is evaluation-only after sealing.

These are workflow guards against accidental leakage, not a sandbox against
arbitrary Python code. No model, credential, training, or API work on import.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import json
import re
from .contracts import digest, verify_frozen_reference, Candidate, LOCK,compile_candidate,Arm

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'AI4I':('AI4I','tasks_ai4i.csv'), 'PdM':('PdM','tasks_azure.csv')}

@lru_cache(maxsize=2)
def registry(dataset):
    if dataset not in DATASETS: raise ValueError('Unknown registered dataset')
    folder,name=DATASETS[dataset]
    with (ROOT/'Dataset/Preprocessed'/folder/name).open(encoding='utf-8-sig',newline='') as f:
        # Labels are intentionally not copied into this access-control index.
        return {r['task_id']:{k:r.get(k) for k in ('split','machineID','udi','failure_time')}
                for r in csv.DictReader(f)}

def require_partition(dataset, task_ids, allowed, *, purpose):
    ids=list(map(str,task_ids)); index=registry(dataset)
    if not ids or len(ids)!=len(set(ids)): raise ValueError(purpose+': unique nonempty task IDs required')
    if any(i not in index or index[i]['split'] not in set(allowed) for i in ids):
        raise ValueError(purpose+': task manifest forbids this partition (test cannot train/optimize/select)')
    return tuple(ids)

def assert_training_views(views, *, purpose='training'):
    if not views: raise ValueError('No training views')
    dataset='AI4I' if 'TWF' in views[0].labels else 'PdM'
    require_partition(dataset,[v.task['task_id'] for v in views],{'search'},purpose=purpose)
    if any(v.task.get('split')!='search' for v in views): raise ValueError('Training view split mismatch')

def search_origin(dataset, task_ids):
    ids=require_partition(dataset,task_ids,{'search'},purpose='search feedback')
    return dict(dataset=dataset,scope='search',task_ids=list(ids),manifest_sha256=digest(registry(dataset)))

def validate_search_origin(origin, dataset):
    if not isinstance(origin,dict) or origin.get('dataset')!=dataset or origin.get('scope')!='search':
        raise ValueError('Search-only feedback provenance required; validation/test are prohibited')
    require_partition(dataset,origin.get('task_ids',[]),{'search'},purpose='Proposer feedback')
    if origin.get('manifest_sha256')!=digest(registry(dataset)): raise ValueError('Stale feedback task manifest')

def assert_search_tree(value,dataset):
    """Reject explicit wrong splits and actual held-out task IDs before dispatch."""
    index=registry(dataset)
    def visit(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in {'split','comparison_split','scope'} and x in ('validation','test'):
                    raise ValueError('Held-out feedback cannot enter a proposal')
                visit(x)
        elif isinstance(v,(tuple,list)):
            for x in v: visit(x)
        elif isinstance(v,str):
            for ident in [v]+re.findall(r'\b(?:AI4I|AZURE)_\d+\b',v):
                if ident in index and index[ident]['split']!='search':
                    raise ValueError('Held-out task ID in proposal context')
    visit(value)

def require_evaluator_phase(evaluator, split):
    if getattr(evaluator,'split',None)!=split:
        raise ValueError('A manifest-bound '+split+' evaluator is required before any proposal/selection call')

@dataclass(frozen=True)
class EvaluationSeal:
    payload_json:str

    @property
    def payload(self): return json.loads(self.payload_json)

    @property
    def sha256(self): return digest(self.payload)

    def authorize(self,*,dataset,candidate_id,signature,task_ids):
        p=self.payload
        if p.get('protocol_hash')!=verify_frozen_reference(): raise ValueError('Protocol changed after selection seal')
        require_partition(dataset,task_ids,{'test'},purpose='sealed final evaluation')
        key=dataset+':'+candidate_id
        if p['systems'].get(key)!=signature: raise ValueError('Unselected or mutated system cannot access test')

def seal_evaluation_systems(*,systems,selection_records,analysis_spec):
    """Only selection records from validation and fixed a-priori controls.

    Caller seals the COMPLETE campaign before test, not one newly tuned patch
    after looking at test. A response/score is never accepted as a fixed control.
    """
    if not analysis_spec or not systems: raise ValueError('Analysis specification and systems required')
    selected={}
    for r in selection_records:
        dataset=r['dataset']; origin=r['origin']
        if origin.get('scope')!='validation': raise ValueError('Only validation selects a candidate')
        require_partition(dataset,origin['task_ids'],{'validation'},purpose='selection seal')
        selected[dataset+':'+r['selected_candidate_id']]=r['selected_signature']
    entries={}
    for s in systems:
        key=s['dataset']+':'+s['candidate_id']
        if key in entries: raise ValueError('Duplicate system in seal')
        if not s.get('a_priori_fixed') and selected.get(key)!=s['signature']:
            raise ValueError('Missing validation selection record')
        fixed={'M0','M1','M2','M3','M4_reference','RULE','PRIOR','DT','GBT'}|{'M2_'+c for c in ('S0E0G0','S0E0G1','S1E0G0','S1E0G1','S1E1G0','S1E1G1')}
        references=json.loads(LOCK.read_text(encoding='utf-8'))['selected_patches']
        ref=references.get(s['dataset'],{})
        if s['candidate_id']==ref.get('candidate_id'):
            candidate=compile_candidate(Arm.TRACE,ref['candidate_id'],json.loads(Path(ref['original_path']).read_text(encoding='utf-8')))
            if s['signature']!=candidate.signature:raise ValueError('Frozen reference patch signature mismatch')
            fixed.add(s['candidate_id'])
        if s.get('a_priori_fixed') and s['candidate_id'] not in fixed:
            raise ValueError('An arbitrary optimized patch cannot be relabelled a fixed control')
        entries[key]=s['signature']
    p=dict(version='evaluation_only_seal_v1',protocol_hash=verify_frozen_reference(),systems=entries,
           selection_records=selection_records,analysis_sha256=digest(analysis_spec),test_feedback_allowed=False)
    return EvaluationSeal(json.dumps(p,sort_keys=True,ensure_ascii=False,allow_nan=False))
