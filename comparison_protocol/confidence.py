"""Validation-only ordinal correctness mapping, never a native class posterior."""
from dataclasses import dataclass
from .protocol import require_partition
from .contracts import digest

LEVELS=('low','medium','high')
CONFIDENCE_ANALYSIS_SPEC=dict(
    version='confidence_analysis_v2',primary='Original ordinal reliability: n/correct/error by level, high-confidence errors and risk/coverage.',
    primary_is_native_probability_calibration=False,
    secondary='Validation-only Beta(1,1) p(correct|level) wrapper; fixed three-level ECE and binary correctness Brier on test.',
    smoothing='(correct+1)/(n+2); an empty level uses the same smoothed overall validation prior.',
    missing='Exclude missing levels from conditional calibration only; publish total/valid/missing n and coverage. Never impute a probability.',
    monotonicity='Report observed level reversals; do not reorder or fit monotonic constraints.',
    scope='No native class-posterior Brier/ECE claim; selection and calibration reuse validation and this dependence is disclosed.')

def fit_correctness_mapping(rows,*,dataset,selection_seal):
    if not selection_seal:raise ValueError('Seal all primary selections before calibration fitting')
    require_partition(dataset,[r['task_id'] for r in rows],{'validation'},purpose='confidence mapping fit')
    if any(r.get('split','validation')!='validation' for r in rows):raise ValueError('Calibration requires validation, never test')
    if any(r['confidence'] not in LEVELS or r['label_correct'] not in (0,1) for r in rows):raise ValueError('Invalid calibration level/correctness')
    total=sum(r['label_correct'] for r in rows);fallback=(total+1)/(len(rows)+2);mapping={};counts={}
    for level in LEVELS:
        group=[r for r in rows if r['confidence']==level];counts[level]=len(group)
        mapping[level]=(sum(r['label_correct'] for r in group)+1)/(len(group)+2) if group else fallback
    result=dict(version='ordinal_correctness_wrapper_v1',mapping=mapping,n=counts,
        calibration_task_ids=[r['task_id'] for r in rows],selection_seal_sha256=selection_seal.sha256,
        dataset=dataset,fit_split='validation',native_class_probability=False)
    result['fit_sha256']=digest(result)
    return result

def score_correctness_mapping(rows,wrapper):
    if wrapper.get('fit_sha256')!=digest({k:v for k,v in wrapper.items() if k!='fit_sha256'}):
        raise ValueError('Confidence mapping changed after fitting')
    require_partition(wrapper['dataset'],list(dict.fromkeys(r['task_id'] for r in rows)),{'test'},purpose='fixed confidence evaluation')
    if not rows:raise ValueError('No evaluation rows')
    groups={k:[r for r in rows if r.get('confidence')==k] for k in LEVELS}
    valid=[r for r in rows if r.get('confidence') in LEVELS]
    if any(r['label_correct'] not in (0,1) for r in rows):raise ValueError('Invalid correctness')
    brier=sum((wrapper['mapping'][r['confidence']]-r['label_correct'])**2 for r in valid)/len(valid) if valid else None
    ece=sum(len(v)*abs(wrapper['mapping'][k]-sum(r['label_correct'] for r in v)/len(v)) for k,v in groups.items() if v)/len(valid) if valid else None
    return dict(binary_correctness_Brier=brier,three_level_ECE=ece,n=len(valid),n_total=len(rows),n_missing=len(rows)-len(valid),
        confidence_coverage=len(valid)/len(rows),evaluation_target='posthoc_binary_correctness_wrapper',
        by_level={k:dict(n=len(v),accuracy=sum(r['label_correct'] for r in v)/len(v) if v else None) for k,v in groups.items()},
        native_class_probability=False,analysis_spec_version=CONFIDENCE_ANALYSIS_SPEC['version'])
