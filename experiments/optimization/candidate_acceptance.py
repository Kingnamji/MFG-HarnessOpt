"""Paired offline selection guards; never part of runtime report repair.

The Paper J formula/weights are preserved. Diagnostic non-regression is a
separate, explicit revised selection constraint against the actual M3 reference.
"""
import json
import math
import pandas as pd
from sklearn.metrics import f1_score
from experiments.llm.report_contract import REPORT_CONTRACT_VERSION
from experiments.optimization.selection_metrics import compute_j_mfg_for_group

DATASETS={'AI4I':'ai4i','AI4I 2020':'ai4i','PdM':'pdm','Azure PdM':'pdm','Azure':'pdm'}


def _population(frame,configs,scope):
    if scope not in {'search','validation'}:
        raise ValueError('Candidate selection is restricted to search/validation, never test')
    if frame.empty:raise ValueError('Missing paired M3/candidate task population')
    required={'task_id','dataset','gold_label','predicted_label','label_correct','report_contract_version'}
    if not required.issubset(frame):raise ValueError('Missing candidate/reference evaluation fields')
    correct=frame.gold_label.astype(str).eq(frame.predicted_label.astype(str)).astype(int)
    if not correct.eq(frame.label_correct).all():raise ValueError('Cached label_correct disagrees with the report labels')
    for name in ['cost_per_report','unsupported_claim_rate','unsafe_recommendation_rate','generic_action_rate','F10']:
        if not pd.to_numeric(frame[name],errors='coerce').map(math.isfinite).all():
            raise ValueError(f'Nonfinite selection metric: {name}')
    if 'report_contract_version' not in frame or not frame.report_contract_version.eq(REPORT_CONTRACT_VERSION).all():
        raise ValueError('Selection requires newly generated reports under the current report contract')
    for dataset,group in frame.groupby('dataset'):
        if dataset not in DATASETS:raise ValueError(f'Unknown dataset: {dataset}')
        manifest=pd.read_csv(configs.paths['dataset'][DATASETS[dataset]]['tasks'],usecols=['task_id','split','gold_label'])
        ids=set(manifest.loc[manifest.split.eq(scope),'task_id'].astype(str))
        if not set(group.task_id.astype(str)).issubset(ids):raise ValueError('Selection task belongs to a different split')
        if group.task_id.astype(str).duplicated().any():raise ValueError('Duplicate task in candidate population')
        expected=manifest.set_index('task_id').gold_label.astype(str).to_dict()
        if not all(str(row.gold_label)==expected[str(row.task_id)] for row in group.itertuples()):
            raise ValueError('Evaluation targets disagree with the frozen task manifest')


def diagnostic_summary(frame,configs):
    rows=[]
    for dataset,group in frame.groupby('dataset'):
        schema=json.loads(configs.paths['dataset'][DATASETS[dataset]]['allowed_schema'].read_text(encoding='utf-8'))
        labels=schema['candidate_labels']
        values=dict(dataset=dataset,n_tasks=len(group),label_accuracy=float(group.label_correct.mean()),
            macro_f1=float(f1_score(group.gold_label.astype(str),group.predicted_label.astype(str),
                                   labels=labels,average='macro',zero_division=0)),
            J_mfg=compute_j_mfg_for_group(group,configs))
        for name in ['evidence_precision','evidence_recall','evidence_linked_action_ratio',
                     'unsupported_claim_rate','unsafe_recommendation_rate','cost_per_report','F10']:
            v=float(group[name].mean());values[name]=v if math.isfinite(v) else None
        rows.append(values)
    return rows


def assess_candidate(candidate,baseline,configs,*,scope):
    _population(candidate,configs,scope);_population(baseline,configs,scope)
    reasons=[]
    if set(candidate.dataset)!=set(baseline.dataset):
        return dict(eligible=False,reasons=['dataset_population_mismatch'],scope=scope,comparisons=[])
    policy=configs.experiment.get('candidate_acceptance',{})
    comparisons=[]
    for dataset,c in candidate.groupby('dataset'):
        b=baseline.loc[baseline.dataset.eq(dataset)]
        # Require the same task/target manifest, not merely equal sample counts.
        ct=c.set_index(c.task_id.astype(str));bt=b.set_index(b.task_id.astype(str))
        if set(ct.index)!=set(bt.index):
            reasons.append(dataset+':task_population_mismatch');continue
        if not ct.loc[sorted(ct.index),'gold_label'].equals(bt.loc[sorted(bt.index),'gold_label']):
            reasons.append(dataset+':target_manifest_mismatch');continue
        cs,bs=diagnostic_summary(c,configs)[0],diagnostic_summary(b,configs)[0]
        comparisons.append(dict(dataset=dataset,candidate=cs,M3_reference=bs))
        for metric,setting in [('label_accuracy','max_accuracy_drop'),('macro_f1','max_macro_f1_drop')]:
            tolerance=float(policy.get(setting,0))
            if tolerance<0 or not math.isfinite(tolerance):raise ValueError('Invalid acceptance tolerance')
            if not math.isfinite(cs[metric]) or cs[metric]+tolerance+1e-12<bs[metric]:
                reasons.append(dataset+':'+metric+'_regression')
        if not math.isfinite(cs['J_mfg']) or not math.isfinite(bs['J_mfg']):
            reasons.append(dataset+':non_finite_objective')
        elif policy.get('require_J_not_worse_than_M3',True) and cs['J_mfg']+1e-12<bs['J_mfg']:
            reasons.append(dataset+':J_mfg_regression')
    return dict(eligible=not reasons,reasons=reasons,scope=scope,comparisons=comparisons)


def guard_candidate_summary(summary,metrics,baseline,configs,*,scope):
    out=summary.copy();decisions=[]
    if baseline.empty:raise ValueError('M3 reference reports are required before candidate selection')
    for _,row in out.iterrows():
        candidate=metrics.loc[metrics.candidate_id.astype(str).eq(str(row['candidate_id']))]
        decisions.append(assess_candidate(candidate,baseline,configs,scope=scope))
    # Never trust cached summary scores over the paired task-level results.
    out['J_mfg']=[compute_j_mfg_for_group(metrics.loc[metrics.candidate_id.astype(str).eq(str(row['candidate_id']))],configs)
                  for _,row in out.iterrows()]
    out['diagnostic_eligible']=[d['eligible'] for d in decisions]
    out['rejection_reasons']=[json.dumps(d['reasons']) for d in decisions]
    out['acceptance_scope']=scope
    out['acceptance_details']=[json.dumps(d,ensure_ascii=False) for d in decisions]
    return out


def project_search_feedback(baseline,history,configs,*,incumbent_id='M3'):
    """Aggregate search feedback only; no target labels or target evidence rows."""
    _population(baseline,configs,'search')
    rows=[];paired=[]
    incumbent=baseline if incumbent_id=='M3' else history.loc[history.candidate_id.astype(str).eq(incumbent_id)]
    _population(incumbent,configs,'search')
    if not history.empty:
        for cid,group in history.groupby('candidate_id'):
            decision=assess_candidate(group,baseline,configs,scope='search')
            rows.append(dict(candidate_id=str(cid),eligible=decision['eligible'],
                             rejection_reasons=decision['reasons'],metrics=diagnostic_summary(group,configs)))
            for dataset,c in group.groupby('dataset'):
                b=incumbent.loc[incumbent.dataset.eq(dataset)]
                pairs=c[['task_id','label_correct']].merge(b[['task_id','label_correct']],on='task_id',suffixes=('_candidate','_incumbent'),validate='one_to_one')
                if len(pairs)!=len(c) or len(c)!=len(b):raise ValueError('Incumbent comparison population mismatch')
                first,last=pairs.label_correct_incumbent,pairs.label_correct_candidate
                paired.append(dict(candidate_id=str(cid),dataset=dataset,n_tasks=len(c),
                    regressions_vs_incumbent=int(((first==1)&(last==0)).sum()),
                    fixes_vs_incumbent=int(((first==0)&(last==1)).sum()),
                    unchanged_correct=int(((first==1)&(last==1)).sum()),
                    unchanged_wrong=int(((first==0)&(last==0)).sum())))
    return dict(scope='search',report_contract_version=REPORT_CONTRACT_VERSION,
                incumbent_candidate_id=incumbent_id,paired_candidate_summary=paired,
                M3_reference=diagnostic_summary(baseline,configs),candidates=rows)
