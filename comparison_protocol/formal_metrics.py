"""Offline metric/report exporters. Never called by generation or selection.

The frozen evaluator and canonical group J remain the primary scores. Missing
measurements are explicit, not silently turned into perfect rates or zero cost.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json, math
import numpy as np
import pandas as pd
from experiments.evaluation.observable_matching import values_equal
from experiments.evaluation.metric_policy import selection_grounding_scores
from experiments.optimization.selection_metrics import compute_j_mfg_for_group,load_metric_weights

def rate(n,d):return n/d if d else None
def mean(values):
    v=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(value=float(np.mean(v)) if v else None,n_defined=len(v),n_missing=len(values)-len(v))
def quantiles(values):
    v=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(n=len(v),mean=float(np.mean(v)) if v else None,p50=float(np.percentile(v,50)) if v else None,p95=float(np.percentile(v,95)) if v else None)

def source_packet_audit(observations):
    rows=observations.to_dict('records') if hasattr(observations,'to_dict') else list(observations)
    text=json.dumps(rows,ensure_ascii=False,default=str,sort_keys=True)
    return dict(scope='generation_visible_packet_only',record_count=len(rows),utf8_bytes=len(text.encode()),
        packet_sha256=sha256(text.encode()).hexdigest(),source_counts=dict(Counter(r.get('source') for r in rows)),
        windows=dict(Counter(str(r.get('window')) for r in rows)),event_times=[r.get('event_time') for r in rows if r.get('event_time')],
        tool_calls_are_in_execution_log=True,evaluation_target_reads_included=False)

def supplemental_report_audit(report,observations,schema,association_guide=None):
    rows=observations.to_dict('records') if hasattr(observations,'to_dict') else list(observations)
    evidence=report.get('evidence') or [];actions=report.get('next_steps') or []
    available={str(r.get('evidence_id')):r for r in rows}
    claimed={str(r.get('evidence_id')):r for r in evidence if isinstance(r,dict)}
    def preserved(ids):return sum(i in claimed and values_equal(claimed[i].get('value'),available[i].get('value')) for i in ids)
    normal=[i for i,r in available.items() if r.get('direction') in ('normal','within_range','unchanged') or r.get('role') in ('exclusion','normal')]
    contextual=[i for i,r in available.items() if r.get('role')=='contextual']
    from experiments.evaluation.public_conditions import check_public_conditions
    conditions=check_public_conditions(report,rows,schema)
    exclusions=[c for c in conditions['checks'] if c.get('satisfied') is False]
    from nonllm_baselines.views import ObservableView,dump,visible_associations
    from nonllm_baselines.predictors import matched_associations
    view=ObservableView(dump(dict(records=rows,associations=visible_associations(association_guide))))
    matches=matched_associations(view);tops={r['top_component'] for r in matches}
    cited_rows=[r for r in rows if str(r.get('evidence_id')) in claimed]
    cited_view=ObservableView(dump(dict(records=cited_rows,associations=visible_associations(association_guide))))
    cited_patterns={r['evidence_key'] for r in matched_associations(cited_view)}
    contrary=[r for r in matches if r['top_component']!=report.get('predicted_label')]
    refs=[a.get('rationale_evidence_ids') or [] for a in actions]
    return dict(normal_exclusion_available=len(normal),normal_exclusion_preserved=preserved(normal),
        normal_exclusion_preservation=rate(preserved(normal),len(normal)),
        contextual_available=len(contextual),contextual_preserved=preserved(contextual),
        public_exclusion_opportunities=len(exclusions),public_exclusion_facts_preserved=sum(bool(c['evidence_ids']) and all(str(i) in claimed for i in c['evidence_ids']) for c in exclusions),
        prior_conflict=len(tops)>1,visible_prior_patterns=[r['evidence_key'] for r in matches],
        contrary_prior_patterns=len(contrary),contrary_prior_facts_preserved=sum(r['evidence_key'] in cited_patterns for r in contrary),
        public_condition_checks=conditions,action_reference_counts=list(map(len,refs)),
        action_duplicate_reference_count=sum(len(r)-len(set(r)) for r in refs),
        single_reference_actions=sum(len(r)==1 for r in refs),multi_reference_actions=sum(len(r)>1 for r in refs),
        field_named_actions=sum(any(str(r.get('field','')) and str(r['field']) in json.dumps(a,ensure_ascii=False) for r in rows) for a in actions),
        action_targets=[a.get('target') for a in actions],n_limitations=len(report.get('limitations') or []))

def classification(rows,labels):
    """Fixed label denominator includes predictions outside the label set."""
    truth=[str(r['gold_label']) for r in rows];pred=[str(r.get('predicted_label','')) for r in rows]
    by={}
    for label in labels:
        tp=sum(t==label and p==label for t,p in zip(truth,pred));fp=sum(t!=label and p==label for t,p in zip(truth,pred));fn=sum(t==label and p!=label for t,p in zip(truth,pred))
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        by[label]=dict(TP=tp,FP=fp,FN=fn,n=sum(t==label for t in truth),precision=precision,recall=recall,F1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    return dict(n=len(rows),accuracy=rate(sum(t==p for t,p in zip(truth,pred)),len(rows)),
        macro_f1=sum(r['F1'] for r in by.values())/len(labels),by_class=by,
        confusion={t:{p:sum(x==t and (y==p if p!='__invalid__' else y not in labels) for x,y in zip(truth,pred)) for p in list(labels)+['__invalid__']} for t in labels},
        invalid_predictions=sum(p not in labels for p in pred))

def canonical_components(rows,configs):
    f=pd.DataFrame(rows);e,a=selection_grounding_scores(f)
    terms=dict(alpha_label=float(f.label_correct.mean()),beta_evidence=e,gamma_action=a,
        lambda_unsupported=-float(f.unsupported_claim_rate.mean()),lambda_unsafe=-float(f.unsafe_recommendation_rate.mean()),
        mu_cost=-min(1.,max(0.,float(f.cost_per_report.mean())/.005)),rho_repair_failure=-float(f.F10.mean()))
    return dict(J_mfg=compute_j_mfg_for_group(f,configs),components=terms,weights=load_metric_weights(configs))

def usage_summary(rows):
    fields=('input_tokens','cached_input_tokens','output_tokens','reasoning_tokens','cost_usd_actual','cost_usd_uncached')
    return dict(calls=len(rows),**{k:dict(total=sum(float(r[k]) for r in rows) if all(r.get(k) is not None for r in rows) else None,
        known_subtotal=sum(float(r[k]) for r in rows if r.get(k) is not None),
        n_known=sum(r.get(k) is not None for r in rows),n_missing=sum(r.get(k) is None for r in rows)) for k in fields},
        status=dict(Counter(r.get('status') for r in rows)),by_phase=dict(Counter(r.get('phase') for r in rows)))

def ordinal_reliability(rows):
    from .confidence import LEVELS
    from scipy.stats import binomtest
    levels={}
    for level in LEVELS:
        group=[r for r in rows if r.get('confidence')==level];correct=sum(r['label_correct'] for r in group)
        ci=binomtest(correct,len(group)).proportion_ci() if group else None
        levels[level]=dict(n=len(group),accuracy=rate(correct,len(group)),CI95=[ci.low,ci.high] if ci else None)
    curve=[]
    for n in (1,2,3):
        accepted=[r for r in rows if r.get('confidence') in LEVELS[-n:]]
        curve.append(dict(minimum_level=LEVELS[-n],n=len(accepted),coverage=rate(len(accepted),len(rows)),risk=rate(sum(1-r['label_correct'] for r in accepted),len(accepted))))
    reversals=[dict(lower=a,higher=b) for a,b in zip(LEVELS,LEVELS[1:]) if levels[a]['n'] and levels[b]['n'] and levels[a]['accuracy']>levels[b]['accuracy']]
    all_reversals=[dict(lower=a,higher=b) for i,a in enumerate(LEVELS) for b in LEVELS[i+1:]
        if levels[a]['n'] and levels[b]['n'] and levels[a]['accuracy']>levels[b]['accuracy']]
    high=[r for r in rows if r.get('confidence')=='high']
    return dict(by_level=levels,risk_coverage=curve,missing=sum(r.get('confidence') not in LEVELS for r in rows),
        n_total=len(rows),n_valid=sum(v['n'] for v in levels.values()),
        high_confidence_errors=dict(count=sum(1-r['label_correct'] for r in high),n=len(high),rate=rate(sum(1-r['label_correct'] for r in high),len(high))),
        observed_adjacent_level_reversals=reversals,native_class_probability=False,
        observed_level_reversals=all_reversals,
        CI_note='Binomial descriptive intervals; repeated draws are not independent task inference. Use paired cluster analysis for comparisons.')

def aggregate_measurements(records,*,labels,configs,expected_task_ids,wrapper=None):
    """One system/run/draw allocation. No successful-only task filtering."""
    from .confidence import score_correctness_mapping
    ids=[r['task_id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_task_ids):raise ValueError('Incomplete/duplicate task allocation; do not publish successful-only scores')
    metrics={f'M{i:02}':None for i in range(1,27)};stages={}
    for stage in ('initial','post_repair','pre_filter','final'):
        selected=[r['common_measurement']['stages'][stage] for r in records if stage in r['common_measurement']['stages']]
        if not selected:continue
        primary=[r['primary'] for r in selected];supp=[r['supplemental'] for r in selected]
        stage_rows=[dict(r,confidence=m.get('confidence')) for r,m in zip(primary,selected)]
        stages[stage]=dict(n=len(primary),classification=classification(primary,labels),objective=canonical_components(primary,configs),
            means={k:mean([r.get(k) for r in primary]) for k in ('evidence_precision','evidence_recall','unsupported_claim_rate','unsafe_recommendation_rate','evidence_linked_action_ratio','generic_action_rate','schema_valid')},
            failures={f'F{i}':dict(count=sum(r[f'F{i}'] for r in primary),n=len(primary)) for i in range(1,11)},
            supplemental={k:mean([r.get(k) for r in supp]) for k in ('evidence_coverage','action_coverage','factual_precision','metadata_copy_precision','all_refs_complete_ratio','normal_exclusion_preservation','n_actions','n_evidence','duplicate_actions')},
            confidence=ordinal_reliability(stage_rows))
    final=stages['final'];rows=[dict(r['common_measurement']['stages']['final']['primary'],confidence=r['common_measurement']['stages']['final'].get('confidence')) for r in records]
    measured=[r['common_measurement'] for r in records];trans=[r['transitions'] for r in measured]
    metrics.update(M01=final['classification'],M02=final['objective'],M03=final['classification']['by_class'],
        M04={k:final['means'][k] for k in ('evidence_precision','evidence_recall')},M05=final['means']['unsupported_claim_rate'],
        M06=final['means']['unsafe_recommendation_rate'],M07=final['means']['evidence_linked_action_ratio'],M08=final['means']['generic_action_rate'],
        M09=final['failures'],M10=dict(expected=len(expected_task_ids),complete=len(records),schema_valid=final['means']['schema_valid']),
        M11={**{k:final['supplemental'][k] for k in ('evidence_coverage','action_coverage')},
             'linked_action_coverage':mean([r.get('evidence_linked_action_present') for r in rows])},
        M12={k:final['supplemental'][k] for k in ('factual_precision','metadata_copy_precision')},M13=final['supplemental']['all_refs_complete_ratio'],
        M14=dict(normal_exclusion=final['supplemental']['normal_exclusion_preservation'],
            public_and_prior_details=[r['common_measurement']['stages']['final']['supplemental'] for r in records]),
        M15=dict(counts={k:final['supplemental'][k] for k in ('n_actions','n_evidence','duplicate_actions')},
            reference_details=[r['common_measurement']['stages']['final']['supplemental'].get('action_reference_counts') for r in records]),
        M16=final['confidence'],M17=score_correctness_mapping(rows,wrapper) if wrapper else dict(status='missing_validation_mapping',n_total=len(rows)),
        M18=usage_summary([u for r in measured for u in (r.get('usage_rows') or [])]),
        M19=dict(end_to_end=quantiles([r.get('end_to_end_latency_sec') for r in measured]),
            API=quantiles([u.get('latency_sec') for r in measured for u in (r.get('usage_rows') or [])]),
            timing_events=[r.get('timing',[]) for r in records],native_intervals=[r.get('native_timing',[]) for r in records]),
        M20=[r.get('source_packet') for r in measured],M21=dict(attempts=sum(r['repair_count'] for r in trans),transitions=trans),
        M22=dict(transitions=trans),M23=dict(status='optimizer_only_see_optimizer_summary'),M24=dict(status='optimizer_only_see_optimizer_summary'),
        M25=dict(status='requires_phase_ledger_and_deployment_means'),M26=dict(status='nonLLM_only_see_training_provenance'))
    return dict(metric_families=metrics,stages=stages,expected_task_ids=list(expected_task_ids),source='formal_archives_only')

def budget_by_phase(ledger):
    groups={}
    for r in ledger:
        key=':'.join(str(r.get(k,'unassigned')) for k in ('dataset','arm','run_id','campaign_phase','phase'))
        groups.setdefault(key,[]).append(r)
    return {k:usage_summary(v) for k,v in groups.items()}

def break_even(setup_cost,baseline_report_cost,optimized_report_cost):
    saving=baseline_report_cost-optimized_report_cost
    return dict(setup_cost=setup_cost,per_report_saving=saving,reports=math.ceil(setup_cost/saving) if saving>0 else None,
        reason=None if saving>0 else 'No finite break-even with nonpositive marginal savings')

def optimizer_summary(search_result,ledger):
    iterations=[];best=None
    for iteration in range(1,5):
        attempts=[a for a in search_result['attempts'] if a['iteration']==iteration]
        ids={a['candidate_id'] for a in attempts};points=[p for p in search_result['points'] if p['candidate_id'] in ids]
        scores=[p['metrics']['J_mfg'] for p in points]
        eligible=[p for p in search_result['points'] if p['eligible'] and (p['candidate_id']=='M3' or p['candidate_id'] in {a['candidate_id'] for a in search_result['attempts'] if a['iteration']<=iteration})]
        if eligible:best=sorted(eligible,key=lambda p:(-p['metrics']['J_mfg'],p['candidate_id']))[0]['candidate_id']
        uses=[u for u in ledger if u.get('iteration',0)<=iteration]
        iterations.append(dict(iteration=iteration,status_counts=dict(Counter(a['status'] for a in attempts)),n_candidates=len(points),
            J=quantiles(scores),incumbent=best,cumulative_usage=usage_summary(uses)))
    return dict(M23=search_result['attempts'],M24=iterations,all_points=search_result['points'])

def holm(pvalues):
    order=sorted(pvalues,key=pvalues.get);out={};last=0
    for i,k in enumerate(order):last=max(last,min(1.,pvalues[k]*(len(order)-i)));out[k]=last
    return out

def paired_bootstrap(left,right,*,score,clusters,n_resamples=10000,seed=20261024):
    """Resample paired unique task/cluster units; preserve all run/draw rows."""
    a=pd.DataFrame(left);b=pd.DataFrame(right)
    ids=set(a.task_id)
    if ids!=set(b.task_id) or set(clusters)!=ids:raise ValueError('Paired task/cluster populations differ')
    groups=sorted(set(clusters.values()),key=str);rng=np.random.default_rng(seed);values=[]
    for _ in range(n_resamples):
        sample=rng.choice(len(groups),len(groups),replace=True);aa=[];bb=[]
        for index in sample:
            tasks={i for i,g in clusters.items() if g==groups[index]}
            aa.append(a[a.task_id.isin(tasks)]);bb.append(b[b.task_id.isin(tasks)])
        values.append(score(pd.concat(aa,ignore_index=True))-score(pd.concat(bb,ignore_index=True)))
    delta=score(a)-score(b);lo,hi=np.percentile(values,[2.5,97.5])
    # Centered paired bootstrap null approximation, rather than calling a
    # percentile sign frequency an exact randomization p-value.
    p=(1+sum(abs(v-delta)>=abs(delta) for v in values))/(n_resamples+1)
    return dict(effect=delta,CI95=[float(lo),float(hi)],p_bootstrap_centered=p,n_clusters=len(groups),
        n_unique_tasks=len(ids),resamples=n_resamples,seed=seed)

def sensitivity_grid():
    names=('alpha_label','beta_evidence','gamma_action','lambda_unsupported','lambda_unsafe','mu_cost','rho_repair_failure')
    weights=[dict.fromkeys(names,1.)]
    for name in names:
        for v in (.5,2.):weights.append({**dict.fromkeys(names,1.),name:v})
    return [dict(weights=w,Cref=ref,clipped=clipped) for w in weights for ref in (.0025,.005,.01) for clipped in (True,False)]

def rescore_pool(pool,*,baseline_id,configs,settings=None):
    """Search/validation only, unchanged non-J guards, no new optimization."""
    from experiments.optimization.candidate_acceptance import assess_candidate
    if any(set(f.comparison_split)-{'search','validation'} for f in pool.values()):raise ValueError('Sensitivity cannot read test')
    results=[];base=pool[baseline_id]
    noJ=deepcopy(configs);noJ.experiment['candidate_acceptance']['require_J_not_worse_than_M3']=False
    for setting in sensitivity_grid() if settings is None else settings:
        scores={};eligibility={}
        for cid,f in pool.items():
            terms=canonical_components(f.to_dict('records'),configs)['components'];terms['mu_cost']=-float(f.cost_per_report.mean())/setting['Cref']
            if setting['clipped']:terms['mu_cost']=max(-1.,min(0.,terms['mu_cost']))
            scores[cid]=sum(setting['weights'][k]*v for k,v in terms.items())
            decision=assess_candidate(f,base,noJ,scope=str(f.comparison_split.iloc[0]));eligibility[cid]=decision['eligible']
        eligible=[cid for cid in scores if eligibility[cid] and scores[cid]+1e-12>=scores[baseline_id]]
        rank=sorted(eligible,key=lambda cid:(-scores[cid],float(pool[cid].unsafe_recommendation_rate.mean()),float(pool[cid].unsupported_claim_rate.mean()),float(pool[cid].cost_per_report.mean()),cid))
        results.append(dict(setting=setting,scores=scores,nonJ_eligible=eligibility,winner=rank[0],ranking=rank))
    return results

def sensitivity_analysis(search_pool,validation_pool,*,configs):
    from scipy.stats import spearmanr
    settings=sensitivity_grid();base=settings[2]  # base weights, Cref .005, clipped
    leave=[]
    for name in base['weights']:
        leave.append(dict(weights={**base['weights'],name:0.},Cref=.005,clipped=True))
    search=rescore_pool(search_pool,baseline_id='M3',configs=configs,settings=settings+leave)
    val=rescore_pool(validation_pool,baseline_id='M3',configs=configs,settings=settings+leave)
    rows=[]
    for a,b in zip(search,val):
        short=[c for c in a['ranking'] if c!='M3'][:4]
        ranks=[c for c in b['ranking'] if c in set(short)|{'M3'}]
        rows.append(dict(setting=a['setting'],shortlist=short,winner=ranks[0],search=a,validation=b))
    original=rows[2]['winner'];base_order=rows[2]['validation']['scores'];ids=sorted(base_order)
    for row in rows:
        x=[base_order[i] for i in ids];y=[row['validation']['scores'][i] for i in ids]
        row['rank_correlation']=float(spearmanr(x,y).statistic) if len(ids)>1 and len(set(x))>1 and len(set(y))>1 else None
    means={c:{k:float(f[k].mean()) for k in ('label_correct','evidence_precision','unsupported_claim_rate','cost_per_report')} for c,f in validation_pool.items()}
    pareto=[]
    for cid,m in means.items():
        if not any((n['label_correct']>=m['label_correct'] and n['cost_per_report']<=m['cost_per_report'] and
                    (n['label_correct']>m['label_correct'] or n['cost_per_report']<m['cost_per_report'])) for other,n in means.items() if other!=cid):pareto.append(cid)
    f=pd.DataFrame(means).T
    correlation=f[['evidence_precision','unsupported_claim_rate']].corr().iloc[0,1]
    return dict(grid90=rows[:90],leave_one_term_out7=rows[90:],primary_winner_unchanged=True,
        diagnostic_base_winner=original,winner_change_rate=sum(r['winner']!=original for r in rows[:90])/90,
        cost_saturation_rate=sum(m['cost_per_report']>=.005 for m in means.values())/len(means),
        evidence_unsupported_correlation=float(correlation) if math.isfinite(correlation) else None,pareto_accuracy_cost=pareto)

def choose_cases(cases,*,limit=6):
    """Categories are assigned by deterministic comparisons before selection."""
    if limit not in (4,5,6):raise ValueError('Four to six cases required')
    groups={};chosen=[]
    for case in cases:
        if str(case['run_id'])!='1701' or case['draw_id']!=1:continue
        groups.setdefault(case['category'],[]).append(case)
    categories=('improvement','same_label','regression','common_failure','repair','uncertainty')
    for category in categories:
        available=groups.get(category,[])
        if available:chosen.append(min(available,key=lambda c:sha256((c['dataset']+c['task_id']).encode()).hexdigest()))
    return dict(cases=chosen[:limit],denominators={k:len(groups.get(k,[])) for k in categories},
        absent=[k for k in categories if not groups.get(k)],pilot=False,selection='fixed category/hash, never best performance')
