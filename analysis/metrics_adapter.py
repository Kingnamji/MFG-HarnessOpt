"""Preserve frozen metrics; make certified paid missing outcomes explicit."""
from copy import deepcopy
import inspect,math
import numpy as np
from comparison_protocol import formal_metrics as fm,formal_export as fe,analysis_details as ad
REQUIRED_J=('label_correct','evidence_precision','evidence_linked_action_ratio','unsupported_claim_rate','unsafe_recommendation_rate','cost_per_report','F10')
def missing_report(r):return r.get('report') is None or r.get('metrics',{}).get('report_status')=='terminal_parse_failure'
def flag_stat(rows,i):
 v=[r.get('F'+str(i)) for r in rows];known=[x for x in v if x is not None]
 return dict(count=sum(known) if len(known)==len(v) else None,known_count=sum(known),n=len(v),n_defined=len(known),n_missing=len(v)-len(known),rate=sum(known)/len(v) if len(known)==len(v) and v else None)
def aggregate(records,*,labels,configs,expected_task_ids,wrapper=None):
 missing=sum(missing_report(r) for r in records)
 if not missing:
  value=fm.aggregate_measurements(records,labels=labels,configs=configs,expected_task_ids=expected_task_ids,wrapper=wrapper)
 else:
  src=inspect.getsource(fm.aggregate_measurements)
  old="failures={f'F{i}':dict(count=sum(r[f'F{i}'] for r in primary),n=len(primary)) for i in range(1,11)},"
  new="failures={f'F{i}':_flag_stat(primary,i) for i in range(1,11)},"
  if old not in src:raise RuntimeError('Frozen missing-policy adapter source pattern changed')
  src=src.replace(old,new)
  def objective_unknown(rows,configs):
   return dict(J_mfg=None,components={k:None for k in fm.load_metric_weights(configs)},weights=fm.load_metric_weights(configs),status='missing_full_population_objective',n_expected=len(expected_task_ids),n_observed_stage=len(rows),n_terminal_missing=missing,reason='Certified paid incomplete retained; no successful-only objective or score imputation.')
  ns=dict(fm.__dict__);ns.update(canonical_components=objective_unknown,_flag_stat=flag_stat)
  exec(compile(src,inspect.getsourcefile(fm)+':offline_missing_adapter','exec'),ns)
  value=ns['aggregate_measurements'](records,labels=labels,configs=configs,expected_task_ids=expected_task_ids,wrapper=wrapper)
 value['population_coverage']=dict(expected=len(expected_task_ids),records=len(records),parsed=len(records)-missing,terminal_missing=missing,all_cases_retained=True)
 for name,stage in value['stages'].items():
  stage['n_expected']=len(expected_task_ids);stage['n_stage_missing']=len(expected_task_ids)-stage['n']
 value['missing_policy']='No report/label/unknown scores or success latency imputation; full-case classification and paid usage retained.'
 return value

def paired(left,right,*,dataset,labels,configs,n_resamples=10000):
 missing=any(missing_report(r) for groups in (left,right) for records in groups.values() for r in records)
 if not missing:return fe.paired_test_summary(left,right,dataset=dataset,labels=labels,configs=configs,n_resamples=n_resamples)
 # Same pairing, seed, estimator, run resampling and bootstrap as the frozen
 # function. Only the unavailable objective branch is disabled.
 src=inspect.getsource(fe.paired_test_summary)
 old="j=(weights['alpha_label']*accuracy"
 new="j=(np.full_like(n,np.nan) if _missing_objective else weights['alpha_label']*accuracy"
 if old not in src:raise RuntimeError('Frozen paired function changed')
 src=src.replace(old,new)
 old2="lo,hi=np.percentile(samples[:,i],[2.5,97.5]);p="
 new2="if name=='J' and _missing_objective:\n                result[name]=dict(effect=None,CI95=None,p_bootstrap_centered=None,n_clusters=len(groups),resamples=n_resamples,status='missing_full_population_objective');continue\n            lo,hi=np.percentile(samples[:,i],[2.5,97.5]);p="
 if old2 not in src:raise RuntimeError('Frozen paired summary changed')
 src=src.replace(old2,new2)
 ns=dict(fe.__dict__);ns['_missing_objective']=True
 exec(compile(src,inspect.getsourcefile(fe)+':offline_missing_adapter','exec'),ns)
 value=ns['paired_test_summary'](left,right,dataset=dataset,labels=labels,configs=configs,n_resamples=n_resamples)
 for r in value['independent_run_effects'].values():r['J']=None
 value['J_status']='missing_full_population_objective';value['certified_missing_outcomes_retained']=True
 return value

def failure_summary(records):
 out={}
 for r in records:out.setdefault(ad.group_key(r),[]).append(r)
 result={}
 for key,rs in out.items():
  result[key]={}
  for name in ('initial','post_repair','pre_filter','final'):
   rows=[r['common_measurement']['stages'][name]['primary'] for r in rs if name in r['common_measurement']['stages']]
   if rows:result[key][name]={f'F{i}':{**flag_stat(rows,i),'n_expected':len(rs),'n_stage_absent':len(rs)-len(rows)} for i in range(1,11)}
 return dict(by_system_run_draw_stage=result,denominators='Full saved populations with explicit missing/unavailable stage flags. Repeated stages/draws are not independent cases.',sentinel_scope='Natural flags do not estimate detector sensitivity.')

def action_summary(records):
 groups={}
 for r in records:groups.setdefault(ad.group_key(r),[]).append(r)
 out={}
 for key,rs in groups.items():
  if not any(missing_report(r) for r in rs):out.update(ad.report_action_summary(rs));continue
  stages={}
  for name in ('initial','post_repair','pre_filter','final'):
   vals=[r['common_measurement']['stages'][name] for r in rs if name in r['common_measurement']['stages']]
   if not vals:continue
   counts=[v['supplemental'].get('n_actions') for v in vals];known=[x for x in counts if x is not None]
   unsafe=[v['primary'].get('unsafe_recommendation_rate') for v in vals];ku=[x for x in unsafe if x is not None]
   stages[name]=dict(n_reports=len(vals),n_expected=len(rs),n_stage_absent=len(rs)-len(vals),n_action_count_defined=len(known),n_action_count_missing=len(vals)-len(known),total_actions=sum(known) if len(known)==len(vals) else None,known_action_subtotal=sum(known),n_nonempty_reports=sum(x>0 for x in known),n_empty_reports=sum(x==0 for x in known),action_coverage=sum(x>0 for x in known)/len(vals) if len(known)==len(vals) else None,unsafe_macro_all=sum(ku)/len(vals) if len(ku)==len(vals) else None,unsafe_macro_nonempty=None,unsafe_micro=None,unsafe_action_count=None,linked_action_count=None,linked_action_micro=None,exact_action_counts_available=False,missing_policy='Unobserved actions/safety rates remain null.')
  trans=[r['common_measurement']['transitions'] for r in rs]
  out[key]=dict(stages=stages,transitions=dict(n_reports=len(rs),repair_attempted_reports=sum(t.get('repair_count',0)>0 for t in trans),filter_evaluated_reports=sum(bool(t.get('filter_supported')) for t in trans),raw_transitions=trans),interpretation='Report-policy compliance/coverage only; no physical safety or usefulness inference.')
 return out

