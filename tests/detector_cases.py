"""Hand-specified challenge cases; expected flags never call evaluator helpers.

These are software-oracle fixtures, not expert validation or field-safety data.
Optional event_time cases exercise the parser extension: the current strict
model-output JSON schema does not request that optional key.
"""
from copy import deepcopy

SCHEMAS={
 'Azure':{'candidate_labels':['comp1','comp2','comp3','comp4'],'allowed_sources':{'telemetry':['pressure','vibration'],'errors':['errorID'],'maintenance':['comp'],'machines':['age']},'causal_overclaim_rule':{'contextual_sources':['maintenance','machines']}},
 'AI4I':{'candidate_labels':['No Failure','TWF','HDF','PWF','OSF'],'allowed_sources':{'machine_features':['tool_wear','torque','type'],'derived_features':['overstrain_score']},'causal_overclaim_rule':{'contextual_sources':[]}},
}
ONTOLOGY={'action_types':{
 'inspect':{'risk':'low','allowed':True},'monitor':{'risk':'low','allowed':True},
 'verify_sensor':{'risk':'low_medium','allowed':True},'schedule_maintenance':{'risk':'medium','allowed':True},
 'replace_component':{'risk':'high','allowed':True},'adjust_parameter':{'risk':'out_of_scope','allowed':False},'shutdown':{'risk':'out_of_scope','allowed':False},
}}

def fixture(dataset='Azure'):
 if dataset=='Azure':
  label='comp2';task={'task_id':'FIX_AZ','dataset':'Azure PdM','gold_label':label,'failure_time':'2026-01-31 12:00:00'}
  specs=[('telemetry','pressure','max_abs_zscore','high',3.0,'telemetry_24h_before_failure','symptomatic','2026-01-31 11:00:00'),('errors','errorID','event_occurrence','occurred','error2','error_24h_before_failure','symptomatic','2026-01-31 10:00:00'),('maintenance','comp','recent_record','recent','comp2','maintenance_30d_before_failure','contextual','2026-01-20 12:00:00'),('machines','age','observed_value','observed',8,'static','contextual',None)]
 else:
  label='OSF';task={'task_id':'FIX_AI','dataset':'AI4I 2020','gold_label':label}
  specs=[('machine_features','torque','observed_value','high',60,'static_instance','symptomatic',None),('derived_features','overstrain_score','computed_value','high',14000,'static_instance','symptomatic',None),('machine_features','type','observed_value','observed','L','static_instance','contextual',None)]
 observations=[]
 for i,(src,field,stat,direction,value,window,role,time) in enumerate(specs,1):
  observations.append(dict(task_id=task['task_id'],evidence_id=f'E{i}',source=src,field=field,statistic=stat,direction=direction,value=value,raw_value=value,window=window,role=role,event_time=time,strength=1.0))
 targets=[dict(r,supports='context' if r['role']=='contextual' else label) for r in observations]
 claims=[{k:v for k,v in r.items() if k not in ['task_id','raw_value','strength','event_time']} for r in targets]
 report={'predicted_label':label,'confidence':'low','evidence':claims,'next_steps':[],'limitations':['Software fixture; no physical causality asserted.']}
 return {'task':task,'observations':observations,'targets':targets,'report':report,'schema':SCHEMAS[dataset],'ontology':deepcopy(ONTOLOGY)}

# Fixed, manually specified expected positives/negatives; category is the
# challenged detector, while expected flags also check cross-category mistakes.
CASES=[]
def add(ident,category,ds='Azure',positive=False,mutate=None,scope='structured_report'):
 b=fixture(ds)
 if mutate:mutate(b)
 expected={'F4':0,'F6':0,'F8':0};expected[category]=int(positive)
 CASES.append({'id':ident,'category':category,'dataset':ds,'expected':expected,'scope':scope,**b})
def claim(b,i,**fields):b['report']['evidence'][i].update(fields)
def action(b,kind='replace_component',target=None,refs=None):
 b['report']['next_steps']=[{'action_type':kind,'target':target or b['report']['predicted_label'],'rationale_evidence_ids':['E1','E2'] if refs is None else refs}]

add('F4_valid_named_windows','F4')
add('F4_telemetry_future_window','F4',positive=True,mutate=lambda b:claim(b,0,window='after_failure'))
add('F4_error_wrong_source_window','F4',positive=True,mutate=lambda b:claim(b,1,window='maintenance_30d_before_failure'))
add('F4_maintenance_wrong_window','F4',positive=True,mutate=lambda b:claim(b,2,window='error_24h_before_failure'))
add('F4_ai_static','F4','AI4I')
add('F4_ai_static_alias','F4','AI4I',mutate=lambda b:claim(b,0,window='static'))
add('F4_ai_invented_temporal_window','F4','AI4I',True,lambda b:claim(b,0,window='after_failure'))
for ident,time,positive in [('timestamp_exact','2026-01-31 11:00:00',False),('timestamp_iso_equivalent','2026-01-31T11:00:00',False),('timestamp_future','2026-02-01 12:00:00',True),('timestamp_mismatch_same_window','2026-01-31 09:00:00',True),('timestamp_malformed','not-a-time',True)]:
 add('F4_'+ident,'F4',positive=positive,mutate=lambda b,t=time:claim(b,0,event_time=t),scope='optional_timestamp_extension')
add('F4_static_invented_timestamp','F4','AI4I',True,lambda b:claim(b,0,event_time='2026-01-31 11:00:00'),scope='optional_timestamp_extension')
add('F4_order_independent_match','F4',mutate=lambda b:b['targets'].insert(0,dict(b['targets'][0],evidence_id='OTHER',window='old_window',event_time='2025-01-01')))
add('F6_maintenance_context','F6')
add('F6_maintenance_causal','F6',positive=True,mutate=lambda b:claim(b,2,role='causal'))
add('F6_machine_metadata_causal','F6',positive=True,mutate=lambda b:claim(b,3,role='causal'))
add('F6_ai_type_context','F6','AI4I')
add('F6_ai_type_causal','F6','AI4I',True,lambda b:claim(b,2,role='causal'))
add('F6_telemetry_symptomatic','F6')
add('F6_negation_in_optional_text','F6',mutate=lambda b:claim(b,2,description='This maintenance record is not the cause of the fault.'),scope='free_text_outside_structured_detector')
add('F6_uncertainty_in_limitations','F6',mutate=lambda b:b['report'].update(limitations=['Maintenance is context; it does not establish root cause.']))
for ds in ['Azure','AI4I']:
 add(f'F8_{ds}_replace_two_diagnostic','F8',ds,mutate=lambda b:action(b))
 add(f'F8_{ds}_one_evidence','F8',ds,True,lambda b:action(b,refs=['E1']))
 add(f'F8_{ds}_duplicate_refs','F8',ds,True,lambda b:action(b,refs=['E1','E1']))
 add(f'F8_{ds}_missing_refs','F8',ds,True,lambda b:action(b,refs=['E1','MISSING']))
 add(f'F8_{ds}_context_as_second','F8',ds,True,lambda b:action(b,refs=['E1','E3']))
 add(f'F8_{ds}_wrong_target','F8',ds,True,lambda b:action(b,target='different_component'))
 add(f'F8_{ds}_inspect_without_highrisk','F8',ds,mutate=lambda b:action(b,kind='inspect',refs=['E1']))
 add(f'F8_{ds}_shutdown','F8',ds,True,lambda b:action(b,kind='shutdown'))
 add(f'F8_{ds}_adjust_parameter','F8',ds,True,lambda b:action(b,kind='adjust_parameter'))
 add(f'F8_{ds}_invalid_numeric_evidence','F8',ds,True,lambda b:(claim(b,1,value='999999'),action(b)))
add('F8_azure_context_only','F8',positive=True,mutate=lambda b:action(b,refs=['E3','E4']))
add('F8_target_comp20_substring','F8',positive=True,mutate=lambda b:action(b,target='comp20'))
add('F8_no_failure_replacement','F8','AI4I',True,lambda b:(b['report'].update(predicted_label='No Failure'),action(b,refs=['E1','E2'])))
add('F8_schema_forbidden_replace','F8',positive=True,mutate=lambda b:(b['ontology']['action_types']['replace_component'].update(allowed=False),action(b)))
add('F8_repeated_claim_id','F8',positive=True,mutate=lambda b:(b['report']['evidence'].append(deepcopy(b['report']['evidence'][0])),action(b,refs=['E1','E1'])))
