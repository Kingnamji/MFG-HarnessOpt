"""Information-separated proposal requests, including a bounded COPRO adapter.

COPRO attribution: https://dspy.ai/current/api/optimizers/COPRO/
Official algorithm: https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/copro_optimizer.py
This is an explicitly constrained single-instruction adaptation, not a call to
native DSPy.compile. Prefix/schema/demos, base harness and model are fixed.
"""
from copy import deepcopy
from collections import Counter
import json,math
from .contracts import Arm,proposal_format,compile_candidate,digest
from .protocol import validate_search_origin,assert_search_tree
from experiments.optimization.proposer_context import build_reference_context,json_safe,enrich_search_failures
from experiments.optimization.harness_proposer import (summarize_failure_bundles,
    build_proposer_system_prompt,build_proposer_user_prompt)
from experiments.llm.observable_records import json_value

METRICS={'n_tasks','label_accuracy','macro_f1','J_mfg','evidence_precision','evidence_recall',
         'evidence_linked_action_ratio','unsupported_claim_rate','unsafe_recommendation_rate','cost_per_report','F10'}

def numeric_metrics(value):
    result={}
    for key in METRICS:
        v=value.get(key)
        if v is None:result[key]=None
        elif isinstance(v,(float,int))and not isinstance(v,bool)and math.isfinite(v):result[key]=v
        else:raise ValueError('Non-numeric aggregate feedback: '+key)
    return result

def metric_feedback(points):
    """Strict reconstruction: no trace, task IDs, labels or free-form errors."""
    return [dict(candidate_id=p['candidate_id'],eligible=bool(p['eligible']),metrics=numeric_metrics(p['metrics']))for p in points]

def public_context(dataset,configs):
    if dataset not in ('AI4I','PdM'):raise ValueError('Unsupported comparison dataset')
    context=build_reference_context([dict(dataset=dataset)],configs)
    # This builder only reads fixed public contracts and search association
    # statistics. There are no selected tasks, reports or target rows here.
    return json_safe(dict(dataset=dataset,M3_configuration=configs.methods['M3'],reference=context,
        model='gpt-5.4-mini',diagnosis_temperature=0,max_repair_steps=1))

def batches(arm,*,T=4,K=3,max_slots=12,breadth=4,depth=4):
    arm=Arm(arm)
    if arm==Arm.TRACE:
        raise ValueError('Full TRACE must use the unchanged native M4 proposer/loop; this adapter does not replace it')
    if any(isinstance(n,bool)or not isinstance(n,int)or n<=0 for n in (T,K,max_slots,breadth,depth)):
        raise ValueError('Positive integer search limits required')
    if T!=4 or K!=3 or max_slots!=12:raise ValueError('This registered comparison fixes T4/K3 and 12 new slots')
    if breadth<=1:raise ValueError('COPRO breadth must exceed one')
    sizes=[breadth-1]+[breadth]*(depth-1)if arm in (Arm.COPRO,Arm.TRACE_PROMPT) else[K]*T
    left=max_slots;out=[]
    for n in sizes:
        if left<=0:break
        out.append(min(n,left));left-=out[-1]
    return out

def copro_attempts(points,*,breadth=4):
    # Native COPRO uses the top B attempts, ordered from lower to higher score.
    # Base M3 is retained separately and does not consume a new-candidate slot.
    complete=[p for p in points if p.get('instruction')is not None]
    top=sorted(complete,key=lambda p:(-p['metrics']['J_mfg'],p['candidate_id']))[:breadth]
    return [dict(instruction=p['instruction'],output_prefix='',score=p['metrics']['J_mfg'])for p in reversed(top)]

def encode_trace_keys(traces):
    """Lossless key dictionary; all source values, reports and cases survive."""
    counts=Counter()
    def count(v):
        if isinstance(v,dict):
            for k,x in v.items():counts[k]+=1;count(x)
        elif isinstance(v,list):
            for x in v:count(x)
    count(traces)
    keys=sorted((k for k,n in counts.items() if len(k)>8 and n>1),key=lambda k:(-counts[k],k))
    aliases={k:f'~{i}' for i,k in enumerate(keys)}
    def encode(v):
        if isinstance(v,dict):return {aliases.get(k,k):encode(x) for k,x in v.items()}
        if isinstance(v,list):return [encode(x) for x in v]
        return v
    return encode(traces),{v:k for k,v in aliases.items()}

def decode_trace_keys(value,dictionary):
    if isinstance(value,dict):return {dictionary.get(k,k):decode_trace_keys(v,dictionary) for k,v in value.items()}
    if isinstance(value,list):return [decode_trace_keys(v,dictionary) for v in value]
    return value

def build_request(*,arm,candidate_id,iteration,slot,seed,public,points,current_candidate=None,earlier_proposals=(),
                  selected_failures=(),configs=None):
    arm=Arm(arm)
    if arm==Arm.TRACE:
        raise ValueError('Use the unchanged native full M4 proposer and loop')
    dataset=public['dataset']
    if arm!=Arm.NONADAPTIVE:
        for p in points: validate_search_origin(p.get('origin'),dataset)
    assert_search_tree(public,dataset)
    if arm in (Arm.TRACE_PROMPT,): assert_search_tree(selected_failures,dataset)
    fixed_base={'base_method':'M3','configuration':public['M3_configuration']}
    body=dict(candidate_id=candidate_id,public_context=public,
        proposal_slot=dict(iteration=iteration,slot=slot,seed_tag=str(seed)),
        edit_scope='one instruction delta only; all other M3 components fixed'if arm.prompt_only else'six constrained harness modules',
        current_harness=fixed_base)
    if arm==Arm.NONADAPTIVE:
        body['proposal_policy']='Sample a candidate from the fixed M3 contract. No performance history or updated incumbent is available.'
    elif arm==Arm.COPRO:
        body['basic_instruction']='No additional instruction; the complete fixed M3 task contract is in public_context.'
        body['generation_stage']='basic_instruction' if iteration==1 else 'ranked_attempts'
        body['attempted_instructions']=[] if iteration==1 else copro_attempts(points)
        body['proposal_policy']='Propose a new instruction informed by the highest-scoring previous instructions, listed from lower to higher score. Keep output_prefix empty.'
    else:
        body['aggregate_search_history']=metric_feedback(points)
        body['current_harness']=compile_candidate(arm,current_candidate['candidate_id'],current_candidate).payload if current_candidate else fixed_base
        body['earlier_proposals']=deepcopy(list(earlier_proposals))
        if arm in (Arm.TRACE,Arm.TRACE_PROMPT):
            if configs is None:raise ValueError('Trace context requires frozen configs')
            # Authoritative search split/observable-only projection is shared
            # with the existing frozen M4 proposer, not a raw trace dump.
            body['observable_failure_traces']=summarize_failure_bundles(enrich_search_failures(list(selected_failures),configs))
    system=('Propose a manufacturing diagnostic harness candidate under the fixed public contract. '
        'Return only the requested JSON schema. Preserve source metadata, required report fields, public rules, action ontology, '
        'the maximum of one repair, and frozen data/evaluator. Do not memorize individual tasks. '
        'The patch content or instruction must be your proposal; no rule-based replacement will be supplied.')
    safe=json_value(json_safe(body))
    if arm==Arm.METRIC:
        if configs is None: raise ValueError('Metric-only uses the frozen native Proposer prompt backbone')
        methods={k:configs.methods[k] for k in sorted(configs.methods)}
        constraints={k:configs.experiment.get(k,{}) for k in
                     ('harness_constraints','m3_fixed_repair','optimization','candidate_screening')}
        numeric=metric_feedback(points)
        feedback=dict(scope='search',M3_reference=[p for p in numeric if p['candidate_id']=='M3'],
                      candidates=[p for p in numeric if p['candidate_id']!='M3'])
        native=build_proposer_user_prompt(candidate_id=candidate_id,selected_failures=[],
            method_configs=methods,experiment_constraints=constraints,
            current_harness=body['current_harness'],reference_context=public['reference'],search_feedback=None)
        # The native template and all public domain/contract guidance are reused.
        # Only feedback-specific prose and the aggregate block change. No case
        # trace, task-level correctness, field correction or failure category is supplied.
        start='Aggregate search candidate feedback (never validation/test feedback):\n'
        end='\n\nCurrent harness configuration'
        left,tail=native.split(start,1);_,right=tail.split(end,1)
        native=left+start+json.dumps(feedback,ensure_ascii=False,sort_keys=True)+end+right
        native=native.replace('from execution failure traces','from aggregate search metrics')
        native=native.replace('justified by actual observed failures','justified by the available aggregate metrics')
        native+='\n\nMetric-only feedback boundary: task reports, typed failure cases and per-task correctness are withheld. Use aggregate metrics and the current configuration; do not invent missing cases.\n'
        native+='Proposal slot: '+json.dumps(body['proposal_slot'],sort_keys=True)
        system=build_proposer_system_prompt().replace('from execution failure traces','from aggregate search metrics')
        if len(native)+len(system)>200000: raise ValueError('Comparison proposal exceeds the common context bound')
        return dict(messages=[dict(role='system',content=system),dict(role='user',content=native)],
                    text_format=proposal_format(arm),reasoning_effort='none')
    text=json.dumps(safe,ensure_ascii=False,allow_nan=False)
    if len(text)>200000:
        # Lossless whitespace compaction only. Keep every task, observation,
        # report delta and regression case under the same fixed context cap.
        text=json.dumps(safe,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    if len(text)>200000 and 'observable_failure_traces' in safe:
        original=safe['observable_failure_traces']
        encoded,dictionary=encode_trace_keys(original)
        if decode_trace_keys(encoded,dictionary)!=original:raise AssertionError('Trace encoding changed information')
        safe['observable_failure_traces']=encoded
        safe['trace_context_key_dictionary']=dictionary
        safe['trace_context_encoding']='Within observable_failure_traces only, decode dictionary aliases as original JSON keys. All values/cases are unchanged. Output schema keys remain the original fixed keys.'
        text=json.dumps(safe,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    if len(text)>200000:raise ValueError('Comparison proposal exceeds the common context bound')
    return dict(messages=[dict(role='system',content=system),dict(role='user',content=text)],
        text_format=proposal_format(arm),reasoning_effort='none')
