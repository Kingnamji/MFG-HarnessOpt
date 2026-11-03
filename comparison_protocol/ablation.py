"""M2 2x2 inputs: identical visible observations, representation x explicit guide.

This removes only explicit guide/prior text. Source-owned direction tags,
derived values, the TWF risk summary and Azure observation summary are fixed.
It is not a domain-knowledge-free or source-access ablation.
"""
from dataclasses import dataclass
import json
import pandas as pd
from .contracts import digest
from experiments.data.generation_evidence import generation_task,validate_generation_evidence,validate_observation_windows
from experiments.llm.observable_records import observation_records,json_value
from experiments.llm.prompts import build_m2_prompt,build_system_prompt,format_proxy_evidence_table

COLUMNS=['evidence_id','source','field','statistic','direction','value','raw_value','window','event_time','strength','role']

@dataclass(frozen=True)
class AblationInput:
    dataset:str
    cell:str
    task_id:str
    messages:tuple
    observation_records:tuple
    visible_records_sha256:str
    request_sha256:str
    exact_full_M2:bool

def _flat_records(records):
    return '\n'.join('Observation '+str(i+1)+': '+ '; '.join(k+'='+json.dumps(v,ensure_ascii=False,allow_nan=False)for k,v in r.items())
                     for i,r in enumerate(records))or'None'

def build_ablation_input(*,task,generation_evidence,allowed_schema,action_ontology,association_guide=None,representation='structured',guide_on=True):
    if representation not in ('structured','flat'):raise ValueError('Unknown representation')
    if not isinstance(guide_on,bool):raise ValueError('guide_on must be boolean')
    clean_task=generation_task(task);evidence=generation_evidence.copy()if isinstance(generation_evidence,pd.DataFrame)else pd.DataFrame(generation_evidence)
    validate_generation_evidence(evidence,[clean_task['task_id']]);validate_observation_windows(evidence,pd.DataFrame([clean_task]))
    dataset='AI4I'if allowed_schema['dataset']=='AI4I 2020'else'PdM'if allowed_schema['dataset']=='Azure PdM'else None
    if dataset is None:raise ValueError('Unsupported dataset')
    if dataset=='PdM'and association_guide:
        c=association_guide.get('leakage_control',{})
        if c.get('uses_split')!='search'or c.get('uses_validation_labels')is not False or c.get('uses_test_labels')is not False:
            raise ValueError('Association prior must be frozen search-only')
    records=observation_records(evidence,columns=COLUMNS,max_rows=10)
    table=format_proxy_evidence_table(evidence,max_rows=10)if representation=='structured'else _flat_records(records)
    guide=association_guide if (dataset=='AI4I'or guide_on)else None
    user=build_m2_prompt(task=task,proxy_evidence=evidence,allowed_schema=allowed_schema,
        action_ontology=action_ontology,association_guide=guide,evidence_table_text=table)
    if dataset=='AI4I'and not guide_on:
        start='Label decision guide:\n';end='\n\nAzure search-derived association guide:'
        if user.count(start)!=1 or user.count(end)!=1:raise ValueError('Frozen prompt section markers changed')
        left,tail=user.split(start,1);_,right=tail.split(end,1)
        user=left+start+'No explicit label-decision guide is supplied in this ablation condition.'+end+right
    messages=(dict(role='system',content=build_system_prompt()),dict(role='user',content=user))
    if representation=='structured'and guide_on:
        exact=build_m2_prompt(task=task,proxy_evidence=evidence,allowed_schema=allowed_schema,
            action_ontology=action_ontology,association_guide=association_guide)
        if exact!=user:raise AssertionError('Full M2 request differs from the frozen original')
    return AblationInput(dataset,f'{representation}_guide_{"on"if guide_on else"off"}',str(task['task_id']),messages,
        tuple(records),digest(records),digest(messages),representation=='structured'and guide_on)

def run_ablation_input(packet,*,client,iteration=0):
    """One unverified M2 generation. No repair or filtering is added to M2."""
    with client.at(candidate_id='M2_'+packet.cell,iteration=iteration,task_id=packet.task_id,max_report_calls=1):
        return client.generate_structured_report(messages=list(packet.messages))
