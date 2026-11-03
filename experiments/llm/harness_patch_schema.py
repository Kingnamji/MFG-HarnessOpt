"""Strict Structured Outputs contract for executable harness configurations."""
from copy import deepcopy
from jsonschema import Draft202012Validator

PATCH_SCHEMA_VERSION = 'harness_patch_v1'
EDITABLE = ['source_policy','evidence_table_format','prompt_instructions','action_filter_rules','repair_prompt','repair_trigger_thresholds']
FROZEN = ['datasets','gold_labels','proxy_evidence_builder','evaluator','metric_definitions','test_split','candidate_labels','hidden_supports_annotations','preprocessed_files','allowed_schema','action_ontology','task_sampling']

def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def strings(enum=None):
    item={'type':'string'}
    if enum is not None:item['enum']=enum
    return {'type':'array','items':item}

S={'type':'string'}
N={'type':['number','null'],'minimum':0,'maximum':1}
HARNESS_PATCH_SCHEMA=obj({
 'candidate_id':S,'base_method':{'type':'string','enum':['M3']},'objective':S,
 'editable_modules':strings(EDITABLE),'frozen_modules':strings(FROZEN),
 'source_policy':obj({'proposed_changes':strings(),'rationale':S}),
 'evidence_table_format':obj({'proposed_columns':strings(['evidence_id','source','field','statistic','direction','value','raw_value','window','event_time','strength','role']),'forbidden_columns':strings(),'formatting_rules':strings()}),
 'prompt_instructions':obj({'diagnosis_instructions':strings(),'evidence_instructions':strings(),'action_instructions':strings()}),
 'action_filter_rules':obj({'rules':strings(),'out_of_scope_actions':strings()}),
 'repair_prompt':obj({'repair_instructions':strings(),'failure_specific_repairs':obj({f'F{i}':strings() for i in range(1,11)})}),
 'repair_trigger_thresholds':obj({**{k:{'type':'boolean'} for k in ['schema_invalid','F3','F4','F5','F6','F7','F8']},'unsafe_recommendation_rate_gt':{'type':'number','minimum':0,'maximum':1},'unsupported_claim_rate_gt':N}),
 'expected_effects':obj({'targeted_failure_reductions':strings([f'F{i}' for i in range(1,11)]),'expected_tradeoffs':strings()}),
 'risks':strings(),'rationale':S,
})

def get_patch_text_format():
    return {'type':'json_schema','name':'manufacturing_harness_patch','strict':True,'schema':deepcopy(HARNESS_PATCH_SCHEMA)}

def patch_schema_errors(value):
    validator=Draft202012Validator(HARNESS_PATCH_SCHEMA)
    return [f"{'.'.join(map(str,e.absolute_path)) or 'root'}: {e.message}" for e in sorted(validator.iter_errors(value),key=lambda e:str(e.absolute_path))]
