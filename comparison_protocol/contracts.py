"""Candidate contracts and fixed-source provenance for the separate comparison."""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib,json,re
from pathlib import Path

from experiments.llm.harness_patch_schema import EDITABLE,FROZEN
from experiments.optimization.harness_proposer import validate_generated_patch
from experiments.optimization.candidate_screening import patch_signature
from experiments.utils.provenance import runtime_provenance

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'harness_configs/reference_lock.json'

class Arm(str,Enum):
    TRACE='trace'
    METRIC='metric_only'
    NONADAPTIVE='nonadaptive'
    TRACE_PROMPT='trace_prompt'
    COPRO='copro'

    @property
    def prompt_only(self):return self in (Arm.TRACE_PROMPT,Arm.COPRO)

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()

def verify_frozen_reference():
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    runtime_provenance.cache_clear()
    if runtime_provenance()!=lock['protocol']:raise ValueError('Frozen M0–M4 code/config/data changed')
    for item in lock['selected_patches'].values():
        if hashlib.sha256(Path(item['original_path']).read_bytes()).hexdigest()!=item['original_sha256']:
            raise ValueError('Frozen reference patch changed')
    return lock['protocol']['code_data_config_sha256']

def m3_envelope(candidate_id):
    """Validation-only envelope. Not a proposed patch or a substitute Proposer."""
    return dict(candidate_id=candidate_id,base_method='M3',objective='Constrained comparison under the frozen Paper objective.',
        editable_modules=list(EDITABLE),frozen_modules=list(FROZEN),
        source_policy=dict(proposed_changes=[],rationale='Fixed M3 source access.'),
        evidence_table_format=dict(proposed_columns=['evidence_id','source','field','statistic','direction','value','raw_value','window','event_time','strength','role'],
            forbidden_columns=['supports','gold_label','gold','ground_truth','target_label','true_label'],formatting_rules=[]),
        prompt_instructions=dict(diagnosis_instructions=[],evidence_instructions=[],action_instructions=[]),
        action_filter_rules=dict(rules=[],out_of_scope_actions=['shutdown','adjust_parameter']),
        repair_prompt=dict(repair_instructions=[],failure_specific_repairs={f'F{i}':[]for i in range(1,11)}),
        repair_trigger_thresholds=dict(**{k:True for k in ['schema_invalid','F3','F4','F5','F6','F7','F8']},unsafe_recommendation_rate_gt=0.0,unsupported_claim_rate_gt=None),
        expected_effects=dict(targeted_failure_reductions=[],expected_tradeoffs=[]),risks=[],rationale='Fixed contract envelope.')

@dataclass(frozen=True)
class Candidate:
    candidate_id:str
    arm:Arm
    payload_json:str
    signature:str

    @property
    def payload(self):return json.loads(self.payload_json)

    @property
    def instruction(self):
        if not self.arm.prompt_only:raise ValueError('Only instruction-only candidates have a single instruction')
        return self.payload['instruction']

def compile_candidate(arm,candidate_id,payload):
    """Validate only; never repair, normalize, or hand-edit proposal content."""
    arm=Arm(arm)
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError):raise ValueError('Nonfinite/non-JSON proposal')from None
    if not isinstance(payload,dict)or payload.get('candidate_id')!=candidate_id:raise ValueError('Candidate identity mismatch')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_\-]{0,100}',candidate_id):raise ValueError('Unsafe candidate ID')
    if arm.prompt_only:
        if set(payload)!={'candidate_id','instruction','output_prefix'}:raise ValueError('Instruction-only edit scope exceeded')
        text=payload['instruction']
        if not isinstance(text,str)or not text.strip()or len(text)>12000:raise ValueError('Invalid instruction')
        if payload['output_prefix']!='':raise ValueError('Structured JSON prefix is fixed')
        envelope=m3_envelope(candidate_id)
        envelope['prompt_instructions']['diagnosis_instructions']=[text]
        errors,_=validate_generated_patch(envelope,candidate_id)
        if errors:raise ValueError('; '.join(errors))
        signature=digest(dict(instruction=text,output_prefix=''))
    else:
        errors,_=validate_generated_patch(payload,candidate_id)
        if errors:raise ValueError('; '.join(errors))
        signature=patch_signature(payload)
    return Candidate(candidate_id,arm,json.dumps(payload,ensure_ascii=False,allow_nan=False),signature)

def proposal_format(arm):
    if not Arm(arm).prompt_only:
        from experiments.llm.harness_patch_schema import get_patch_text_format
        return get_patch_text_format()
    return dict(type='json_schema',name='fixed_m3_instruction_delta',strict=True,schema=dict(type='object',additionalProperties=False,
        properties=dict(candidate_id=dict(type='string'),instruction=dict(type='string'),output_prefix=dict(type='string',enum=[''])),
        required=['candidate_id','instruction','output_prefix']))
