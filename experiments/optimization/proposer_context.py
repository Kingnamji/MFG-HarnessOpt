"""Search-only observable context for LLM harness proposal, without target rows."""
from pathlib import Path
import math
import pandas as pd
from experiments.data.generation_evidence import TASK_FIELDS, generation_task, visible_evidence, validate_generation_evidence, validate_observation_windows
from experiments.evaluation.metric_policy import EVALUATION_POLICY_VERSION
from experiments.llm.report_contract import REPORT_CONTRACT_VERSION
from experiments.evaluation.runtime_verifier import verify_runtime_report
from experiments.evaluation.runtime_verifier import audit_runtime_repair
import json

def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if value is None:return None
    if isinstance(value,(pd.Timestamp,)):return value.isoformat() if not pd.isna(value) else None
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def build_reference_context(failures,configs):
    """Public executable M3 base, rather than YAML summaries alone."""
    from experiments.llm.prompts import build_m3_prompt,format_label_decision_guide
    from experiments.llm.report_contract import format_common_contract
    from experiments.llm.observable_records import format_minimal_repair_contract
    from experiments.harness.verified_postprocessing import format_action_filter_contract
    from experiments.data.azure_observation_context import format_context_contract
    aliases={'AI4I':'ai4i','AI4I 2020':'ai4i','PdM':'pdm','Azure':'pdm','Azure PdM':'pdm'}
    bases=[]
    for key in sorted({aliases[f['dataset']] for f in failures}):
        schema=json.loads(configs.paths['dataset'][key]['allowed_schema'].read_text(encoding='utf-8'))
        if key=='ai4i':
            from experiments.runners.run_smoke_test import make_default_action_configs
            ontology,preconditions,generic=make_default_action_configs()
            association_guide={}
        else:
            ontology=json.loads(configs.paths['dataset']['pdm']['action_ontology'].read_text(encoding='utf-8'))
            from experiments.data.load_preprocessed import load_azure_association_guide
            folder=Path(configs.paths['dataset']['pdm']['dir'])
            if not folder.is_absolute():folder=configs.project_root/folder
            association_guide=load_azure_association_guide(folder/'azure_search_association_guide.json')
            preconditions,generic=(json.loads((folder/name).read_text(encoding='utf-8'))
                for name in ['action_preconditions.json','generic_action_rules.json'])
        bases.append(dict(dataset=schema['dataset'],allowed_schema=schema,
            action_ontology=ontology,action_preconditions=preconditions,generic_action_rules=generic,
            M4_execution_policy='Exact M3 base with proposer-owned configuration deltas; no fixed M4-only Azure association policy',
            label_decision_guide=format_label_decision_guide(schema),
            M3_base_prompt_template=build_m3_prompt(
                task={'task_id':'OBSERVABLE_TASK_PLACEHOLDER','dataset':schema['dataset']},
                proxy_evidence=[],allowed_schema=schema,action_ontology=ontology,association_guide=association_guide)))
    return dict(report_contract_version=REPORT_CONTRACT_VERSION,common_contract=format_common_contract(),
        input_observation_contract=dict(forbidden_target_columns=['gold_label','supports','failure flags'],
            permitted_columns=['evidence_id','source','field','statistic','direction','value','raw_value','window','event_time','strength','role']),
        output_report_contract=dict(required_fields=['predicted_label','confidence','evidence','next_steps','limitations'],
            required_evidence_fields=['evidence_id','source','field','statistic','direction','value','window','supports','role'],
            supports='Required OUTPUT inference: exactly one task candidate label or context. This is distinct from forbidden hidden INPUT support annotations.'),
        shared_repair_contract=format_minimal_repair_contract(),
        auxiliary_observation_contract=format_context_contract(),
        fixed_final_action_filter=format_action_filter_contract(),
        observation_format='JSON records with explicit field names; preserve supplied statistic/direction/value separately',
        executable_M3_bases=bases,
        objective=dict(S_label='mean label correctness (Paper); macro-F1 is separately monitored',
            weights=configs.metric_weights['primary_selection'],normalization=configs.metric_weights['normalization'],
            candidate_acceptance=configs.experiment.get('candidate_acceptance',{})),
        feedback_semantics=dict(F1='Offline search label error; no gold label is supplied',
            F3='Observable value/direction/statistic/role mismatch; use field-specific expected values',
            F7='Action with no admissible observable evidence reference',
            runtime_scope='Record compliance and narrow public-condition consistency do not guarantee label correctness; TWF and Azure remain probabilistic.'))


def safe_search_feedback(feedback):
    if feedback is None:return {}
    if feedback.get('scope')!='search':raise ValueError('Only aggregate search feedback may reach the proposer')
    allowed={'dataset','n_tasks','label_accuracy','macro_f1','J_mfg','evidence_precision','evidence_recall',
             'evidence_linked_action_ratio','unsupported_claim_rate','unsafe_recommendation_rate','cost_per_report','F10'}
    metrics=lambda rows:[{k:v for k,v in row.items() if k in allowed} for row in rows]
    request=feedback.get('proposal_request',{})
    request={k:request[k] for k in ['iteration','slot','candidates_per_iteration'] if k in request}
    peers=[{k:p[k] for k in ['candidate_id','patch_signature','prompt_instructions'] if k in p}
           for p in feedback.get('earlier_proposals',[])]
    return json_safe(dict(scope='search',report_contract_version=REPORT_CONTRACT_VERSION,
        incumbent_candidate_id=feedback.get('incumbent_candidate_id','M3'),
        paired_candidate_summary=[{k:v for k,v in row.items() if k in {
            'candidate_id','dataset','n_tasks','regressions_vs_incumbent','fixes_vs_incumbent',
            'unchanged_correct','unchanged_wrong'}} for row in feedback.get('paired_candidate_summary',[])],
        proposal_request=request,earlier_proposals=peers,
        M3_reference=metrics(feedback.get('M3_reference',[])),candidates=[dict(
            candidate_id=c.get('candidate_id'),eligible=c.get('eligible'),
            rejection_reasons=c.get('rejection_reasons',[]),metrics=metrics(c.get('metrics',[])))
            for c in feedback.get('candidates',[])]))

def enrich_search_failures(failures,configs):
    """Check both declared split and authoritative task manifest before any client.

    Only observation files and a task-column allowlist are read. No gold labels,
    evaluator target annotations, validation records or test records are exposed.
    """
    datasets={
        'AI4I':('ai4i','ai4i'),'AI4I 2020':('ai4i','ai4i'),
        'PdM':('pdm','azure'),'Azure':('pdm','azure'),'Azure PdM':('pdm','azure'),
    }
    result=[];cache={}
    for failure in failures:
        if failure.get('split')!='search':raise ValueError('Proposer accepts declared search traces only')
        if failure.get('metric_row',{}).get('evaluation_policy_version')!=EVALUATION_POLICY_VERSION:
            raise ValueError('Search traces must be newly evaluated under the current policy; legacy scores/reports cannot seed revised optimization')
        if failure.get('metric_row',{}).get('report_contract_version')!=REPORT_CONTRACT_VERSION:
            raise ValueError('Search traces predate the current report contract; regenerate seed reports before optimization')
        dataset=failure.get('dataset')
        if dataset not in datasets:raise ValueError(f'Unknown proposer dataset: {dataset}')
        key,suffix=datasets[dataset]
        if key not in cache:
            folder=Path(configs.paths['dataset'][key]['dir'])
            if not folder.is_absolute():folder=configs.project_root/folder
            tasks=pd.read_csv(folder/f'tasks_{suffix}.csv',usecols=lambda c:c in TASK_FIELDS)
            if key == 'pdm':
                from experiments.data.azure_observation_context import attach_contexts
                tasks = attach_contexts(tasks, folder)
            evidence=pd.read_csv(folder/f'generation_evidence_{suffix}.csv',parse_dates=['event_time'])
            validate_generation_evidence(evidence,tasks.task_id);validate_observation_windows(evidence,tasks)
            schema=json.loads((folder/f'allowed_schema_{suffix}.json').read_text(encoding='utf-8'))
            if key=='ai4i':
                from experiments.runners.run_smoke_test import make_default_action_configs
                ontology,preconditions,generic=make_default_action_configs()
            else:
                ontology,preconditions,generic=(json.loads((folder/name).read_text(encoding='utf-8'))
                    for name in ['action_ontology.json','action_preconditions.json','generic_action_rules.json'])
            cache[key]=(tasks,evidence,schema,ontology,preconditions,generic)
        tasks,evidence,schema,ontology,preconditions,generic=cache[key];task_id=str(failure.get('task_id',''))
        selected=tasks.loc[tasks.task_id.eq(task_id)&tasks.split.eq('search')]
        if len(selected)!=1:raise ValueError(f'Proposer task is not a unique search task: {task_id}')
        # Raw selected bundles may contain offline annotations; only explicitly
        # projected fields are used later by compact_failure_bundle.
        observations=evidence.loc[evidence.task_id.eq(task_id)]
        feedback=verify_runtime_report(report=failure.get('report') or {},generation_evidence=observations,
            allowed_schema=schema,action_ontology=ontology,action_preconditions=preconditions,generic_action_rules=generic)
        initial=failure.get('initial_report')
        repair_trace=None
        if isinstance(initial,dict):
            before=verify_runtime_report(report=initial,generation_evidence=observations,
                allowed_schema=schema,action_ontology=ontology,action_preconditions=preconditions,generic_action_rules=generic)
            repair_trace=dict(initial_report=initial,initial_observable_feedback=before,
                repair_metadata_audit=audit_runtime_repair(initial,failure.get('report') or {},observations))
        result.append({**failure,'observable_task':generation_task(selected.iloc[0].to_dict()),
            'available_observations':visible_evidence(observations).to_dict('records'),
            'observable_feedback':feedback,'repair_trace':repair_trace})
    return json_safe(result)
