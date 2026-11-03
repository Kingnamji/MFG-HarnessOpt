"""Common, sufficiently detailed report and action template for every comparator.

The report never receives targets. Statistical classifier explanations stay in
the audit and limitations; they do not become invented sensor measurements.
"""
from copy import deepcopy
import pandas as pd
from experiments.llm.observable_records import report_copy_record
from experiments.harness.verified_postprocessing import finalize_verified_report
from .predictors import Prediction, ai4i_conditions, pattern_key, matched_associations


def evidence_interpretations(view):
    """Public interpretations for each record, independently of chosen label."""
    relations, exclusions = {}, []
    if 'TWF' in view.labels:
        _, checks = ai4i_conditions(view)
        fields = {'HDF': {'temperature_gap','rotational_speed'},
                  'PWF': {'mechanical_power_w'}, 'OSF': {'overstrain_score'}, 'TWF': {'tool_wear'}}
        for row in view.records:
            support = 'context'
            if row.get('role') != 'contextual':
                modes = [label for label in ('HDF','PWF','OSF','TWF')
                         if checks[label] is True and row['field'] in fields[label]]
                if modes:
                    support = modes[0]
                elif all(v is False for v in checks.values()) and row['field'] in set.union(*fields.values()):
                    support = 'No Failure'
            relations[str(row['evidence_id'])] = support
        exclusions = [label for label, status in checks.items() if status is False]
    else:
        mapping = {item['evidence_key']:item for item in matched_associations(view)}
        for row in view.records:
            item = mapping.get(pattern_key(row))
            support = 'context'
            if item and row.get('role') != 'contextual':
                scores = item['component_distribution']
                maximum = max(float(scores.get(k,0)) for k in view.labels)
                top = [k for k in view.labels if abs(float(scores.get(k,0))-maximum)<1e-10]
                # A tied prior does not justify assigning a diagnostic relation.
                if len(top)==1:
                    support = top[0]
            relations[str(row['evidence_id'])] = support
    return relations, exclusions


def render_report(*, view, prediction, allowed_schema, action_ontology,
                  action_preconditions=None, generic_action_rules=None):
    if prediction.label not in view.labels or prediction.confidence not in ('low','medium','high'):
        raise ValueError('Prediction outside the frozen report schema')
    interpretations, exclusions = evidence_interpretations(view)
    evidence = []
    for row in view.records:
        item = report_copy_record(row)
        item['supports'] = interpretations[str(row['evidence_id'])]
        evidence.append(item)
    limitations = list(prediction.limitations)
    if exclusions:
        limitations.append('Public conditions not met (exclusion, not causal proof): '+', '.join(exclusions)+'.')
    opposing = sorted({e['supports'] for e in evidence if e['supports'] not in ('context', prediction.label)})
    if opposing:
        limitations.append('Visible records also support competing interpretations: '+', '.join(opposing)+'.')
    if not any(e['supports']==prediction.label for e in evidence):
        limitations.append('No citable record has an independent public-rule/prior relation to the selected label. Statistical prediction alone does not establish physical cause.')
    if view.task.get('azure_observation_context'):
        limitations.append('Four-channel/event summaries inform the predictor but have no citable evidence IDs. They cannot alone justify actions or overwrite a cited observation.')
    actions = []
    # Inspect exact recorded conditions, with opposing evidence retained too.
    groups = {}
    for item in evidence:
        groups.setdefault(item['source'], []).append(item)
    for source, items in groups.items():
        fields = ', '.join(dict.fromkeys(e['field'] for e in items))
        refs = [e['evidence_id'] for e in items]
        if source == 'maintenance':
            target = 'Review documented maintenance history ('+fields+'); verify its timing against current observations'
        elif source == 'errors':
            target = 'Inspect recorded error codes '+', '.join(str(e['value']) for e in items)+' and check for correlated events'
        else:
            target = 'Inspect recorded '+source+' conditions: '+fields
        actions.append(dict(action_type='inspect', target=target, rationale_evidence_ids=refs))
    sensors = [e for e in evidence if e['source'] in ('telemetry','machine_features','derived_features') and e['field'] != 'type']
    if sensors:
        actions.append(dict(action_type='verify_sensor',
            target='Verify measurement acquisition for '+', '.join(dict.fromkeys(e['field'] for e in sensors)),
            rationale_evidence_ids=[e['evidence_id'] for e in sensors]))
    if evidence and (prediction.confidence=='low' or opposing):
        actions.append(dict(action_type='collect_more_data',
            target='Collect a follow-up observation of '+', '.join(dict.fromkeys(e['source']+'.'+e['field'] for e in evidence))+' to resolve diagnostic uncertainty',
            rationale_evidence_ids=[e['evidence_id'] for e in evidence]))
    if not evidence:
        limitations.append('No admissible citable observations. Actions are empty because no rationale IDs are available; empty actions earn no linkage credit.')
    raw = dict(predicted_label=prediction.label, confidence=prediction.confidence,
        evidence=evidence, next_steps=actions, limitations=list(dict.fromkeys(limitations)))
    output = finalize_verified_report(report=raw, generation_evidence=pd.DataFrame(view.records),
        allowed_schema=allowed_schema, action_ontology=action_ontology,
        action_preconditions=action_preconditions, generic_action_rules=generic_action_rules)
    output.update(raw_report=raw, prediction_audit=deepcopy(prediction.audit),
        input_view_sha256=view.sha256, template_version='nonllm_common_report_v1',
        citations=dict(available_ids=[e['evidence_id'] for e in evidence],
            relations=interpretations, exclusionary_conditions=exclusions),
        usage=dict(API_calls=0,input_tokens=0,output_tokens=0,API_cost_usd=0))
    return output


def same_label_control(*, label, view, label_origin, inherited_usage=None, **report_options):
    """Accept only a label and original observations, never an LLM report.

    End-to-end API usage remains unknown unless supplied from the label log.
    """
    prediction = Prediction(label, 'low', limitations=[
        'Conditional same-label report control. The inherited label was produced by '+str(label_origin)+'.'],
        audit=dict(method='same-label control', label_origin=str(label_origin)))
    result = render_report(view=view, prediction=prediction, **report_options)
    result['usage'] = dict(report_API_calls=0, report_API_cost_usd=0,
        label_origin=label_origin, inherited_label_usage=deepcopy(inherited_usage),
        end_to_end_API_cost_usd=None if inherited_usage is None else inherited_usage.get('cost_usd'),
        fully_nonLLM_diagnosis=False)
    return result
