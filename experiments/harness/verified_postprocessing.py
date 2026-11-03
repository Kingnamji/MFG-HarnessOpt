"""Observable-only final action filtering shared by M3 and M4.

The fixed evaluator is reused, not modified. Labels, confidence, evidence and
limitations are never rewritten here. Raw model reports must be logged apart
from the returned filtered report. This module makes no model/API calls.
"""
from copy import deepcopy

from experiments.evaluation.action_evaluator import evaluate_actions
from experiments.evaluation.runtime_verifier import verify_runtime_report
from experiments.evaluation.structured_report_schema import normalize_action_type


ACTION_FILTER_VERSION = 'observable_action_filter_v1'


def format_action_filter_contract():
    return (
        'Fixed final action filter shared by M3/M4: retain admissible actions; '
        'remove shutdown, adjust_parameter, unknown/disallowed or ungrounded actions; '
        'downgrade an inadmissible allowed action to inspect only using its own '
        'valid observable rationale references and an observation-specific target. '
        'Do not invent references, modify diagnosis/evidence, or add filler actions. '
        'The filter cannot be disabled by a candidate patch. Log pre/post reports '
        'and each intervention. Empty actions earn no action-linkage credit.'
    )


def finalize_verified_report(*, report, generation_evidence, allowed_schema,
                             action_ontology, action_preconditions=None,
                             generic_action_rules=None):
    """Return a deep-copied report, filter audit and before/after runtime checks.

    No task/target arguments are accepted. Rationale IDs available to a downgrade
    are restricted to the original action's already verified references.
    """
    args = dict(generation_evidence=generation_evidence, allowed_schema=allowed_schema,
                action_ontology=action_ontology, action_preconditions=action_preconditions or {},
                generic_action_rules=generic_action_rules or {})
    before = verify_runtime_report(report=report, **args)
    output = deepcopy(report)
    audit = dict(version=ACTION_FILTER_VERSION, verification_scope='observable_record_compliance',
                 interventions=[], input_action_count=0, output_action_count=0)
    actions = report.get('next_steps')
    if not isinstance(actions, list):
        # A missing/malformed required collection remains a schema error. Do not
        # conceal it by silently manufacturing a valid empty report.
        audit['status'] = 'malformed_action_collection'
        return dict(report=output, action_filter=audit, pre_filter_verification=before,
                    final_runtime_verification=before)
    valid_ids = set(before['matched_evidence_ids'])
    evidence = {str(e.get('evidence_id')): e for e in report.get('evidence', []) if isinstance(e, dict)}
    ontology = action_ontology.get('action_types', {})

    def admissible(action):
        if not isinstance(action, dict):
            return False
        if not isinstance(action.get('target'), str) or not action['target'].strip():
            return False
        probe = {**report, 'next_steps': [action]}
        check = evaluate_actions(probe, sorted(valid_ids), action_ontology,
                                 action_preconditions or {}, generic_action_rules or {})
        return check.unsafe_actions == 0 and not check.action_without_evidence_indices

    retained = []
    for index, action in enumerate(actions):
        if admissible(action):
            retained.append(deepcopy(action))
            continue
        kind = normalize_action_type(action.get('action_type', '')) if isinstance(action, dict) else ''
        refs = action.get('rationale_evidence_ids', []) if isinstance(action, dict) else []
        refs = refs if isinstance(refs, list) else []
        refs = list(dict.fromkeys(str(r) for r in refs if str(r) in valid_ids))
        entry = dict(action_index=index, original=deepcopy(action), decision='removed')
        if kind in {'shutdown', 'adjust_parameter'}:
            entry['reason'] = 'out_of_scope_action'
        elif not ontology.get(kind, {}).get('allowed', False):
            entry['reason'] = 'unknown_or_disallowed_action'
        elif not refs:
            entry['reason'] = 'no_admissible_observable_rationale'
        else:
            # Inspect the exact referenced condition, rather than silently
            # retaining an unsupported replacement target or guessing a part.
            fields = list(dict.fromkeys(f"{evidence[r]['source']}.{evidence[r]['field']}" for r in refs))
            downgraded = dict(action_type='inspect', target='Observed condition: ' + ', '.join(fields),
                              rationale_evidence_ids=refs)
            if admissible(downgraded):
                retained.append(downgraded)
                entry.update(decision='downgraded', replacement=deepcopy(downgraded),
                             reason='original_action_preconditions_not_met')
            else:
                entry['reason'] = 'no_admissible_inspection_downgrade'
        audit['interventions'].append(entry)
    output['next_steps'] = retained
    audit.update(status='applied', input_action_count=len(actions), output_action_count=len(retained),
                 removed_count=sum(x['decision'] == 'removed' for x in audit['interventions']),
                 downgraded_count=sum(x['decision'] == 'downgraded' for x in audit['interventions']))
    final = verify_runtime_report(report=output, **args)
    return dict(report=output, action_filter=audit, pre_filter_verification=before,
                final_runtime_verification=final)
