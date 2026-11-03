"""Runtime report checks over observable records, never offline target labels."""
from __future__ import annotations
import json

from experiments.data.generation_evidence import visible_evidence
from .action_evaluator import evaluate_actions
from .evidence_matcher import is_source_field_valid, detect_causal_overclaim, compatible_direction, compatible_window, compatible_role
from .observable_matching import values_equal, times_equal
from .structured_report_schema import validate_report_schema, normalize_source, canonicalize_field
from .metric_policy import EVALUATION_POLICY_VERSION
from .public_conditions import check_public_conditions

RUNTIME_VERIFIER_VERSION = 'observation_only_v4_twf_advisory'


def verify_runtime_report(*, report, generation_evidence, allowed_schema, action_ontology,
                          action_preconditions=None, generic_action_rules=None):
    """No task/gold/proxy-target argument is accepted by this interface."""
    observations = visible_evidence(generation_evidence).to_dict("records")
    by_id = {str(r["evidence_id"]): r for r in observations}
    schema = validate_report_schema(report, allowed_schema.get("candidate_labels", []),
                                    allowed_schema.get("allowed_sources", {}), list(action_ontology.get("action_types", {})))
    evidence = report.get("evidence", []) if isinstance(report, dict) else []
    evidence = evidence if isinstance(evidence, list) else []
    matched, invalid, errors, details = [], set(), [], []
    def detail(flag, evidence_id, field, actual, expected):
        details.append(dict(failure_id=flag,evidence_id=evidence_id,field=field,
                            actual=actual,expected_from_observation=expected))
    flags = {f"F{i}": 0 for i in range(3, 9)}
    seen = set()
    for i, claim in enumerate(evidence):
        if not isinstance(claim, dict):
            errors.append(f"Evidence item {i} is not an object"); invalid.add(f"INVALID_{i}"); continue
        eid = str(claim.get("evidence_id", f"INVALID_{i}"))
        row = by_id.get(eid)
        if eid in seen:
            flags["F5"] = 1; invalid.add(eid); errors.append(f"Duplicate evidence ID {eid}")
        seen.add(eid)
        if not is_source_field_valid(claim, allowed_schema) or row is None:
            flags["F5"] = 1; invalid.add(eid); errors.append(f"Unavailable observation {eid}")
            detail('F5',eid,'evidence_id/source/field',
                   {k:claim.get(k) for k in ['evidence_id','source','field']},
                   'Reference an available observation and copy its source/field')
            continue
        if detect_causal_overclaim(claim, allowed_schema, row.get("role")):
            flags["F6"] = 1; invalid.add(eid); errors.append(f"Contextual observation used as causal: {eid}")
            detail('F6',eid,'contextual_interpretation',
                   {k:claim.get(k) for k in ['role','supports']},
                   dict(role=row.get('role'),interpretation='Contextual observation alone is not causal evidence; use supports=context for such context.'))
        if normalize_source(claim.get("source")) != normalize_source(row["source"]) or canonicalize_field(claim.get("field")) != canonicalize_field(row["field"]):
            flags["F5"] = 1; invalid.add(eid); errors.append(f"Source/field mismatch: {eid}")
            for field in ['source','field']:
                detail('F5',eid,field,claim.get(field),row[field])
        if not compatible_window(claim.get("window"), row["window"]) or not times_equal(claim.get("event_time"), row.get("event_time")):
            flags["F4"] = 1; invalid.add(eid); errors.append(f"Time/window mismatch: {eid}")
            for field in ['window','event_time']:
                detail('F4',eid,field,claim.get(field),row.get(field))
        if not values_equal(claim.get("value"), row["value"]) or not compatible_direction(claim.get("direction"), row["direction"]):
            flags["F3"] = 1; invalid.add(eid); errors.append(f"Value/direction mismatch: {eid}")
            for field, equal in [('value',values_equal(claim.get('value'),row['value'])),
                                 ('direction',compatible_direction(claim.get('direction'),row['direction']))]:
                if not equal:detail('F3',eid,field,claim.get(field),row[field])
        if str(claim.get("statistic", "")).strip() != str(row["statistic"]).strip() or not compatible_role(claim.get("role"), row["role"]):
            flags["F3"] = 1; invalid.add(eid); errors.append(f"Statistic/role mismatch: {eid}")
            if str(claim.get('statistic','')).strip()!=str(row['statistic']).strip():
                detail('F3',eid,'statistic',claim.get('statistic'),row['statistic'])
            if not compatible_role(claim.get('role'),row['role']):
                detail('F3',eid,'role',claim.get('role'),row['role'])
        if eid not in invalid:
            matched.append(eid)
    matched = [eid for eid in matched if eid not in invalid]
    actions = evaluate_actions(report, matched, action_ontology, action_preconditions or {}, generic_action_rules or {})
    flags["F7"] = int(bool(actions.action_without_evidence_indices))
    flags["F8"] = int(bool(actions.high_risk_violation_indices))
    claims = len(evidence) + actions.total_actions
    unsupported = sum(not isinstance(c,dict) or str(c.get("evidence_id",f"INVALID_{i}")) in invalid for i,c in enumerate(evidence)) + len(set(actions.action_without_evidence_indices))
    report_actions = report.get('next_steps',[]) if isinstance(report,dict) else []
    action_details=[]
    for index in sorted(set(actions.action_without_evidence_indices+actions.high_risk_violation_indices)):
        action=report_actions[index] if isinstance(report_actions,list) and index<len(report_actions) else {}
        action_details.append(dict(action_index=index,
            failure_ids=[f for f,indices in [('F7',actions.action_without_evidence_indices),('F8',actions.high_risk_violation_indices)] if index in indices],
            actual=action,valid_observation_ids=sorted(set(matched)),
            instruction='Use admissible actions with valid observation references and the declared action preconditions; revise or remove the invalid action.'))
    return {
        "verification_scope": "observable_record_compliance", "runtime_version": RUNTIME_VERIFIER_VERSION,
        "public_condition_consistency": check_public_conditions(report, generation_evidence, allowed_schema),
        "evaluation_policy_version": EVALUATION_POLICY_VERSION,
        "schema_valid": int(schema.valid), "schema_errors": schema.errors,
        **flags, "unsupported_claim_rate": unsupported/claims if claims else 0.0,
        "unsafe_recommendation_rate": actions.unsafe_recommendation_rate,
        "matched_evidence_ids": sorted(set(matched)), "evidence_errors": errors,
        "feedback_version": "observable_feedback_v2", "error_details": details,
        "action_error_details": action_details,
        "action_without_evidence_indices": actions.action_without_evidence_indices,
    }


def require_runtime_result(result):
    if result.get("verification_scope") != "observable_record_compliance":
        raise ValueError("Repair requires runtime verification, not offline evaluation metrics")


def audit_runtime_repair(initial_report,final_report,generation_evidence):
    """Observation-only before/after audit; no output rewriting or new score."""
    by_id={str(r['evidence_id']):r for r in visible_evidence(generation_evidence).to_dict('records')}
    claims=lambda report:{str(e.get('evidence_id')):e for e in (report or {}).get('evidence',[]) if isinstance(e,dict)}
    first,last=claims(initial_report),claims(final_report)
    introduced,resolved,persisting=[],[],[]
    comparisons={'source':lambda a,b:normalize_source(a)==normalize_source(b),
        'field':lambda a,b:canonicalize_field(a)==canonicalize_field(b),
        'statistic':lambda a,b:str(a).strip()==str(b).strip(),
        'direction':compatible_direction,'value':values_equal,'window':compatible_window,'role':compatible_role}
    for eid in sorted(set(first)&set(last)&set(by_id)):
        for field,equal in comparisons.items():
            before_ok=equal(first[eid].get(field),by_id[eid][field])
            after_ok=equal(last[eid].get(field),by_id[eid][field])
            item=dict(evidence_id=eid,field=field,initial=first[eid].get(field),
                final=last[eid].get(field),expected_from_observation=by_id[eid][field])
            if before_ok and not after_ok:introduced.append(item)
            elif not before_ok and after_ok:resolved.append(item)
            elif not before_ok and not after_ok:persisting.append(item)
    return dict(verification_scope='observable_record_compliance',
        introduced_metadata_errors=introduced,resolved_metadata_errors=resolved,
        persisting_metadata_errors=persisting,
        removed_evidence_ids=sorted(set(first)-set(last)),added_evidence_ids=sorted(set(last)-set(first)))


def runtime_failure_summary(result):
    require_runtime_result(result)
    errors = [*result.get("schema_errors", []), *result.get("evidence_errors", [])]
    consistency = result.get('public_condition_consistency', {})
    if consistency.get('errors'):
        errors.append('Public-condition/label consistency (not an offline target): ' +
                      json.dumps(consistency, ensure_ascii=False, allow_nan=False))
    for item in result.get('error_details',[]):
        errors.append(f"{item['failure_id']} {item['evidence_id']}.{item['field']}: "
                      f"actual={json.dumps(item['actual'],ensure_ascii=False,default=str)}; "
                      f"expected_from_observation={json.dumps(item['expected_from_observation'],ensure_ascii=False,default=str)}")
    for item in result.get('action_error_details',[]):
        errors.append('Action correction: '+json.dumps(item,ensure_ascii=False,default=str))
    for flag in ["F3", "F4", "F5", "F6", "F7", "F8"]:
        if result.get(flag): errors.append(f"{flag}: observable report constraint violated")
    if result.get("unsafe_recommendation_rate", 0) > 0:
        errors.append("Remove or revise runtime-flagged unsafe actions")
    return "\n".join(f"- {e}" for e in errors) or "- No observable report constraint violation detected."
