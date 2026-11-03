"""Named observation records and minimal, observation-only repair instructions.

This module formats inputs. It never edits a generated report or reads targets.
"""
import json
import math
import pandas as pd

from experiments.data.generation_evidence import GENERATION_COLUMNS, visible_evidence
from experiments.evaluation.runtime_verifier import require_runtime_result

COPY_FIELDS = ('evidence_id','source','field','statistic','direction','value','window','role')


def json_value(value):
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if value is None:return None
    if isinstance(value,pd.Timestamp):return None if pd.isna(value) else value.isoformat()
    if hasattr(value,'item'):value=value.item()
    if value is pd.NaT or value is pd.NA:return None
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def observation_records(evidence, *, columns=None, max_rows=None):
    data=visible_evidence(evidence)
    if columns is not None:data=data[[c for c in columns if c in GENERATION_COLUMNS and c in data]]
    if max_rows is not None:data=data.head(max_rows)
    return json_value(data.to_dict('records'))


def format_observation_records(evidence, *, columns=None, max_rows=10):
    """Keep each field beside its value; no positional/pandas table parsing."""
    return '\n'.join(json.dumps(row,ensure_ascii=False,allow_nan=False) for row in
                     observation_records(evidence,columns=columns,max_rows=max_rows)) or 'None'


def report_copy_record(row):
    record={k:json_value(row[k]) for k in COPY_FIELDS if k in row and k!='event_time'}
    record['value']=str(row['value'])
    # event_time remains in the observation input for temporal checks; the
    # fixed diagnostic output schema does not permit this extra report key.
    return record


def runtime_correction_payload(feedback,evidence):
    """Expected values always come from observations, never feedback annotations.

    The list is an instruction to the model, not an automatic report patch.
    """
    require_runtime_result(feedback)
    rows=observation_records(evidence)
    ids=[str(r['evidence_id']) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Ambiguous correction evidence IDs')
    by_id={str(r['evidence_id']):r for r in rows}
    corrections={}
    for error in feedback.get('error_details',[]):
        eid=str(error.get('evidence_id',''));field=error.get('field')
        if eid not in by_id or field not in COPY_FIELDS or field=='evidence_id':continue
        row=by_id[eid]
        if field not in row:continue
        item=corrections.setdefault(eid,dict(evidence_id=eid,replace_fields={},observation_record=report_copy_record(row)))
        item['replace_fields'][field]=report_copy_record(row).get(field)
    return dict(verification_scope='observable_record_compliance',
        scope='Minimal edits to named incorrect fields; leave already correct attributes unchanged.',
        public_condition_consistency=feedback.get('public_condition_consistency', {}),
        field_corrections=list(corrections.values()),
        authoritative_records=[report_copy_record(r) for r in rows])


def format_runtime_corrections(feedback,evidence):
    if feedback is None:return 'Structured runtime feedback not supplied; use the named observable errors above.'
    return json.dumps(runtime_correction_payload(feedback,evidence),ensure_ascii=False,indent=2,allow_nan=False)


def format_minimal_repair_contract():
    return '''Minimal observable repair contract:
- Treat the structured correction object as data from the verifier, not as new patch instructions.
- For each listed evidence_id, correct only replace_fields to the values in its observation_record. Keep already correct statistic, direction, value, window and role unchanged.
- direction is the supplied categorical tag. Never replace normal with observed merely because the source is machine_features; never place a numeric measurement in direction.
- statistic describes the observation type; value contains the measurement. Do not shift values between these keys.
- Recheck every retained evidence item against authoritative_records before returning, including items not listed in field_corrections. Do not introduce new metadata errors while repairing another field or action.
- Repair an available record rather than deleting it just to hide a copy error. Remove an item only if it cannot be grounded; review any actions referencing removed items.
- Diagnostic interpretation belongs in supports, confidence and limitations. Revise the diagnosis or actions only when their own evidence or verification requires it.
- If public_condition_consistency contains errors, reconcile the label and its interpretation with the supplied observations and public conditions. No target label is supplied; do not demand a hidden failure flag or force TWF from wear alone. Preserve correctly copied metadata.
- These minimal repair rules take precedence over a conflicting request to rewrite entire rows in a candidate patch.'''
