"""One explicit inference boundary, matched to the frozen initial prompts.

The JSON payload is immutable. Identifiers are retained for joins/audits, never
as model features. Auxiliary summaries cannot acquire citable evidence IDs.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import pandas as pd

from experiments.data.generation_evidence import generation_task, visible_evidence, validate_observation_windows
from experiments.llm.observable_records import observation_records, json_value
from experiments.llm.prompts import format_proxy_evidence_table, format_azure_association_guide

NUMERIC_TASK_FIELDS = ('air_temperature', 'process_temperature', 'rotational_speed',
    'torque', 'tool_wear', 'temperature_gap', 'mechanical_power_w', 'overstrain_score', 'age')
REQUIRED_COPY_FIELDS = {'evidence_id','source','field','statistic','direction','value','window','role'}


def dump(value):
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


def visible_associations(guide):
    """Only fields actually printed by format_azure_association_guide, first 12.

    Parse the printed pattern key, rather than trusting unprinted source/field
    metadata in the guide file. Global class counts are NOT part of this view.
    """
    if not guide or guide.get('dataset') != 'Azure PdM':
        return []
    rows = []
    for item in (guide.get('top_associations') or [])[:12]:
        key = str(item.get('evidence_key', ''))
        parts = key.split(':')
        if len(parts) != 3:
            raise ValueError('Unparseable visible association key')
        row = {k: json_value(item.get(k)) for k in ('evidence_key','top_component',
            'n','top_probability','confidence','component_distribution')}
        row.update(source=parts[0], field=parts[1], pattern=parts[2])
        rows.append(row)
    return rows


@dataclass(frozen=True)
class ObservableView:
    payload: str

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    @property
    def task(self):
        return self.data['task']

    @property
    def records(self):
        return self.data['records']

    @property
    def associations(self):
        return self.data['associations']

    @property
    def labels(self):
        return self.data['candidate_labels']


def build_view(*, task, generation_evidence, allowed_schema, method='M3', patch=None, association_guide=None):
    """Caller passes the current task's records, never a proxy target table.

    Hidden columns are projected out even in adversarial tests. Cross-task rows,
    illegal source fields, malformed metadata and out-of-window rows fail closed.
    M3 currently explicitly passes max_rows=10 (not formatter default 8).
    """
    if method not in ('M3', 'M4') or (method == 'M4' and patch is None):
        raise ValueError('M3 or M4 with a frozen patch required')
    clean = json_value(generation_task(task))
    for name, value in clean.items():
        if name != 'azure_observation_context' and isinstance(value, (dict, list, bool)):
            raise ValueError('Task scalar required: ' + name)
    evidence = visible_evidence(generation_evidence)
    if not evidence.empty:
        if 'task_id' not in evidence or not evidence.task_id.astype(str).eq(str(clean['task_id'])).all():
            raise ValueError('Evidence belongs to a different task')
        if evidence.evidence_id.duplicated().any():
            raise ValueError('Ambiguous evidence IDs')
        for row in evidence.to_dict('records'):
            if row['field'] not in allowed_schema.get('allowed_sources', {}).get(row['source'], []):
                raise ValueError('Undeclared source/field')
            if any(isinstance(v, (dict, list, tuple)) for v in row.values()):
                raise ValueError('Evidence scalar required')
        validate_observation_windows(evidence, pd.DataFrame([clean]))
    columns = None
    if method == 'M4':
        from experiments.harness.m4_patched import get_m4_evidence_columns, format_m4_evidence_table
        columns = get_m4_evidence_columns(patch)
        evidence_text = format_m4_evidence_table(evidence, patch, max_rows=10)
    else:
        evidence_text = format_proxy_evidence_table(evidence, max_rows=10)
    # Parse the exact frozen formatter output; its column projection, including
    # omission of task_id, is authoritative. No richer parallel table is used.
    records = [] if evidence_text == 'None' else [json.loads(line) for line in evidence_text.splitlines()]
    if any(REQUIRED_COPY_FIELDS - set(row) for row in records):
        raise ValueError('Visible patch omits metadata needed for the common report contract')
    return ObservableView(dump(dict(version='nonllm_visible_v1', method=method,
        task=clean, records=records, candidate_labels=allowed_schema['candidate_labels'],
        associations=visible_associations(association_guide),
        evidence_text=evidence_text, association_text=format_azure_association_guide(association_guide),
        patch_sha256=hashlib.sha256(dump(patch).encode()).hexdigest() if patch is not None else None)))


def feature_dict(view, *, prior_prediction=None):
    """Only observable scalars, category indicators and relative time features.

    No task_id, evidence_id, split, UDI, serial, machineID or absolute timestamp.
    Prior scores are optional, explicitly supplied by a fold-trained artifact.
    """
    task = view.task
    result = {f'task.{k}': finite(task.get(k)) for k in NUMERIC_TASK_FIELDS}
    for k in ('type', 'model'):
        result[f'task.{k}'] = str(task.get(k) or '__missing__')
    for i, row in enumerate(view.records):
        # Position/metadata are prompt-visible; preserve all rows without hidden
        # source rereads or expanding a truncated evidence pool.
        root = f'row{i}.{row["source"]}.{row["field"]}'
        for k in ('value', 'raw_value', 'strength'):
            if k not in row:
                continue
            value = finite(row[k])
            result[root + '.' + k] = value if value is not None else str(row[k])
        for k in ('statistic', 'direction', 'window', 'role'):
            result[root + '.' + k] = str(row[k])
        if row.get('event_time') and task.get('failure_time'):
            result[root + '.age_hours'] = (pd.Timestamp(task['failure_time'])-pd.Timestamp(row['event_time'])).total_seconds()/3600
    context = task.get('azure_observation_context', {})
    for row in context.get('channels', []):
        for k, v in row.items():
            if k != 'field':
                result[f'aux.{row["field"]}.{k}'] = finite(v)
    for source in ('errors', 'maintenance'):
        # Absence is known only in a complete supplied auxiliary summary.
        if source not in context:
            continue
        names = [f'error{i}' for i in range(1,6)] if source == 'errors' else [f'comp{i}' for i in range(1,5)]
        rows = {r['value']: r for r in context[source]}
        for name in names:
            row = rows.get(name)
            result[f'aux.{source}.{name}.count'] = row['count'] if row else 0
            result[f'aux.{source}.{name}.last_age_hours'] = row['last_age_hours'] if row else None
    if prior_prediction is not None:
        for label in view.labels:
            result['prior.'+label] = float(prior_prediction.scores.get(label, 0))
        result['prior.coverage'] = float(prior_prediction.audit['matched_pattern_count'])
        result['prior.conflict'] = int(prior_prediction.audit['conflict'])
    return {k: float('nan') if v is None else v for k, v in result.items()}
