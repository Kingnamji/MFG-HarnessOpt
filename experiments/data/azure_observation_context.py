"""Label-free auxiliary context. Does not change the Paper evidence target pool.

All four channels are summarized before the prediction cutoff, without top-k
selection. This is task context, not new output evidence IDs or learned labels.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

CONTEXT_VERSION = 'azure_observations_v1'
CONTEXT_KEY = 'azure_observation_context'
CONTEXT_FILE = 'azure_observation_context.json'
CHANNELS = ('volt', 'rotate', 'pressure', 'vibration')
NUMERIC_FIELDS = ('baseline_n', 'window_n', 'baseline_mean', 'baseline_std',
                  'window_mean', 'window_min', 'window_max', 'mean_z', 'peak_abs_z',
                  'fraction_above_2z', 'fraction_below_minus2z', 'early_12h_mean_z',
                  'late_12h_mean_z')
TIME_FIELDS = ('baseline_start', 'baseline_end', 'window_start', 'window_end')


def anchor_digest(tasks):
    # Deliberately excludes split labels, gold, component IDs and failure flags.
    rows = [dict(task_id=str(t['task_id']), machineID=int(t['machineID']),
                 failure_time=pd.Timestamp(t['failure_time']).isoformat())
            for t in tasks[['task_id', 'machineID', 'failure_time']].to_dict('records')]
    return hashlib.sha256(json.dumps(sorted(rows, key=lambda t:t['task_id']),
                                    sort_keys=True).encode()).hexdigest()


def finite_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('Boolean is not an observable measurement')
    value = float(value)
    return value if math.isfinite(value) else None


def project_context(value, task):
    """Typed nested allowlist; unknown keys (including answers) never propagate."""
    if not isinstance(value, dict) or value.get('version') != CONTEXT_VERSION:
        raise ValueError('Missing/current Azure observation context required')
    cutoff = pd.Timestamp(task['failure_time'])
    times = {k:pd.Timestamp(value[k]) for k in TIME_FIELDS}
    if (any(pd.isna(t) for t in times.values()) or
        times['window_end'] != cutoff or times['window_start'] != cutoff-pd.Timedelta(hours=24) or
        times['baseline_start'] != cutoff-pd.Timedelta(days=30) or
        times['baseline_end'] not in (cutoff-pd.Timedelta(days=7), cutoff-pd.Timedelta(hours=24))):
        raise ValueError('Invalid pre-cutoff observation windows')
    rows = value.get('channels', [])
    if len(rows) != 4 or {r.get('field') for r in rows} != set(CHANNELS):
        raise ValueError('Exactly four unique Azure sensor summaries required')
    clean = dict(version=CONTEXT_VERSION, **{k:v.isoformat() for k,v in times.items()}, channels=[])
    for field in CHANNELS:
        row = next(r for r in rows if r['field'] == field)
        out = dict(field=field, **{k:finite_number(row.get(k)) for k in NUMERIC_FIELDS})
        for k in ('baseline_n', 'window_n'):
            if out[k] is None or out[k] < 0 or out[k] != int(out[k]):
                raise ValueError('Invalid observation count')
            out[k] = int(out[k])
        for k in ('fraction_above_2z', 'fraction_below_minus2z'):
            if out[k] is not None and not 0 <= out[k] <= 1:
                raise ValueError('Invalid exceedance fraction')
        clean['channels'].append(out)
    for source, horizon, names in [('errors', 24, {f'error{i}' for i in range(1,6)}),
                                    ('maintenance', 720, {f'comp{i}' for i in range(1,5)})]:
        clean[source] = []
        seen = set()
        for row in value.get(source, []):
            name = row.get('value')
            if name not in names or name in seen:
                raise ValueError('Unknown or duplicate event category')
            seen.add(name)
            count, age = finite_number(row.get('count')), finite_number(row.get('last_age_hours'))
            if count is None or count < 1 or int(count) != count or age is None or not 0 < age <= horizon:
                raise ValueError('Invalid pre-cutoff event summary')
            clean[source].append(dict(value=name, count=int(count), last_age_hours=age))
        clean[source].sort(key=lambda r:r['value'])
    return clean


def build_contexts(tasks, telemetry, errors, maintenance):
    """Raw values + timestamps only; the task manifest is explicitly projected."""
    telemetry = telemetry[['machineID', 'datetime', *CHANNELS]].copy()
    if telemetry.duplicated(['machineID', 'datetime']).any():
        raise ValueError('Duplicate telemetry timestamps cannot define sample fractions')
    streams = {'telemetry':telemetry, 'errors':errors[['machineID', 'datetime', 'errorID']],
               'maintenance':maintenance[['machineID', 'datetime', 'comp']]}
    indexed = {k:dict(tuple(v.groupby('machineID'))) for k,v in streams.items()}
    contexts = {}
    for task in tasks[['task_id', 'machineID', 'failure_time']].to_dict('records'):
        cutoff = pd.Timestamp(task['failure_time']); machine = task['machineID']
        start = cutoff-pd.Timedelta(hours=24)
        bstart, bend = cutoff-pd.Timedelta(days=30), cutoff-pd.Timedelta(days=7)
        data = indexed['telemetry'].get(machine, telemetry.iloc[:0])
        hist = data.loc[data.datetime.ge(bstart) & data.datetime.lt(bend)]
        if hist.empty:
            bend = start
            hist = data.loc[data.datetime.ge(bstart) & data.datetime.lt(bend)]
        window = data.loc[data.datetime.ge(start) & data.datetime.lt(cutoff)]
        context = dict(version=CONTEXT_VERSION, baseline_start=bstart.isoformat(), baseline_end=bend.isoformat(),
                       window_start=start.isoformat(), window_end=cutoff.isoformat(), channels=[])
        for field in CHANNELS:
            h = pd.to_numeric(hist[field], errors='coerce').replace([float('inf'), -float('inf')], float('nan')).dropna()
            w = pd.to_numeric(window[field], errors='coerce').replace([float('inf'), -float('inf')], float('nan')).dropna()
            mean = float(h.mean()) if len(h) else None
            std = float(h.std(ddof=0)) if len(h) >= 2 else None
            row = dict.fromkeys(NUMERIC_FIELDS)
            row.update(field=field, baseline_n=len(h), window_n=len(w), baseline_mean=mean, baseline_std=std)
            if len(w):
                row.update(window_mean=float(w.mean()), window_min=float(w.min()), window_max=float(w.max()))
            if len(w) and std is not None and std > 0:
                z = (w-mean)/std
                row.update(mean_z=float(z.mean()), peak_abs_z=float(z.abs().max()),
                           fraction_above_2z=float(z.ge(2).mean()), fraction_below_minus2z=float(z.le(-2).mean()))
                early = window.loc[w.index, 'datetime'].lt(cutoff-pd.Timedelta(hours=12))
                for key, values in [('early_12h_mean_z',z[early]), ('late_12h_mean_z',z[~early])]:
                    row[key] = float(values.mean()) if len(values) else None
            context['channels'].append(row)
        for source, field, hours in [('errors','errorID',24), ('maintenance','comp',720)]:
            data = indexed[source].get(machine, streams[source].iloc[:0])
            window = data.loc[data.datetime.ge(cutoff-pd.Timedelta(hours=hours)) & data.datetime.lt(cutoff)]
            context[source] = [dict(value=str(name), count=len(g),
                last_age_hours=(cutoff-g.datetime.max()).total_seconds()/3600)
                for name,g in window.groupby(field)]
        contexts[str(task['task_id'])] = project_context(context, task)
    return contexts


def attach_contexts(tasks, folder):
    """Fail closed on missing/stale sidecars; never silently use weaker inputs."""
    payload = json.loads((Path(folder)/CONTEXT_FILE).read_text(encoding='utf-8'))
    if payload.get('version') != CONTEXT_VERSION or payload.get('task_anchor_sha256') != anchor_digest(tasks):
        raise ValueError('Azure context version/task anchors changed; rebuild observable context')
    contexts = payload['tasks']
    if set(contexts) != set(tasks.task_id.astype(str)):
        raise ValueError('Azure summary task coverage mismatch')
    result = tasks.copy()
    result[CONTEXT_KEY] = [project_context(contexts[str(t['task_id'])],t) for t in tasks.to_dict('records')]
    return result


def format_context_contract():
    return '''Azure auxiliary observation context (shared input, not output evidence records):
- All four sensor channels are present without top-k/peak selection. mean_z is the window mean shift in baseline standard deviations; peak_abs_z is a single extreme and does not describe persistence. Fractions count valid samples beyond +2/-2 standard deviations; they are not failure probabilities. Compare early/late 12-hour mean_z for temporal change, not causal proof.
- Windows are half-open and strictly before the task cutoff. baseline_n/window_n report valid sample counts, not uptime. null standardized values mean insufficient/constant baseline or missing observations, never normality. Baseline fallback ends 24 hours before cutoff and may be less representative.
- Error/maintenance counts and last_age_hours are observed history, not component ground truth. Repeated or simultaneous error codes may be correlated; do not multiply association probabilities as independent likelihoods. Maintenance remains contextual. Search-derived associations are weak priors, not a deterministic sensor-to-component map.
- Weigh sustained changes, isolated peaks, recency and conflicting channels together. A missing channel from the primary evidence table is not a normal sensor. A conflict can remain unresolved; express low confidence and the competing explanation when appropriate.
- Auxiliary summaries can inform the diagnosis and limitations. They have no new citable evidence IDs: output evidence must copy existing diagnostic evidence-table records, never attach a new summary statistic to an old ID or invent an ID. Do not use auxiliary context alone to satisfy action prerequisites. Mention a missing citable channel as a limitation and request further evidence when needed.'''
