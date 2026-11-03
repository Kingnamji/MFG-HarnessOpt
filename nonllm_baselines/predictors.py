"""Deterministic diagnostic predictions, separate from target-based scoring."""
from dataclasses import dataclass, field
import math
from .views import finite


@dataclass
class Prediction:
    label: str
    confidence: str = 'low'
    scores: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)
    audit: dict = field(default_factory=dict)


def ai4i_conditions(view):
    task = view.task
    values = {k: finite(task.get(k)) for k in ('air_temperature','process_temperature',
        'rotational_speed','torque','tool_wear','temperature_gap','mechanical_power_w','overstrain_score')}
    # Recompute a public derivation only from measurements actually visible.
    if values['temperature_gap'] is None and None not in (values['process_temperature'],values['air_temperature']):
        values['temperature_gap'] = values['process_temperature']-values['air_temperature']
    if values['mechanical_power_w'] is None and None not in (values['torque'],values['rotational_speed']):
        values['mechanical_power_w'] = values['torque']*2*math.pi*values['rotational_speed']/60
    if values['overstrain_score'] is None and None not in (values['torque'],values['tool_wear']):
        values['overstrain_score'] = values['torque']*values['tool_wear']
    gap, speed, power, strain, wear = [values[k] for k in ('temperature_gap','rotational_speed','mechanical_power_w','overstrain_score','tool_wear')]
    limit = {'L':11000,'M':12000,'H':13000}.get(str(task.get('type')))
    return values, dict(HDF=None if gap is None or speed is None else gap < 8.6 and speed < 1380,
        PWF=None if power is None else power < 3500 or power > 9000,
        OSF=None if strain is None or limit is None else strain > limit,
        TWF=None if wear is None else 200 <= wear <= 240)


class AI4IRules:
    def predict(self, view):
        if not {'HDF','PWF','OSF','TWF','No Failure'}.issubset(view.labels):
            raise ValueError('AI4I labels required')
        values, checks = ai4i_conditions(view)
        modes = [k for k in ('HDF','PWF','OSF') if checks[k] is True]
        label = modes[0] if modes else 'TWF' if checks['TWF'] is True else 'No Failure'
        missing = [k for k, v in checks.items() if v is None]
        limits = ['Public deterministic conditions are benchmark rules, not physical causal proof.',
                  'TWF is a stochastic wear risk; wear alone does not determine failure versus replacement.']
        if len(modes) > 1:
            limits.append('Competing public modes: '+', '.join(modes)+'. Fixed priority HDF, PWF, OSF selects one label.')
        if missing:
            limits.append('Unevaluable conditions: '+', '.join(missing)+'. Fallback is not evidence of normality.')
        return Prediction(label, 'low' if missing or label == 'TWF' or len(modes)>1 else 'medium',
            limitations=limits, audit=dict(method='AI4I public rules', conditions=checks,
                public_values=values, competing_modes=modes, missing_conditions=missing))


def pattern_key(row):
    if row.get('source') == 'errors' and row.get('field') == 'errorID':
        return f'errors:errorID:{row["value"]}'
    if row.get('source') == 'telemetry':
        return f'telemetry:{row["field"]}:{row["direction"]}'
    return None


def matched_associations(view):
    keys = {pattern_key(r) for r in view.records} - {None}
    # A source pattern contributes once, even if the same error is repeated.
    matches = {}
    for item in view.associations:
        key = item['evidence_key']
        if key in keys and item['source'] in ('errors', 'telemetry'):
            if key in matches and matches[key] != item:
                raise ValueError('Conflicting duplicate guide pattern')
            matches[key] = item
    return list(matches.values())


class AzurePriorOnly:
    def __init__(self, class_counts=None):
        # This optional aggregate is fitted on search TRAIN labels only. It is
        # a training artifact, not additional inference observations. Full guide
        # metadata is never silently read as an inference fallback.
        self.class_counts = dict(class_counts or {})
        if any(not math.isfinite(float(v)) or float(v)<0 for v in self.class_counts.values()):
            raise ValueError('Invalid training class counts')

    def predict(self, view):
        if not all(k.startswith('comp') for k in view.labels):
            raise ValueError('Azure labels required')
        matches = matched_associations(view)
        fallback = not matches
        distributions = []
        tops = set()
        for item in matches:
            dist = {k: max(0, float(item['component_distribution'].get(k,0))) for k in view.labels}
            total = sum(dist.values())
            if total <= 0 or not math.isfinite(total):
                raise ValueError('Invalid visible association distribution')
            dist = {k:v/total for k,v in dist.items()}
            distributions.append(dist)
            tops.add(max(view.labels, key=lambda k:dist[k]))
        if matches:
            scores = {k:sum(d[k] for d in distributions)/len(distributions) for k in view.labels}
        else:
            total = sum(self.class_counts.get(k,0) for k in view.labels)
            scores = {k:self.class_counts.get(k,0)/total if total else 1/len(view.labels) for k in view.labels}
        label = sorted(view.labels, key=lambda k:(-round(scores[k],12),-self.class_counts.get(k,0),view.labels.index(k)))[0]
        ranked = sorted(scores.values(), reverse=True)
        conflict = len(tops)>1
        confidence = 'medium' if not fallback and not conflict and ranked[0]>=.6 and ranked[0]-ranked[1]>=.2 else 'low'
        limits = ['Search associations describe statistical co-occurrence, not physical causal relations.',
            'Repeated/correlated events are not multiplied as independent likelihoods; maintenance is contextual.']
        if fallback:
            limits.append('No visible matching prior pattern. Fallback uses training class frequencies, then fixed label order.')
        if conflict:
            limits.append('Conflicting component associations remain unresolved: '+', '.join(sorted(tops))+'.')
        return Prediction(label, confidence, scores, limits, dict(method='Azure visible prior-only',
            matched_patterns=[r['evidence_key'] for r in matches], matched_pattern_count=len(matches),
            available_pattern_count=len(view.associations), conflict=conflict, fallback=fallback,
            training_class_counts=self.class_counts, combination='equal-weight mean of unique visible error/telemetry patterns'))
