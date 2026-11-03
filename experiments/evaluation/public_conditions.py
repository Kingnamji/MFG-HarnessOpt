"""Narrow public-rule consistency checks, independent of task answers.

This is a runtime contract diagnostic, not Paper F1/F3 or a label predictor.
Unknown/missing observations, TWF risk and Azure priors are never converted
into deterministic label targets. Reports are not mutated.
"""
import math
from experiments.data.generation_evidence import visible_evidence
from experiments.data.twf_risk import assess_twf_risk

PUBLIC_CONDITION_VERSION = 'public_conditions_v2_twf_advisory'


def check_public_conditions(report, generation_evidence, allowed_schema):
    result = dict(version=PUBLIC_CONDITION_VERSION, applicable=False,
                  checks=[], errors=[])
    if allowed_schema.get('dataset') != 'AI4I 2020':
        return result
    result['applicable'] = True
    rows = visible_evidence(generation_evidence).to_dict('records')

    def observed(field, source, numeric=True):
        values, ids = [], []
        for r in rows:
            if r.get('field') != field or r.get('source') != source:
                continue
            try:
                value = float(r['value']) if numeric else str(r['value'])
            except (KeyError, TypeError, ValueError):
                return None, []
            if numeric and not math.isfinite(value):
                return None, []
            values.append(value); ids.append(str(r['evidence_id']))
        # Ambiguous observations cannot establish a deterministic contradiction.
        if not values or len(set(values)) != 1:
            return None, []
        return values[0], ids

    gap, gi = observed('temperature_gap', 'derived_features')
    speed, si = observed('rotational_speed', 'machine_features')
    power, pi = observed('mechanical_power_w', 'derived_features')
    strain, oi = observed('overstrain_score', 'derived_features')
    variant, ti = observed('type', 'machine_features', numeric=False)
    wear, wi = observed('tool_wear', 'machine_features')
    result['probabilistic_risks'] = [dict(label='TWF', evidence_ids=wi, **assess_twf_risk(wear))]
    limit = {'L':11000, 'M':12000, 'H':13000}.get(variant)
    checks = [
        dict(label='HDF', condition='temperature_gap < 8.6 and rotational_speed < 1380',
             observed_values=dict(temperature_gap=gap, rotational_speed=speed),
             evidence_ids=gi+si, satisfied=None if gap is None or speed is None else gap < 8.6 and speed < 1380),
        dict(label='PWF', condition='mechanical_power_w < 3500 or mechanical_power_w > 9000',
             observed_values=dict(mechanical_power_w=power), evidence_ids=pi,
             satisfied=None if power is None else power < 3500 or power > 9000),
        dict(label='OSF', condition='overstrain_score > type threshold (L=11000, M=12000, H=13000)',
             observed_values=dict(overstrain_score=strain, type=variant, threshold=limit),
             evidence_ids=oi+ti, satisfied=None if strain is None or limit is None else strain > limit),
    ]
    result['checks'] = checks
    label = report.get('predicted_label') if isinstance(report, dict) else None
    if label == 'No Failure' and any(c['satisfied'] is True for c in checks):
        result['errors'].append(dict(code='normal_label_with_satisfied_public_condition',
            predicted_label=label, instruction='Reconcile the diagnosis with the satisfied public conditions; absence of a hidden failure flag is not contrary evidence. Express uncertainty in confidence/limitations.'))
    elif any(c['label'] == label and c['satisfied'] is False for c in checks):
        result['errors'].append(dict(code='label_contradicts_its_public_condition',
            predicted_label=label, instruction='The selected label contradicts its observable public condition. Reconsider the available candidates without inventing evidence or a new threshold.'))
    return result
