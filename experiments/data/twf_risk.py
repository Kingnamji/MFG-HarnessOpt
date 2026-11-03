"""Public stochastic mechanism, not a fitted posterior or target-label rule."""
import math

TWF_RISK_VERSION = 'twf_public_risk_v1'
PUBLIC_SOURCE = 'https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset'


def assess_twf_risk(tool_wear):
    try:
        wear = float(tool_wear)
    except (TypeError, ValueError):
        wear = float('nan')
    if not math.isfinite(wear) or wear < 0:
        state = 'unavailable'
    elif wear < 200:
        state = 'below_public_event_interval'
    elif wear <= 240:
        state = 'within_public_event_interval'
    else:
        state = 'above_public_event_interval_do_not_extrapolate'
    return dict(version=TWF_RISK_VERSION, wear_minutes=wear if math.isfinite(wear) and wear >= 0 else None,
                public_event_interval_minutes=[200,240], state=state, task_failure_probability=None,
                deterministic_label_available=False)


def format_twf_guide():
    return '''Probabilistic TWF interpretation (public dataset mechanism):
- The public generator selects a tool replacement/failure time randomly in 200-240 minutes. Wear in this interval is positive qualitative TWF risk evidence; it is not an observed failure event. The snapshot does not reveal the selected event time or whether replacement versus failure occurred.
- Explicitly compare TWF with No Failure when wear is in this interval, including when HDF/PWF/OSF conditions are absent. Those negative conditions do not exclude TWF. TWF is a permissible best-supported hypothesis without a failure flag; if selected from wear alone, use low confidence and state that preventive replacement/non-failure remains possible. No Failure remains a permissible hypothesis but must acknowledge unresolved TWF risk rather than claim that other normal indicators prove no wear failure.
- A satisfied public HDF/PWF/OSF condition is direct evidence for that mode; elevated wear alone does not override it. Below the event interval, do not invent positive TWF support; above it or with missing wear, state the public mechanism's limitation instead of extrapolating a cutoff.
- There is no calibrated P(TWF | this snapshot). Do not use an event-level failure fraction, a wear percentile, the high direction tag, or an arbitrary 0.5 cutoff as a task probability. The source does not specify a complete conditional distribution or a uniform event-time law. Do not emit a numerical posterior or high certainty from wear alone.
- Keep the Paper single-label prediction and categorical confidence schema. The risk assessment is advisory: it does not force TWF/No Failure, trigger a deterministic label repair, change the label target or alter scoring.'''
