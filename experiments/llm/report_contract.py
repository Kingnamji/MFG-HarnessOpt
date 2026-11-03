"""Shared observable-report contract, independent of any task's targets."""

REPORT_CONTRACT_VERSION = 'diagnostic_contract_v6_azure_twf'


def format_record_contract():
    return '''Observable record contract (diagnostic_contract_v6_azure_twf):
- Copy evidence_id, source, field, statistic, direction, value, window, and role from the referenced observable record. Use event_time for temporal checks; only emit keys permitted by the output schema.
- Fields are explicitly named in each JSON observation. Copy by field name, never by column position. statistic is the observation type; direction is the supplied categorical tag; value is the measurement.
- direction is source-owned metadata, independent of the source name. Preserve normal as normal and observed as observed only when that exact record supplies it. Do not replace normal with observed for a numeric machine_features record or put its measurement in direction.
- role is source-owned metadata. Preserve symptomatic/contextual/exclusionary exactly as supplied. A normal symptomatic observation must not be relabeled exclusionary merely because it excludes another fault.
- Infer supports from the observations and candidate labels; use exactly one candidate label or the literal context. Never obtain hidden support annotations or a task answer.
- supports is a label-valued field, not an explanation: do not append semicolons, exclusions, negation, multiple labels, or diagnostic prose. Put explanations and exclusions in limitations; express uncertainty through confidence and limitations.
- A normal observation may support No Failure when justified by the supplied public rules; it is not positive support for the fault it excludes. Normal measurements for one fault do not establish that every other fault is absent. Use context when no candidate-specific support is asserted.
- Preserve the source-owned role even for normal/exclusion reasoning. Do not change symptomatic to exclusionary to express a diagnostic interpretation.'''


def format_diagnostic_contract():
    return '''Diagnosis preservation contract:
- Apply the supplied dataset label decision guide independently for each fault. Do not require all feature families to be abnormal.
- Normal indicators for other fault modes do not negate a satisfied label-specific condition. Use the observations required for the particular candidate label.
- Uncertainty is not evidence of No Failure. Select the best-supported candidate label and express justified uncertainty through confidence and limitations.
- Conversely, the existence of a measurement is not evidence of a fault. No Failure is a valid candidate when the available observations do not support a failure condition; do not force a fault merely to avoid an uncertain normal diagnosis.
- For AI4I, preserve the supplied PWF, HDF, and variant-specific OSF rules. TWF wear is a risk indicator, not a deterministic proof; do not invent a new wear threshold.
- For AI4I, compare measured values numerically with the stated thresholds before choosing a label. Normal-range power does not support PWF, and a below-threshold overstrain product does not support OSF, even if another risk indicator is high. Check that the selected label agrees with the rule comparisons described in supports and limitations.
- Keep recommendations conservative without automatically converting the diagnosis to No Failure.'''


def format_common_contract():
    return format_record_contract() + '\n\n' + format_diagnostic_contract()
