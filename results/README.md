# Numerical results

`main_performance.csv` contains draw means for fixed M0–M3 systems and run/draw
allocation means for M4, Metric-only and COPRO-style. `paired_comparisons.csv`
contains the pooled paired estimators, which need not equal differences of
allocation means for nonlinear outcomes. These are different estimands.

The prespecified Holm families contain 2 primary J comparisons, 2 secondary J
comparisons, 6 diagnostic-accuracy comparisons and 14 ablation-accuracy
comparisons across the datasets. Unadjusted supplementary outcomes have a blank
`p_Holm` field; they are not assigned the correction from another outcome.
Null J values denote unavailable full-population objectives, not zero effects.

`diagnostic_controls.csv` reports RULE, PRIOR, DT and GBT under the recorded
observable views. AI4I RULE implements published domain conditions; PdM PRIOR
and DT/GBT derive their fitted quantities from search data. Their diagnostic
accuracy is relevant even when M4 improves the composite reporting objective.
Neither the composite objective nor proxy evidence/action scores establish
physical causality, expert usefulness or safe field operation.

The full analysis bundles include trajectory, sensitivity, economics,
calibration, subgroup, action, failure and qualitative-case records. The default
reproduction command verifies nine core allocation metric families; the optional
bootstrap command also checks all 24 prespecified comparisons. Other archived
analysis families are supplied as saved outputs, not claimed to be recalculated
by that command.

Each dataset has 150 unique test cases. Repeated reports do not increase the
independent case count; PdM inference retains 45 test-machine clusters.
