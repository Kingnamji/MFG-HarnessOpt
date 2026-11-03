# Fixed protocol

The recorded model snapshot is `gpt-5.4-mini-2026-03-17`. Reports use temperature
0 and 1,200 output tokens. Proposals use 4,096 output tokens; COPRO-style uses
temperature 1.4 and the other procedures use 0. SDK retries are 0. Campaign
execution used one API worker. Permitted report repair is at most one call.
Shared source defaults are materialized as these recorded execution settings
in the exported model configuration.

Search, validation and test contain 100/50/150 cases per dataset. Search is
the only fitting/optimization population. Validation selects configurations
and supplies frozen confidence mappings. Test labels and proxy annotations
are offline evaluation inputs and are not generation-visible features.
AI4I RULE uses public domain thresholds. PdM PRIOR and DT/GBT use search data.
DT/GBT use common three-fold cross-validation and twelve settings per model.

The AI4I guide and the PdM association guide are included as supplied to the
generator. The PdM association table has sixteen stored patterns; only the
first twelve are displayed by the shared formatter. Guide confidence tags are
distinct from the generated report's ordinal confidence and its frozen mapping.

Descriptive results average draw metrics for fixed systems and run/draw metrics
for optimized systems. Inference pools sufficient statistics. AI4I resamples
paired cases; PdM resamples paired machines, retaining all associated cases and
repeated draws. The bootstrap uses 10,000 resamples and seed 20261024.
The Holm family sizes are 2 primary, 2 secondary, 6 diagnostic and 14 ablation
contrasts across both datasets. Accuracy and Macro-F1 are also supplied as
supplementary outcomes without expanding the prespecified families.

The seven-term J is a fixed research objective. Its cost term uses an uncached
USD 0.005 reference and clipping. It is not calibrated operational utility.
Actual expenditure applies USD 0.75/0.075/4.50 per million uncached-input,
cached-input and output tokens. Uncached accounting prices all input at 0.75.
These are the recorded study prices, not a claim about current API pricing.
Paid failures and repaired reports retain all applicable charges exactly once.

Failure categories describe the frozen evaluator. F4/F6/F8/F10 were not
observed in the core systems; five known F4 flags occurred in the PdM
S1E0G0 ablation, with five additional reports unobserved. These observations
do not establish field safety or zero detector error. Software fixtures,
natural failures, repair transitions and filter interventions are different
evidence sources. No expert utility or deployment experiment is implied.
