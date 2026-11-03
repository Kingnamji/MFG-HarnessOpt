# MFG-HarnessOpt

Manufacturing diagnostic harness optimization and offline result reproduction
for AI4I and Azure PdM. M4 optimizes a fixed-model reporting workflow using
typed failure feedback. The package also includes Metric-only, COPRO-style,
fixed M0–M3 systems, diagnostic controls, and source/evidence/guide ablations.

## Scope

Each dataset has 100 search, 50 validation, and 150 test cases. Optimized
procedures have three runs and three test draws. Repeated draws are not
independent cases; PdM inference preserves 45 test-machine clusters.
AI4I means the single fixed partition distributed in `Dataset/Preprocessed/AI4I`.
Those manifests are authoritative throughout this repository.

M4 improves the specified composite objective relative to the compared
optimizers. RULE/PRIOR and trained tabular controls can achieve higher
diagnostic accuracy. Proxy report scores are not measurements of maintenance
utility, physical causality, or deployment safety.

## Reproduce from saved outputs

1. Use Python 3.12 and install the dependencies with
   `python -m pip install -r environment/requirements.txt`.
2. Place the artifact files listed in `artifacts/index.json` in
   `artifacts/bundles/`. All files are verified against their SHA-256 hashes.
3. Run `python -m reproduce.verify` to check the package and fixed partitions.
4. Run `python -m reproduce.results` to verify the core saved allocation metrics and
   reconstruct the main performance table.
5. Run `python -m reproduce.results --bootstrap` to additionally recompute the
   prespecified paired comparisons with 10,000 bootstrap samples.

These commands disable network access. They do not call an API, train a model,
select another harness, or generate another test report. Outputs are written
to `outputs/reproduced/`.

The code includes the model interface and optimization components for
inspection. A turnkey paid campaign launcher is not part of this release.
The documented reproduction commands consume archived observations and
outputs; a future API response is not guaranteed to match a stored response.

## Find the evidence

| Material | Location |
|---|---|
| Task identities and partitions | `Dataset/Preprocessed/`, `manifests/` |
| Prompts and observable input construction | `experiments/llm/`, `experiments/data/` |
| Runtime verification and offline evaluation | `experiments/evaluation/`, `experiments/harness/` |
| Optimizer and candidate handling | `experiments/optimization/`, `comparison_protocol/` |
| Complete diagnostic guides | `guides/` |
| Frozen selections and confidence maps | `harness_configs/` |
| RULE, PRIOR, DT, GBT and deterministic reports | `nonllm_baselines/` |
| Statistics and missing-outcome adapter | `comparison_protocol/`, `analysis/` |
| Reports, requests, responses, traces, search and usage | Artifact bundles indexed in `artifacts/index.json` |
| Reported numerical analysis | `results/` and the analysis bundles |

See `docs/artifact_schema.md` for bundle contents and `docs/protocol.md` for
the evaluation, inference and accounting conventions.

Run `python -m tests.check_detectors AI4I` and
`python -m tests.check_detectors PdM` for controlled evaluator and repair-flow
checks. These hand-specified software cases are distinct from empirical test
reports and do not estimate field-safety or expert-validation accuracy.

## License and citation

The original software and its original software configuration files are
licensed under Apache-2.0; see `LICENSE` and `NOTICE`. Third-party data and
source-derived materials retain their respective terms and attribution
requirements in `THIRD_PARTY_NOTICES.md` and `Dataset/README.md`.

`CITATION.cff` lists the software authors in the manuscript author order.
The fixed software version is **1.0.0**, identified by Git tag **v1.0.0**.
The tag resolves to the exact source commit; cite that tag and its full commit
identifier rather than the moving `main` branch. The release notes also record
the full commit identifier. The tag must not be moved; later changes require
a new version. Citation metadata do not require publishing author email addresses.

## Access status

Version 1.0.0 was fixed on 2026-11-03. The repository remains private and the
associated artifact release remains an unpublished draft. Access to the
materials for peer review may be requested from the corresponding author.
No public-release date or permanent archive identifier is asserted.
