# Reproduction boundaries

The documented commands verify hashes, preserve all scheduled cases and
recompute saved-report aggregates. They do not create new model responses or
refit classifiers. The optional bootstrap mode uses the frozen estimator body,
seed, pairing, resampling count and hypothesis families.

Exported code retains the experiment algorithms. The portable adapter changes
package imports, filesystem destinations and materializes execution overrides;
the statistical function bodies and model-facing generation payloads remain
unchanged. `manifests/code_export.json` records source and exported hashes.

Artifacts are normalized exports: personal filesystem metadata is replaced by
opaque IDs. Source payload hashes and asset hashes are different concepts.
Metadata-only normalization never supplies a missing label, report, runtime,
response identifier or provider completion time.

AI4I and PdM each use only their own indexed task and experimental artifact
sources. Task manifests and all dataset columns are checked against those
bindings. AI4I input values, labels and fixed class quotas are verified against
the authoritative 300-row manifest.
