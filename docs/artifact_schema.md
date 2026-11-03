# Artifact schema

All bundles contain UTF-8 JSON Lines compressed with gzip. They are addressed
by dataset and file hash, not by a workstation path. Logical request IDs,
run/draw/candidate IDs and task IDs connect the records.

- `*-records.jsonl.gz`: S7 and S8 report allocations, CPU predictions, and
  same-label templates. Each row contains `dataset`, `kind`, `group`, and the
  saved `record`, including initial/final measurements and execution traces.
- `*-calls.jsonl.gz`: dataset-specific physical attempts, exact model-facing
  request content and provider response content when available, recorded
  usage, original content hashes and completion or terminal receipts.
- `*-usage.jsonl.gz`: the corresponding dataset-specific usage rows.
- `*-search.jsonl.gz`: candidate configurations, proposal feedback, search and
  validation records, screening/selection decisions, and search-job summaries.
- `*-analysis.jsonl.gz`: the complete saved scientific analysis for the dataset.

The `id` of a physical call distinguishes its stage from its recorded call
number. A logical allocation is not necessarily an independently generated
response. Shared fallback responses are counted once in physical expenditure.
Same-label template costs include parent generation and are not extra invoices.

Local paths in provenance metadata are replaced by opaque `artifact://`
identifiers. These identifiers are non-dereferenceable provenance labels;
actual bundle lookup uses the indices above. Model-facing request payloads,
generated response content, predictions, observations and measurement values
are preserved. Original and exported file hashes distinguish byte-level
identity from metadata normalization.

CSV cells containing Python-literal provenance lists are exported as JSON-encoded
lists. Path metadata is normalized while candidate IDs, proposal statuses and
patch signatures are preserved. Dataset-wide operational budget snapshots are
excluded; the indexed usage bundles provide dataset-specific accounting.

Paid incomplete responses retain their status, partial provider output and
usage. A terminal receipt is not a successful completion receipt. Partial text
does not supply a diagnosis. Missing predictions receive zero classification
credit; unavailable report scores remain null. An affected full population has
null J. No failed case is dropped or filled from another draw.
