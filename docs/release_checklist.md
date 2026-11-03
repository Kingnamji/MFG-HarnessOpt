# Release record

Version 1.0.0 is publicly released on 2026-11-03. Apache-2.0 applies to the
original software; CITATION.cff records manuscript-order authors and the release
date. Git tag v1.0.0 remains fixed to commit
`a5a9fb243bc84dc28ffcaec288ddb93f10a3563e`. The ten original artifact bundles retain
their audited hashes. Publication documentation on main does not move that tag.
No permanent archive identifier is asserted.

## Fixed contents

- AI4I and PdM inputs have separate authoritative manifests and artifact indexes.
- Saved prompts, responses, observations and metrics retain their experimental
  values. Machine-specific provenance is replaced by opaque artifact identifiers.
- Raw responses, usage and terminal missing outcomes accompany successful reports.
- The repository contains no credentials, personal filesystem paths or private
  correspondence. Only reviewed, explicitly selected files enter version control.
- Large artifact bundles are release assets; `.gitignore` excludes generated
  outputs and local credential files from source commits.

## Reproduction checks

1. `python -m reproduce.verify` checks source/input hashes and partition integrity.
2. `python -m tests.check_detectors AI4I` and `python -m tests.check_detectors PdM`
   check the controlled software fixtures without constructing an API client.
3. `python -m reproduce.results --bootstrap` checks archived allocation metrics
   and paired comparisons using the fixed resampling procedure.
4. Compare generated tables to the saved reference metrics, and verify all release
   asset hashes before citing a release.

Use a fixed release tag and commit identifier in a publication citation. Changes
after a release should produce a new version with an explicit change log.
