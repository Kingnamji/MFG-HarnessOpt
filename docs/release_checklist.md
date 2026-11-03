# Release preparation

The preparation repository is private. The source-code license, final citation
metadata and public release status must be settled before publication.

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
