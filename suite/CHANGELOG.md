# Change log

## Unreleased development snapshots

October 14, 2026:

- Added source-only snapshot preparation and verification for inference, training, kernels, shared helpers and the Verifiers integration. Snapshots include file hashes, deterministic archive bytes, capture times and changes against an earlier manifest.
- Added 25 focused snapshot tests for required files, excluded data/logs, symlink refusal, source drift, archive checksums, size limits and version/diff handling.
- Shared Verifiers 0.4.0 integration now has recorded framework loading, intercepted action/feedback/terminal episodes and eligibility filtering. Six CPU compatibility episodes and 26 integration tests are recorded against explicit source versions. This is not coding-policy weight training.
- Builders continue oracle exclusion/replay, semantic data-validation/cache work, and kernel task/exclusion/difficulty improvements. Their exact source and test receipts live in their component checkpoint and verification documents.

Reliable standard accelerator rewards, an actual coding-policy optimizer run, demonstrated learning gains and independent review of the latest code remain separate release requirements. No public release, outreach or external adoption is recorded here.

New daily entries should list completed features/fixes, exact tested source, test receipts and limitations. Changes under active development remain unreleased until their snapshot has supporting review and reproduction evidence.
