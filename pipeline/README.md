# Full Vertex AI Pipeline

This directory is reserved for orchestration of the complete two-tower batch
workflow. The new pipeline is not implemented yet.

It will own BigQuery preprocessing and deduplication, text-embedding batch
prediction, assembly of customer/product projection inputs, projection batch
prediction, reconciliation and selective retries, final source joins,
publication, and optional cleanup.

Use Vertex AI Pipelines and managed BigQuery/Vertex components where suitable.
Do not create a fourth application image or put orchestration in either model
container or the recommendation API.

The future implementation will introduce `pipeline.py`, `submit_pipeline.py`,
`requirements.txt`, and `sql/` as needed. No dummy implementations or compiled
pipeline templates are provided.

## Preserved migration material

[`legacy/`](legacy/README.md) contains the copied Cloud Run runner, shared
workflow, text-only Vertex pipeline, SQL, and original documentation. It is
not the complete pipeline described above.

Run its commands from `pipeline/legacy/`; retain it until replacements have
been validated. The predictor's independent build commands now live in
[`batch_text_embedding/`](../batch_text_embedding/README.md).

See [`plan.md`](../plan.md) for migration steps and
[`ARCHITECTURE.md`](../ARCHITECTURE.md) for the target design.
