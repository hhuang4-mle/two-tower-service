# Batch projection (placeholder)

This directory reserves the independent projection image. No application,
Dockerfile, or deployment is implemented yet.

The image will load the trained `TwoTowerProjectionHead` and serve
Vertex-compatible prediction and health routes. Customer inputs contain
ordered long-term/mid-term base embeddings shaped `[batch, 2, 768]`; product
inputs are `[batch, 768]`. Call `project_user` or `project_item` to produce
L2-normalized `[batch, 128]` embeddings in FP32.

Do not load EmbeddingGemma, train the head, or orchestrate BigQuery jobs here.
An unavailable or failed trained checkpoint must prevent readiness and
prediction. The full [`pipeline/`](../pipeline/README.md) will own data
preparation, batch-job submission, retries, and publication.

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) and [`plan.md`](../plan.md).
