# Two-tower embedding and recommendation services

Status: design draft. The text predictor has initial inference-contract
updates; Vertex smoke validation is pending. The full pipeline, projection
service, and recommendation API are not implemented.

## Goal

Generate 128-dimensional, L2-normalized customer or product embeddings with
Vertex AI batch prediction. Customer and product inputs arrive independently
and use separate BigQuery source tables and jobs. Vertex AI Pipelines
coordinates preparation, the two batch prediction stages, and publication.
An independent API service will encode in-session customer signals, combine
them with precomputed customer embeddings, and perform product recommendation.
Its design and implementation are deferred; matching, ranking, and score
calculation are outside the current batch implementation scope.

EmbeddingGemma-300m generates the 768-dimensional text embeddings used by the
trained two-tower projection head. The customer head has two towers, ordered
long-term then mid-term, with gated fusion. The product head uses the item
tower. For batch inference, call `project_user` or `project_item` directly;
`forward`, `log_temperature`, and optional `logq` handling are not required.
Training remains outside this implementation's scope.

## Three independent application images

| Image | Responsibility | Execution |
|---|---|---|
| Text embedding | Load EmbeddingGemma only; encode distinct texts into 768-dimensional base embeddings. | Vertex AI batch prediction |
| Batch projection | Load the trained `TwoTowerProjectionHead` only; project precomputed customer or product base embeddings into 128-dimensional vectors. | Vertex AI batch prediction |
| Recommendation API | Future in-session signal encoding, combination with precomputed customer embeddings, and product recommendation. | Online deployment to be designed |

Build and version the images independently. Neither batch image runs the
other model or owns pipeline orchestration. The projection stage consumes
precomputed embeddings from BigQuery, not raw text requiring an encoder;
text fields remain available as join keys. It supports customer-only and
product-only jobs independently.

The API is currently represented only by [`api/README.md`](api/README.md).
Do not implement its endpoints, inference, signal fusion, retrieval, Dockerfile,
or deployment yet. It is not assumed to be model-free: how it encodes
in-session signals and performs recommendation will be designed later.

## Pipeline-owned two-stage batch architecture

```text
Vertex AI Pipeline (customer run or product run)
    |
    v
BigQuery source table (customers or products)
    |
    v
Pipeline preparation: normalize NULL/missing text to "" and deduplicate texts
    |
    v
Vertex batch job 1: EmbeddingGemma
    |-- encode each distinct text to 768 dimensions
    `-- write text and embedding to a BigQuery intermediate table
    |
    v
Pipeline BigQuery joins
    |-- customers: build distinct (long-term, mid-term) embedding pairs
    `-- products: build distinct product embeddings
    |
    v
Vertex batch job 2: two-tower projection head
    |-- project and fuse each distinct customer pair, or project each product
    `-- write 128-dimensional predictions to BigQuery
    |
    v
Pipeline reconciliation, retries, source-row joins, and publication
    |
    v
BigQuery final customer or product embeddings

Future recommendation API (separate image; detailed design deferred)
    |-- encode in-session customer signals
    |-- combine signals with precomputed customer embeddings
    `-- perform product recommendation
```

Vertex AI Pipelines owns BigQuery preparation and deduplication, launching and
monitoring both Vertex batch prediction jobs, intermediate joins, validation,
reconciliation, retry rounds, final publication, and optional temporary-table
cleanup. Model containers own only artifact loading and Vertex-compatible
prediction and health routes. They do not submit downstream jobs, perform
BigQuery joins, or publish final tables. Vertex handles the BigQuery input and
output of each batch prediction job.

Use managed pipeline components for BigQuery and Vertex job operations rather
than introducing a fourth application image. Pipeline definitions and
configuration are separate from the three application images; platform-owned
component images do not constitute additional application services. The API
does not orchestrate these batch runs.

Use Vertex AI Pipelines rather than a long-running Cloud Run job so individual
steps and their outcomes are visible. This does not imply unlimited execution:
verify pipeline task and Vertex batch job timeout, cancellation, and retry
settings before deployment.

For example, customers A and B may share long-term text but have different
mid-term text. Encode their three distinct texts once each, then project two
different customer pairs. Deduplication is by exact text, not by customer ID.
After NULL normalization, do not trim, change case, or apply other text
normalization.

Each customer or product run deduplicates all relevant individual texts in its
own input. Because all text types use the same encoder without prompts,
identical text can share a base embedding within a run. Reuse across
independently scheduled runs is not required by this design.

The final join must preserve every successful source row and its multiplicity,
including repeated `crn` or `article_nbr` values. Customer projection keys are
the ordered pair of normalized texts; product projection keys are normalized
product text. Join intermediate predictions using text directly, not a
generated key.

## Vertex and compute

Both batch stages use Vertex AI batch prediction in `us-central1`, coordinated
by Vertex AI Pipelines. The initial
configuration for each stage is four `g2-standard-4` replicas, with one NVIDIA
L4 GPU per replica and a maximum of four replicas. Confirm quota, regional
availability, and actual Vertex job support before deployment.

For EmbeddingGemma, set the initial Vertex request batch size to 32 unique
texts. Container microbatching is a separate configurable limit. The earlier
idea of 32 customers producing a 64-text encoder batch does not apply to the
two-stage design. Select and benchmark the projection request batch size
separately. Handle partial batches and larger requests safely. Four replicas
do not guarantee uniform work distribution or linear speedup.

Use Vertex `instanceConfig.instanceType = "object"` so BigQuery input rows
reach the custom container as named fields rather than the default array.
Keep the text fields available in batch prediction output for the BigQuery
joins. Verify the actual prediction and error schemas and field preservation
with a small end-to-end job; do not rely on global output row order.

Each batch image exposes Vertex-compatible HTTP prediction and health routes. Each
replica loads its stage's model artifacts once and reports readiness only when
ready to serve. Use `eval()` and inference/no-gradient mode.

EmbeddingGemma and its tokenizer/configuration are assumed to exist at:

```text
gs://wx-1b759ae5-bd00-4de8-bf4d-427baa30e931/fs/text-embedding-models/embeddinggemma-300m/
```

Pipeline components and Vertex jobs require appropriate BigQuery access;
model containers require access to their own artifacts.
Artifact layout, local download mechanism, image, service account, IAM roles,
and compatible BigQuery dataset locations must be verified before deployment.

## Model and precision contract

Use the complete saved Sentence Transformers artifact, including its pooling
and normalization modules, with no added prompt. Keep
`normalize_embeddings=True` from the copied predictor as agreed, in addition
to any saved normalization module.
Ensure the encoding API does not silently add a default prompt. The model
version, tokenizer, pooling, normalization, and preprocessing used in
training must match inference.

Truncate oversized text with the saved model's tokenizer from the right,
retaining the beginning within at most 2,048 tokens including special tokens,
not bytes or characters. Keep a smaller saved sequence limit if present.
Preserve original text as the pipeline join key; do not deduplicate by
truncated tokens. Training must use the same truncation convention.

The text inference and logical pipeline data/failure contracts are defined in
[`batch_text_embedding/README.md`](batch_text_embedding/README.md).
The pipeline assumes 768-dimensional base embeddings. The predictor retains
its existing output-dimension options without enforcing 768; pipeline/model
configuration must select compatible dimensions.

Mid-term text may be `""`. Pass it to EmbeddingGemma as empty text rather than
substituting an all-zero vector. Its embedding will normally be nonzero, so the
provided projection head will treat the mid-term tower as present in its fusion
gate. Training must use the same convention. Missing or NULL long-term and
product text are also normalized to empty strings.

Use BF16 for EmbeddingGemma on L4. Convert its output to FP32 before the
projection head; retain the supplied head's FP32 computation and
normalization. Do not silently substitute FP16: the EmbeddingGemma model card
says its activations do not support FP16. An FP32-versus-BF16 quality
comparison is not part of the current design.

The projection head is not trained yet. Leave a future `load_model_weights()`
hook for integration, but an unimplemented or failed loader must prevent
projection-stage readiness and prediction. Random initialization must never
be presented as a trained model. The eventual loader must validate architecture,
checkpoint keys, and dimensions. Synthetic test-only weights may be used for
testing and must not be used for production predictions.

Use the supplied `TwoTowerProjectionHead` architecture without changing its
projection or fusion logic. Although the class supports one or multiple
customer towers and defaults to one, configure this deployment explicitly with
`gemma_dim=768`, `projection_dim=128`, and `num_customer_towers=2`.
Customer inputs have shape `[batch, 2, 768]`, ordered long-term then mid-term;
product inputs have shape `[batch, 768]`. Each produces `[batch, 128]`.
The checkpoint's dropout configuration must also match training.

The head normalizes each customer tower, computes gated fusion weights with
absent towers masked, and normalizes the weighted sum. Its first customer
tower is always treated as present. Preserve these semantics, including the
existing nonzero empty-text embedding convention. Batch projection does not
call `forward` to produce similarity matrices or pairwise temperature-scaled
scores, and does not append a `logq` feature to product vectors.

## BigQuery outputs and lifecycle

The final customer table contains `crn`, `long_term_user_text`,
`mid_term_user_text`, and the 128-dimensional embedding. The final product
table contains `article_nbr`, `product_text`, and the embedding. Do not include
run ID or model-version metadata in these final tables.

Source texts are assumed not to change during a job; a separate source
snapshot is not required under this assumption. Outputs may be partial.
Successful records may be published even when other records fail; consumers
must not assume that a completed or visible result contains every input record.
Exact table names, schemas, and publication mechanics remain to be specified.

Keep temporary tables by default. Provide a configuration option to remove
them after the run has been reconciled. Temporary and final tables contain raw
text, so their access, retention, and cleanup policy require appropriate
controls. Avoid logging raw customer text.

## Validation and failures

Validate that final embeddings have 128 elements, are finite, and have an L2
norm within `1e-4` of 1. Exclude zero, materially non-unit, or non-finite
vectors from final results, emit warnings, and report affected source records
in a separate failure table after retries. Do not replace them with plausible
values. An invalid or non-finite intermediate embedding must not silently
propagate into a successful projection.

The pipeline must automatically retry failed or incomplete keys for up to two rounds after the
initial attempt. Retry affected records and, when needed, reduce batch size or
isolate individual records. Model-loading and configuration failures must be
surfaced as stage failures rather than blindly retrying every record.
Reconcile successful, failed, and incomplete predictions, including records
that depend on a failed text. Verify Vertex batch error behavior and the
mechanics of per-key retry with an end-to-end job.

## Acceptance checks

- Verify no prompts are added and the saved artifact's pooling and
  normalization are retained.
- Verify identical text is encoded once per run, including across long-term and
  mid-term fields, and that distinct text pairs are projected separately.
- Verify tower order, empty text, duplicate source-row multiplicity, output
  dimensions and norm, partial results, warnings, and failure mapping.
- Exercise both Vertex stages and BigQuery joins with a small end-to-end job,
  including retries and actual output and error schemas.
- Verify the two batch images load only their respective models and that
  projection accepts precomputed embeddings without loading EmbeddingGemma.
- Verify pipeline step visibility, customer-only and product-only execution,
  failure propagation, retry limits, publication, and optional cleanup.
- Test locally with clearly identified synthetic weights; validate the real
  checkpoint and training/inference compatibility when trained weights exist.

## Implementation plan

1. Define BigQuery source, intermediate, output, and failure schemas and
   pipeline configuration. Keep text join keys and source-row multiplicity,
   with separate customer and product runs.
2. Implement and independently package the text-embedding image with saved
   EmbeddingGemma artifacts, Vertex health/prediction routes, BF16 inference,
   and input/output validation.
3. Implement and independently package the batch-projection image using the
   supplied head, explicit two-customer-tower configuration, FP32 inference,
   and a fail-closed trained-checkpoint loader. Accept only precomputed base
   embeddings for model input; preserve text keys for downstream joins.
4. Define the Vertex AI Pipeline using managed components for BigQuery
   preparation, text embedding batch prediction, embedding joins, projection
   batch prediction, reconciliation and selective retries, final source joins,
   publication, and optional cleanup. Keep orchestration out of both model
   containers and the API.
5. Validate both images locally with synthetic test-only projection weights,
   then exercise the complete customer and product pipelines in Vertex.
   Validate the real checkpoint when it becomes available; do not enable
   production projection without trained weights.
6. Reserve `api/` with a brief README only. Defer the third image's
   implementation until in-session encoding, combination with precomputed
   customer embeddings, recommendation behavior, interfaces, and deployment
   are agreed.

## Remaining implementation details

1. BigQuery project, datasets, table names, schemas, output publication
   mechanics, and temporary table cleanup timing.
2. Expected data volumes, text lengths, schedule, completion target, and
   projection batch size.
3. Artifact versions, checkpoint format, IAM, container configuration, and
   training/inference compatibility of tokenizer truncation and normalization.
4. Vertex-specific output schemas, error reporting, retry implementation, and
   warning/monitoring destinations.
5. Pipeline component selection, service accounts, task/job timeout and
   cancellation settings, scheduling, and independent image release/version
   configuration.
6. Future API contracts, in-session encoder/model requirements, signal fusion,
   access to precomputed embeddings, product retrieval/ranking, freshness,
   latency, and online hosting. These are deferred, not batch prerequisites.

## References

- [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m)
- [Vertex AI custom container requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements)
- [Vertex AI batch prediction job API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.batchPredictionJobs)
- [Vertex AI inference machine types](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute)
- [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/)
- [Vertex AI batch predictions](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions)
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
