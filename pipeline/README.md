# Vertex AI text-embedding pipeline

Run customer or product text through EmbeddingGemma-300m BF16, then map
768-dimensional embeddings back to the original BigQuery rows. Projection
through the two-tower neural network is deferred.

```text
Prepare distinct texts → Vertex batch prediction → Validate/retry
    → Map back to source → Publish _d768 output and failures
```

## 1. Install

Run commands from `pipeline/`:

```bash
make install
```

No local model download or Docker build is needed. Use the registered model
that passed the encoder smoke test.

## 2. Submit

### Product example

```bash
make submit \
  ENTITY_TYPE=product \
  SOURCE_TABLE=gcp-wow-rwds-ai-super-plo-dev.embedding_smoke_20261029_225755.sample_products \
  TEXT_COLUMNS="product_embedding_text" \
  MODEL_RESOURCE_NAME=projects/gcp-wow-rwds-ai-super-plo-dev/locations/us-central1/models/embedding_smoke_20261029_225755 \
  PIPELINE_ROOT=gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/vertexai-pipelines/embedding-pipeline \
  BIGQUERY_LOCATION=US \
  REPLICA_COUNT=1
```

### Customer example

Replace the customer table and text-column names with your own.
**Text columns must be ordered long-term first, mid-term second.**

```bash
make submit \
  ENTITY_TYPE=customer \
  SOURCE_TABLE=gcp-wow-rwds-ai-super-plo-dev.two_tower.YOUR_CUSTOMER_TABLE \
  TEXT_COLUMNS="long_term_user_text mid_term_user_text" \
  MODEL_RESOURCE_NAME=projects/gcp-wow-rwds-ai-super-plo-dev/locations/us-central1/models/embedding_smoke_20261029_225755 \
  PIPELINE_ROOT=gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/vertexai-pipelines/embedding-pipeline \
  BIGQUERY_LOCATION=US \
  REPLICA_COUNT=1 \
  RETRY_ROUNDS=0
```

Both examples use the default service account:
`super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com`.
Override `SERVICE_ACCOUNT` if needed.

Before submitting, confirm the model/table exist, text columns are scalar
`STRING`, and `BIGQUERY_LOCATION` matches the source dataset. The account
needs BigQuery read/write/job access, Vertex access, and access to the GCS
pipeline root and model artifacts. The caller needs permission to use that
account; relevant Vertex service agents need BigQuery/image access. Confirm
L4 quota and network access for component dependency installation.

Submission recompiles the pipeline, creates billable cloud resources, prints
the pipeline resource name, and returns without waiting. Monitor the run in
**Vertex AI > Pipelines**. To compile without submitting, use `make compile`.

## 3. Check outputs

All tables are created in the source dataset. For source `<source>`:

| Purpose | Derived table |
|---|---|
| Successful source rows with embeddings | `<source>_d768` |
| Unresolved source-row failures | `<source>_d768_failures` |
| Distinct encoder inputs | `<source>_d768_input_<run_id>` |
| Validated text embeddings | `<source>_d768_embeddings_<run_id>` |
| Vertex job metadata | `<source>_d768_jobs_<run_id>` |

The output preserves all original columns, NULL values, IDs, and duplicate
rows, then appends:

| Run type | Added columns, each `ARRAY<FLOAT64>` with 768 elements |
|---|---|
| Product | `product_embedding` |
| Customer | `long_term_embedding`, `mid_term_embedding` |

Customer text columns share one deduplicated embedding table, joined twice.
NULL becomes `""` only for encoding/joining; other text is compared exactly.
Original text remains the join key even when the tokenizer truncates it.
No ID parameter is required.

Only rows with every required vector valid enter `_d768`. Failures contain
`source_record JSON`, `failed_fields ARRAY<STRING>`, `stage`, `error_code`,
`attempts`, and `message`. Inspect the job manifest and corresponding Vertex
jobs for raw prediction/error tables.

**Existing output and failure rows are replaced together on each successful
publication.** Destination schemas must remain compatible. Keep the source
unchanged during the run, and do not use source tables that already contain
the added embedding column names.

## Options and retries

| Make variable | Default / meaning |
|---|---|
| `PROJECT_ID` | `gcp-wow-rwds-ai-super-plo-dev` |
| `VERTEX_LOCATION` | `us-central1`; must match the model |
| `REQUEST_BATCH_SIZE` | `32` |
| `REPLICA_COUNT` | `4`; examples use one L4 replica to limit cost |
| `RETRY_ROUNDS` | `2` after the initial attempt; use `0` to disable or `1` for one retry |
| `CLEANUP_INTERMEDIATES` | `false`; `true` removes input/retry and validated embedding tables after publication |
| `PYTHON` | `python` |

The graph always shows three attempt stages, but retries create jobs only
for unresolved texts. Successful embeddings are reused; retry request batches
are halved. Invalid/duplicate/missing predictions remain unresolved; unexpected
text keys and whole-job failures stop the pipeline. Empty sources create no
GPU jobs and publish empty outputs.

## Concurrency and recovery

Customer/product runs can execute concurrently for different source tables.
A `<source>_d768_lock` table prevents overlapping runs for the same source and
is normally released even after upstream failure.

If cancellation or infrastructure failure leaves a lock, confirm its owning
execution and batch jobs have stopped before manually removing that exact
lock. Cancelling a pipeline does not guarantee cancellation of its Vertex
batch job. Submit a new pipeline after recovery rather than manually retrying
individual tasks in the old execution.

Intermediate tables are kept by default. Job manifests and raw Vertex tables
remain even when cleanup is enabled. Validate mapping, retries, failures,
empty inputs, and cleanup in small cloud runs before production use.
Compilation alone does not establish cloud validation.

The old Cloud Run workflow remains in [`legacy/`](legacy/README.md).
See [`plan.md`](../plan.md) for deferred projection and cutover work.
