# Copied text-embedding migration plan

Status: the standalone text-embedding implementation and Vertex AI batch
smoke test are complete. The user reported a successful test on 2026-10-30.
The encoder-only pipeline is implemented and compiles; its cloud execution
has not yet been validated. Full two-stage integration and Cloud Run cutover
are not complete.

## Progress and next milestone

| Step | Status | Remaining work |
|---|---|---|
| 1. Contracts and configuration | Partially complete | Logical contracts defined and serialized BigQuery smoke path validated; confirm real customer/product schemas, physical projection/failure schemas, and pipeline configuration. |
| 2. Inference-only text embedding | Complete for the initial milestone | Independent image, model registration, and BigQuery-to-BigQuery Vertex smoke test successful. |
| 3. Pipeline preprocessing | Implemented; cloud validation pending | Verify normalization, cross-field deduplication, and staging with real BigQuery fixtures. |
| 4. Full Vertex AI Pipeline | Encoder-only implementation complete | Validate the initial pipeline in Vertex; add projection when ready. |
| 5. Postprocessing and failures | Encoder-only implementation complete | Validate reconciliation, retries, source mapping, transactional publication, and cleanup in BigQuery/Vertex. |
| 6. End-to-end validation | Partially complete | Standalone encoder smoke test passed; full customer/product pipelines and failure paths remain unvalidated. |
| 7. Cutover | Partially complete | Directory separation and build paths complete; replace and retire legacy execution only after full-pipeline validation. |

**Next: validate the new encoder-only pipeline in Vertex**, first supplying
the source and submission settings still open in Step 1:

1. Confirm real customer/product source schemas and text columns, BigQuery
   locations, model resource, and submission settings. Preserve source
   identifier types automatically and derive staging/output names.
2. Follow [`pipeline/README.md`](pipeline/README.md) to submit small customer
   and product runs, initially with one replica.
3. Validate deduplication, mapping, retries, failures, empty inputs, source-lock
   behavior, and `_d768` publication. No local model downloads are required;
   no test suite was added, as requested.
4. Integrate neural-network projection later, when its image and checkpoint
   loader are available.

The projection service is still a placeholder. Implementing its supplied
two-tower head, fail-closed checkpoint loading, and independent image is a
prerequisite for the full two-stage scope of Steps 4 and 6, but not for the
initial text-embedding pipeline below. Confirm trained-checkpoint availability
before production integration.

## Agreed initial pipeline: text embedding and source mapping

One parameterized Vertex AI Pipeline definition is now implemented. Launch customer
and product encoding as separate runs with different parameters; they may
execute concurrently without waiting for one another. Do not require the
projection model or load it in this initial pipeline.

### Execution steps

1. Read the customer or product source table.
2. Normalize NULL text to `""` and deduplicate exact texts. Customer runs
   combine both text columns before deduplication; retain the original
   long-term/mid-term ordering for later mapping.
3. Run the registered EmbeddingGemma model with Vertex batch prediction to
   produce 768-dimensional text embeddings.
4. Validate and reconcile predictions against staged text keys. Check
   dimensions, finite/nonzero values, missing/error records, duplicates, and
   unexpected keys. Retry failed/incomplete keys for at most two rounds after
   the initial attempt; report unresolved failures separately.
5. Join validated embeddings back to the original source: twice for customer
   long-term/mid-term fields, once for product text. Preserve source-row
   multiplicity; do not join or deduplicate by customer/product ID.
6. Publish projection-ready `_d768` output and source-row failures. Publish a
   customer row only when both embeddings are valid, and a product row only
   when its embedding is valid. Keep intermediate tables by default.

Stop after this mapping/publication step. The future projection stage will
stack customer embeddings in long-term, mid-term order and generate
128-dimensional vectors. It is not part of this initial implementation.

### Required run parameters

| Parameter | Meaning |
|---|---|
| `entity_type` | `customer` or `product`. |
| `source_table` | Fully qualified `project.dataset.table` source. |
| `text_columns` | Customer: ordered `[long_term_column, mid_term_column]`; product: `[product_text_column]`. Source column names are configurable. |
| `embedding_model_resource_name` | Exact registered Vertex model resource, not its display name. |

No `id_column` is required. IDs are not sent to the encoder or used as
embedding join keys. Preserve all original source columns, including any
identifiers, in the mapped output and source-row failure reports.

No output, failure, or staging table/dataset parameter is required for this
initial design. Derive them from `source_table` in its existing dataset.

### Configuration and submission settings

| Setting | Default or requirement |
|---|---|
| `project_id` | `gcp-wow-rwds-ai-super-plo-dev`. |
| `vertex_location` | `us-central1`. |
| `bigquery_location` | Explicit location compatible with the source dataset and Vertex region. |
| `request_batch_size` | `32`. |
| `replica_count` | `4`, configurable for quota/cost; initially use the same starting and maximum count. |
| Compute | `g2-standard-4`, one NVIDIA L4 per replica. |
| `retry_rounds` | `2` after the initial attempt. |
| `cleanup_intermediates` | `false`; cleanup only after reconciliation when enabled. |
| Submission | Configure a GCS pipeline root. Execution account defaults to `super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com`, is overridable, and is also passed to the batch container. |

The registered model already references its image and GCS artifacts; those
URIs are not pipeline run inputs. Disable data-dependent caching for mutable
tables. Keep source data unchanged throughout the run so final joins use the
same text relationships as preprocessing.

### Derived BigQuery table names

All tables use the source project/dataset. Here `<source>` means the fully
qualified source table and `<run_id>` is generated automatically with a
BigQuery-safe, unique value.

| Purpose | Derived name |
|---|---|
| Mapped output | `<source>_d768` |
| Unresolved source-row failures | `<source>_d768_failures` |
| Distinct encoder input | `<source>_d768_input_<run_id>` |
| Validated distinct-text embeddings | `<source>_d768_embeddings_<run_id>` |
| Batch-job manifest | `<source>_d768_jobs_<run_id>` |
| Exclusive source lock | `<source>_d768_lock` |

For example, `project.dataset.customer_text` publishes to
`project.dataset.customer_text_d768`. Vertex generates its own raw prediction
and error table names; record those names rather than assuming or renaming
them.

Unique intermediate names isolate concurrent customer/product runs. Prevent
overlapping runs for the same source because their fixed output/failure names
would collide. An atomically created lock table now rejects overlapping runs.
Publication replaces output and failure rows together in a BigQuery
transaction. Cloud validation is pending. See the pipeline README for stale
lock recovery and the unchanged-source requirement.

### Published output schemas

Preserve every original source column and value, including original NULL text
and identifier types; use NULL-normalized text expressions only for joins.
Append these standard embedding columns regardless of configured text-column
names:

| Entity | Appended columns |
|---|---|
| Customer | `long_term_embedding ARRAY<FLOAT64>`, `mid_term_embedding ARRAY<FLOAT64>` |
| Product | `product_embedding ARRAY<FLOAT64>` |

Each embedding contains 768 values representing FP32 model output. Reject
source schemas that already use the appended embedding column names rather
than silently overwriting source data.

The distinct-text intermediate table contains normalized `text STRING` and
`embedding ARRAY<FLOAT64>`; both customer text fields share this one table.
Mapping joins it twice, so customers do not require two text-embedding tables.
Failure output must identify the affected original source rows and failed
text field(s), with safe error details, without substituting empty or zero
vectors. The implemented schema contains `source_record JSON`,
`failed_fields ARRAY<STRING>`, `stage`, `error_code`, `attempts`, and `message`.
Raw output/error diagnostics remain accessible through the recorded jobs.

These are 768-dimensional intermediate outputs for future projection, not the
final 128-dimensional tables in `ARCHITECTURE.md`. Preserving all source
columns supersedes the earlier strict standardized source-field output
proposal for this initial stage only.

## Goal and boundaries

Adapt the copied `batch_text_embedding/` project to the three-image design in
[`ARCHITECTURE.md`](ARCHITECTURE.md):

| Application image | Responsibility |
|---|---|
| Text embedding | Load EmbeddingGemma and encode prepared texts into 768-dimensional base embeddings through Vertex AI batch prediction. |
| Batch projection | Load the trained two-tower head and project precomputed customer or product embeddings into normalized 128-dimensional vectors through Vertex AI batch prediction. |
| Recommendation API | Future in-session encoding and recommendation; retain the existing placeholder only during this migration. |

The full Vertex AI Pipeline will own preprocessing, coordination of both batch
jobs, intermediate joins, postprocessing, retries, and publication. It will
replace Cloud Run Job orchestration, not Vertex AI batch inference.

## Current starting point

The predictor is now an independently buildable application. Its Dockerfile
copies only its application and requirements. The legacy runner, shared code,
SQL, text-only pipeline, and original documentation have been moved together
so their relative imports and runner build context are preserved.

| Existing location | Migration use |
|---|---|
| `batch_text_embedding/` | Independent text-embedding image, build/push commands, and inference contract. |
| `pipeline/legacy/shared/embedding_workflow/util.py` | Existing preprocessing, model registration, batch submission/polling, postprocessing, and cleanup logic to migrate. |
| `pipeline/legacy/batch_job_runner/` | Current Cloud Run orchestration entry point; retire after pipeline validation and cutover. |
| `pipeline/legacy/vertex_pipeline/` | Existing text-only pipeline, submission tooling, SQL, and compiled YAML to adapt. |
| `pipeline/legacy/Makefile` | Preserved Cloud Run and runner commands; predictor build/push targets forward to its new directory. |
| `pipeline/` | Implemented encoder-only graph, lightweight SDK components, SQL, submission tooling, Makefile, and README; full two-stage integration is deferred. |
| `batch_projection/` | Placeholder README for the future projection image. |

## Migration steps

### 1. Define pipeline contracts and configuration

Status: partially complete.

Initial contracts are in
[`batch_text_embedding/README.md`](batch_text_embedding/README.md). The smoke
test has validated the standalone serialized BigQuery input/output path in
Vertex. Real customer/product schemas, identifier types, physical projection
and failure schemas, and full-pipeline resources/IAM remain to be confirmed;
the smoke result does not verify every production configuration.

Confirm actual source schemas, prediction/error schemas, and publication
mechanics using the initial pipeline contract above. Derive destination names
instead of requesting them from the user. Customer/product runs share one
pipeline definition but execute independently.

Use the existing architecture's contracts: normalize NULL/missing text to
`""`, otherwise compare text exactly, and deduplicate individual texts across
both customer fields within a run. Preserve ordered long-term/mid-term pairs
and source-row multiplicity. Keep text as the join key.

Define separate Vertex model resources and artifact/image versions for text
embedding and projection. Verify project, dataset locations, artifact paths,
service accounts, and IAM rather than inheriting copied defaults.

### 2. Make text embedding an inference-only application

Status: complete for the initial standalone encoder milestone.

The predictor is isolated in `batch_text_embedding/`, with its own Dockerfile,
dependencies, Cloud Build configuration, and build/push commands. It owns
artifact loading, readiness, request validation, tokenization, and inference.
Preprocessing, joins, submission/polling, publication, and cleanup remain
outside the image. Copied orchestration is preserved in `pipeline/legacy/`.

Implemented behavior: complete saved Sentence Transformers artifact loading,
BF16 CUDA inference, FP32 output, no added prompt, retained
`normalize_embeddings=True`, and right truncation at at most 2,048 tokens
including special tokens. Original text remains the join key. The pipeline
assumes 768 dimensions without rejecting other predictor dimension options.
Existing BigQuery-compatible request formats and response schema are retained.

The uploaded artifact is
`gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m`.
The image repository is
`gcr.io/gcp-wow-rwds-ai-super-plo-dev/embeddinggemma-300m-bf16`;
the Vertex registry display name is `embeddinggemma-300m-bf16`.

The user successfully ran
[`VERTEX_SMOKE_TEST.md`](batch_text_embedding/VERTEX_SMOKE_TEST.md) on
2026-10-30, validating the standalone BigQuery -> Vertex inference -> BigQuery
flow. Exact image digest/tag, model resource, job name, and output table were
not recorded in this plan. This is not full-pipeline, performance, or
training-vector quality acceptance.

### 3. Adapt preprocessing into explicit pipeline steps

Status: implemented in `pipeline/sql/prepare.sql`; cloud validation pending.

Reuse compatible logic from `preprocess_input()` and
`pipeline/legacy/vertex_pipeline/sql/preprocess.sql` rather than maintaining
two independent implementations. Put the new SQL under `pipeline/sql/`,
outside the predictor and legacy directories.

Prepare distinct text inputs in BigQuery before starting the encoder job.
After reconciling encoder predictions, map them to all original source columns
and publish the agreed projection-ready `_d768` output. Neural-network input
assembly and distinct customer-pair projection remain future work.
Keep the empty-text embedding convention; do not substitute zero vectors.

### 4. Construct the full Vertex AI Pipeline

Status: initial encoder-only graph and submission tooling implemented and
compiled successfully. Full two-stage integration and cloud validation remain
pending. The preserved legacy graph is separate.

Adapt the copied text-only graph into a full pipeline with separately visible
steps. First implement the agreed encoder-only flow above; defer the
projection job until its model is ready. The eventual full flow is:

```text
Prepare and deduplicate source texts in BigQuery
    -> Vertex batch prediction: text-embedding image
    -> Reconcile base embeddings and assemble projection inputs
    -> Vertex batch prediction: batch-projection image
    -> Reconcile projections, join successful results to source, publish
    -> Optional cleanup after reconciliation
```

Add selective retry paths for failed/incomplete keys at each model stage.
Use managed BigQuery and Vertex components where suitable; do not introduce a
fourth application image or put orchestration in the API.

Retain the proven serialized encoder input format for the initial pipeline,
and keep text keys in prediction outputs. Separately verify
named-field input and the projection/error schemas as they are integrated;
do not infer those results from the encoder smoke test or rely on row order.
Align copied compute settings with the architecture: initially four
`g2-standard-4` replicas with one L4 each and a maximum of four; text request
batch size 32, with projection batch size benchmarked separately.

Keep model registration and pipeline submission in deployment/submission
tooling, outside the predictor. Replace the Cloud Run runner's polling loop
with pipeline-managed job dependencies. Verify task/job timeouts,
cancellation, and retry settings; Vertex AI Pipelines does not imply
unlimited execution. Disable data-dependent caching where mutable table
names could otherwise reuse stale results.

### 5. Adapt postprocessing and failure handling

Status: encoder reconciliation, selective retries, mapped publication, source
locking, and optional staging cleanup implemented; cloud validation pending.

Reuse compatible validation and join logic from `postprocess_output()` and
`pipeline/legacy/vertex_pipeline/sql/postprocess.sql`, but do not copy the
existing `embedding = []` publication behavior into the final output contract.

Validate finite 768-dimensional intermediate vectors. Validate final vectors
as finite, 128-dimensional, and unit length within `1e-4`; exclude zero or
invalid final vectors. Reconcile missing, failed, duplicate, and unexpected
predictions and map failures to affected source records.

For the initial pipeline, publish mapped 768-dimensional outputs and failures
using the derived names above; retain all source columns and require both
customer embeddings before publishing that source row. The 128-dimensional
checks apply only after future projection integration.

Retry failed/incomplete keys for up to two rounds after the initial attempt,
reducing batches or isolating records when necessary. Surface model-loading
and configuration failures as stage failures instead of retrying every key.
Publish available successful records with their original multiplicity and
report remaining affected records in a separate failure table.

Skip inference for empty inputs while preserving appropriate empty outputs.
Keep temporary tables by default; offer cleanup only after reconciliation.
Avoid logging raw customer text.

### 6. Validate before replacing the current entry point

Status: partially complete. The standalone encoder's Vertex smoke test passed.
The new encoder-only pipeline compiles and its Make commands have been
dry-run. Its cloud execution, projection integration, named-field input, and
negative/error-path results are not established by those checks.

Add focused SQL checks and cloud validation for deduplication,
NULL/empty/whitespace text, ordered customer pairs, duplicate source rows,
invalid vectors, missing predictions, retry limits, and empty inputs. Keep
model validation on Vertex; no local model downloads are required. Verify
each batch image loads only its own model.

Compile the new pipeline and run small customer-only and product-only
end-to-end jobs to confirm BigQuery schemas, job dependencies, visible step
outcomes, failure mapping, publication, and cleanup behavior.

Projection integration depends on its independent image and checkpoint
loader. Synthetic weights are test-only; an unavailable or failed trained
checkpoint must prevent production projection readiness and prediction.
Do not treat successful encoder execution as full-pipeline acceptance.

### 7. Cut over commands and documentation

Status: partially complete. Predictor/legacy directory separation, independent
build commands, and updated paths/documentation are done. The Cloud Run runner
and old workflow remain available under `pipeline/legacy/`; final execution
cutover and retirement are pending.

Only after validation, provide full-pipeline compile/submit commands and
update the active documentation. Retain the text-embedding image's independent
build/push targets.

Retire the copied Cloud Run deployment/execution targets and runner image
from the active workflow. Consolidate reusable SQL and orchestration under
the full pipeline and remove obsolete duplicated workflow paths and stale
compiled templates after their replacements are verified. Preserve the
ability to inspect existing executions; stopping a runner alone does not
cancel a previously submitted Vertex batch job.

## Completion criteria

The text-embedding application owns inference only. The full Vertex AI
Pipeline owns all batch preprocessing and postprocessing and coordinates both
independently packaged model images. Customer and product end-to-end checks
pass, no Cloud Run orchestration image is required, and the API remains a
placeholder. Existing Cloud Run orchestration behavior is preserved under
`pipeline/legacy/` until pipeline migration and cutover are implemented and
validated.
