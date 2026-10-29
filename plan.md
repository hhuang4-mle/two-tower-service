# Copied text-embedding migration plan

Status: initial contracts and predictor changes implemented; Vertex smoke
validation pending. This plan remains separate from the copied project's
documentation. Full-pipeline migration and Cloud Run cutover remain deferred.
The predictor is now isolated in `batch_text_embedding/`; copied workflow
material is preserved under `pipeline/legacy/`.

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
| `pipeline/` | Reserved for the future full pipeline; currently documented by a README only. |
| `batch_projection/` | Placeholder README for the future projection image. |

## Migration steps

### 1. Define pipeline contracts and configuration

Initial contracts are in
[`batch_text_embedding/README.md`](batch_text_embedding/README.md). Physical
Vertex/BigQuery schemas, identifier types, resources, and IAM still require
deployment verification.

Specify customer and product source schemas, staging tables, prediction/error
schemas, and final publication mechanics before moving workflow logic.
Keep customer-only and product-only runs independent.

Use the existing architecture's contracts: normalize NULL/missing text to
`""`, otherwise compare text exactly, and deduplicate individual texts across
both customer fields within a run. Preserve ordered long-term/mid-term pairs
and source-row multiplicity. Keep text as the join key.

Define separate Vertex model resources and artifact/image versions for text
embedding and projection. Verify project, dataset locations, artifact paths,
service accounts, and IAM rather than inheriting copied defaults.

### 2. Make text embedding an inference-only application

Retain artifact loading, readiness, prediction-request validation,
tokenization, and model inference in `batch_text_embedding/`. Do not move these
model-local operations into BigQuery preprocessing.

Keep BigQuery preparation, deduplication, source-row reconstruction, batch-job
submission/polling, publication, and cleanup outside the text-embedding image.
Organize those responsibilities with the future full pipeline, outside the
text-embedding application.

Audit the copied predictor against the target model contract before reuse:
768-dimensional outputs, the complete saved Sentence Transformers artifact,
no added prompt, retained `normalize_embeddings=True`, BF16 encoder inference,
and FP32 vectors for projection. Review the current `encode_document`,
`normalize_embeddings`, and `TRUNCATE_DIM` behavior rather than assuming it
matches training. As agreed, truncate oversized inputs with the saved
tokenizer to at most 2,048 tokens including special tokens; retain original
text join keys. Assume 768 dimensions for the pipeline without rejecting
other predictor dimension options.

The existing predictor has been updated accordingly without changing its
BigQuery-compatible request formats or response schema. Cloud-only validation
is documented in
[`VERTEX_SMOKE_TEST.md`](batch_text_embedding/VERTEX_SMOKE_TEST.md)
for the user to run; no smoke jobs have been submitted.

### 3. Adapt preprocessing into explicit pipeline steps

Reuse compatible logic from `preprocess_input()` and
`pipeline/legacy/vertex_pipeline/sql/preprocess.sql` rather than maintaining two independent
implementations.

Prepare distinct text inputs in BigQuery before starting the encoder job.
After reconciling encoder predictions, construct distinct ordered customer
embedding pairs or product embedding inputs for the projection job.
Keep the empty-text embedding convention; do not substitute zero vectors.

### 4. Construct the full Vertex AI Pipeline

Adapt the copied text-only graph into a full pipeline with separately visible
steps:

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

Use named-field Vertex inputs and retain text keys in prediction outputs.
Confirm actual output/error schemas rather than relying on output row order.
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

Reuse compatible validation and join logic from `postprocess_output()` and
`pipeline/legacy/vertex_pipeline/sql/postprocess.sql`, but do not copy the existing
`embedding = []` publication behavior into the final output contract.

Validate finite 768-dimensional intermediate vectors. Validate final vectors
as finite, 128-dimensional, and unit length within `1e-4`; exclude zero or
invalid final vectors. Reconcile missing, failed, duplicate, and unexpected
predictions and map failures to affected source records.

Retry failed/incomplete keys for up to two rounds after the initial attempt,
reducing batches or isolating records when necessary. Surface model-loading
and configuration failures as stage failures instead of retrying every key.
Publish available successful records with their original multiplicity and
report remaining affected records in a separate failure table.

Skip inference for empty inputs while preserving appropriate empty outputs.
Keep temporary tables by default; offer cleanup only after reconciliation.
Avoid logging raw customer text.

### 6. Validate before replacing the current entry point

Add focused tests for deduplication, NULL/empty/whitespace text, ordered
customer pairs, duplicate source rows, invalid vectors, missing predictions,
retry limits, and empty inputs. Verify the predictor can run without the
workflow runner and each batch image loads only its own model.

Compile the new pipeline and run small customer-only and product-only
end-to-end jobs to confirm BigQuery schemas, job dependencies, visible step
outcomes, failure mapping, publication, and cleanup behavior.

Projection integration depends on its independent image and checkpoint
loader. Synthetic weights are test-only; an unavailable or failed trained
checkpoint must prevent production projection readiness and prediction.
Do not treat successful encoder execution as full-pipeline acceptance.

### 7. Cut over commands and documentation

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
