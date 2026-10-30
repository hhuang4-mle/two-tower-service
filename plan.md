# Copied text-embedding migration plan

Status: the standalone text-embedding implementation and Vertex AI batch
smoke test are complete. The user reported a successful test on 2026-10-30.
Full-pipeline implementation and Cloud Run cutover are not complete.

## Progress and next milestone

| Step | Status | Remaining work |
|---|---|---|
| 1. Contracts and configuration | Partially complete | Logical contracts defined and serialized BigQuery smoke path validated; confirm real customer/product schemas, physical projection/failure schemas, and pipeline configuration. |
| 2. Inference-only text embedding | Complete for the initial milestone | Independent image, model registration, and BigQuery-to-BigQuery Vertex smoke test successful. |
| 3. Pipeline preprocessing | Next | Implement source normalization/deduplication and staging SQL, followed by projection-input assembly. |
| 4. Full Vertex AI Pipeline | Not started | Implement orchestration and integrate both model stages. |
| 5. Postprocessing and failures | Not started | Implement reconciliation, selective retries, failure mapping, publication, and cleanup. |
| 6. End-to-end validation | Partially complete | Standalone encoder smoke test passed; full customer/product pipelines and failure paths remain unvalidated. |
| 7. Cutover | Partially complete | Directory separation and build paths complete; replace and retire legacy execution only after full-pipeline validation. |

**Start next with Step 3**, first resolving the source/staging details still
open in Step 1:

1. Confirm real customer/product table names, identifier types, text columns,
   BigQuery locations, and staging/output configuration.
2. Implement BigQuery preprocessing under `pipeline/sql/`: normalize NULL
   texts, deduplicate across both customer fields within a run, and stage
   distinct encoder inputs while preserving source-row mappings.
3. Validate that SQL with small fixtures and connect preparation plus the
   registered encoder to an initial Vertex AI Pipeline.
4. Add projection-input assembly and the remaining stages as the independent
   projection image and checkpoint loader become available.

The projection service is still a placeholder. Implementing its supplied
two-tower head, fail-closed checkpoint loading, and independent image is a
prerequisite for completing Steps 4 and 6. Confirm trained-checkpoint
availability before production integration.

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

Status: partially complete.

Initial contracts are in
[`batch_text_embedding/README.md`](batch_text_embedding/README.md). The smoke
test has validated the standalone serialized BigQuery input/output path in
Vertex. Real customer/product schemas, identifier types, physical projection
and failure schemas, and full-pipeline resources/IAM remain to be confirmed;
the smoke result does not verify every production configuration.

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

Status: next; not implemented.

Reuse compatible logic from `preprocess_input()` and
`pipeline/legacy/vertex_pipeline/sql/preprocess.sql` rather than maintaining
two independent implementations. Put the new SQL under `pipeline/sql/`,
outside the predictor and legacy directories.

Prepare distinct text inputs in BigQuery before starting the encoder job.
After reconciling encoder predictions, construct distinct ordered customer
embedding pairs or product embedding inputs for the projection job.
Keep the empty-text embedding convention; do not substitute zero vectors.

### 4. Construct the full Vertex AI Pipeline

Status: not started; `pipeline/README.md` is a placeholder for the new
implementation. The preserved text-only legacy graph is not this pipeline.

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
The serialized encoder path passed the smoke test. Separately verify
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

Status: not started.

Reuse compatible validation and join logic from `postprocess_output()` and
`pipeline/legacy/vertex_pipeline/sql/postprocess.sql`, but do not copy the
existing `embedding = []` publication behavior into the final output contract.

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

Status: partially complete. The standalone encoder's Vertex smoke test passed.
Full-pipeline validation, projection integration, named-field input, and
negative/error-path results are not established by that report.

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
