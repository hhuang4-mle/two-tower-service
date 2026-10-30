# EmbeddingGemma-300m inference contract

The predictor loads the complete saved EmbeddingGemma-300m Sentence
Transformers artifact and serves inference for Vertex AI batch prediction.
Vertex, not this application, reads BigQuery inputs and writes raw predictions.
Preprocessing, deduplication, joins, retries, and publication belong to the
future full Vertex AI Pipeline. The copied Cloud Run runner and its commands
are preserved in [`pipeline/legacy/`](../pipeline/legacy/README.md).

## Build and layout

This directory is inference-only: `app.py`, `Dockerfile`, dependencies,
Cloud Build configuration, and these documents. Run `make build` or
`make push` from this directory. Override `PROJECT_ID`, `IMAGE_NAME`, or
`IMAGE_TAG` as needed.

The default image repository is
`gcr.io/gcp-wow-rwds-ai-super-plo-dev/embeddinggemma-300m-bf16`.
Build/push defaults to the `latest` tag; the smoke procedure uses a unique tag.

The full pipeline will be implemented in [`pipeline/`](../pipeline/README.md).
Legacy execution commands now run from `pipeline/legacy/`, not here.

## Input and output

`POST /predict` accepts named text instances:

```json
{"instances": [{"text": "Product description"}, {"text": ""}]}
```

For compatibility with the proven BigQuery workflow, it also retains raw
instance lists, JSON-encoded objects, and singleton-list wrappers. For example:

```json
{"instances": [["{\"text\":\"Product description\"}"]]}
```

Each `text` must be a string. Empty and whitespace-only strings are accepted.
NULL normalization and deduplication must happen upstream. The predictor does
not modify text itself; preprocessing inside the saved model/tokenizer remains
part of the training contract.

The existing response schema is unchanged:

```json
{"predictions": [[0.0123, -0.0456], [0.0234, 0.0567]]}
```

The example vectors are shortened. Each instance has one finite, nonzero
vector in request order, converted to FP32 before serialization. An empty
request returns `{"predictions": []}` without model inference.

Assume 768 dimensions for this pipeline. Do not enforce 768 in the predictor:
retain positive-integer `parameters.truncate_dim` overrides and the
`TRUNCATE_DIM` environment setting, defaulting to 768. Other dimensions remain
supported; callers are responsible for downstream compatibility.

Vertex must preserve original input text in its BigQuery prediction output for
joins. The predictor does not echo text or IDs. Do not join batch output by row
order. Verify both the existing serialized-input format and the future named
object format using the [Vertex smoke procedure](VERTEX_SMOKE_TEST.md).

## Model and tokenization

Require `modules.json` and load the complete saved artifact, retaining its
pooling, dense, and normalization modules. Do not reconstruct a bare model.
Load locally downloaded weights on CUDA in BF16, set `eval()`, and use
inference mode. Do not substitute FP16.

Use `encode(prompt="")` to prevent saved default/document prompts from being
added. As agreed, keep `normalize_embeddings=True` from the copied predictor,
in addition to any normalization module saved in the artifact.

Set the tokenizer sequence limit to **at most 2,048 tokens including special
tokens, not bytes or characters**. Retain a smaller saved limit if present.
Oversized text is truncated from the right, retaining the beginning. Do not
slice characters or decode/re-encode tokens. Training must use the same
prompt, normalization, and truncation conventions.

Keep the original untruncated text as the BigQuery join key. Different source
texts can have identical truncated token inputs without becoming the same key.

## Failures and readiness

| Condition | Behavior |
|---|---|
| Invalid JSON, instance/text type, parameters, or non-positive/non-integer dimension | HTTP 400 before inference; no source text in error messages. |
| Oversized text | Tokenizer truncation and normal inference, not rejection. |
| Unloaded model | HTTP 503 for health and prediction. |
| Artifact, configuration, or model initialization failure | Startup fails; no successful readiness. |
| Inference failure, malformed/incomplete output, non-finite or zero vector | HTTP 500 for the request; do not return partial vectors or placeholders. |
| Missing prediction in Vertex output | Future pipeline reconciles staged keys and Vertex errors; predictor does not fabricate a vector. |

`GET /health` returns 200 only after initialization succeeds. Failures are
request-wide; isolating records and retrying keys belongs to the pipeline.
The copied runner's existing `embedding=[]` partial-publication behavior is
not the future pipeline contract.

## Logical BigQuery boundaries

These are contracts for future pipeline implementation, not deployed schemas.
BigQuery source identifier types must be inspected and preserved.

| Boundary | Logical fields |
|---|---|
| Customer source | `crn`, `long_term_user_text STRING`, `mid_term_user_text STRING` |
| Product source | `article_nbr`, `product_text STRING` |
| Encoder input | `text STRING NOT NULL`, one row per distinct normalized text |
| Reconciled encoder output | `text STRING NOT NULL`, `embedding ARRAY<FLOAT64>` representing 768 finite FP32 values |
| Customer projection input | Both normalized customer texts and ordered `user_vectors [2, 768]` |
| Product projection input | Normalized `product_text` and `item_vectors [768]` |
| Final customer output | `crn`, both normalized texts, `embedding [128]` |
| Final product output | `article_nbr`, normalized `product_text`, `embedding [128]` |

The existing input physically stores a JSON string in `instances`; retain that
format until the named-field `text` input with Vertex `instanceType="object"`
is verified. Final embeddings use BigQuery `ARRAY<FLOAT64>`. The customer
matrix needs a verified array-of-struct or JSON representation because
BigQuery does not support arrays of arrays directly.

Normalize NULL/missing text to `""`; otherwise compare text exactly without
trimming or case folding. Customer runs deduplicate across both text fields.
Customer projection keys are ordered long-term/mid-term text pairs; product
keys are product texts. Final joins preserve all successful source rows and
their multiplicity, even when IDs repeat.

The future failure table has logical fields `stage STRING`,
`error_code STRING`, `attempt INT64`, safe `message STRING`, and source mapping: the
original identifier plus relevant normalized text fields. Define physical
customer/product schemas after confirming identifier types. Map encoder
failures to every dependent source row.

Retry failed/incomplete keys for at most two rounds after the initial attempt.
Publish available successes and remaining failures separately, not fabricated
vectors. Keep temporary tables by default; optional cleanup follows
reconciliation. Disable data-dependent caching for mutable tables.

## Configuration

| Setting | Meaning |
|---|---|
| `AIP_STORAGE_URI` | Vertex-provided GCS saved-artifact prefix. |
| `ENCODER_BATCH_SIZE` | Positive microbatch limit, default 32; separate from Vertex request batching. |
| `TRUNCATE_DIM` | Positive default output dimension truncation, default 768. |
| Model registration | Display name `embeddinggemma-300m-bf16`; `/predict`, `/health`, port 8080, explicit artifact and image versions. |
| Future pipeline | Explicit project, Vertex region, compatible BigQuery location, table names, separate encoder/projection model resources, pipeline root, and execution service account. |

The uploaded EmbeddingGemma-300m artifact is
`gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m`.
This matches the copied registration tooling's default artifact URI.
Artifact completeness, IAM, dataset locations, physical output schemas, and
training compatibility remain cloud verification prerequisites.
