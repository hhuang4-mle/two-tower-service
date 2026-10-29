# Vertex AI BigQuery smoke test

Status: procedure prepared, not executed. No local model downloads or local
inference tests are required. These commands create billable cloud resources
when you run them. Use synthetic text only, an isolated image/model and a new
dataset; do not overwrite existing models, jobs, or production tables.

Prerequisites: authenticated `gcloud`, `bq`, `curl`, `make`, Cloud Build and
Vertex APIs enabled, L4 quota in `us-central1`, and approved access to the
image registry and saved GCS artifacts. The caller and relevant Vertex service
agents need the appropriate image, artifact, and BigQuery permissions.
Verify that your BigQuery location is supported with the Vertex region.

Run the shell commands in the same shell from `batch_text_embedding/`.

## 1. Build a separate image in Cloud Build

```bash
export PROJECT_ID=gcp-wow-rwds-ai-super-plo-dev
export LOCATION=us-central1
export BQ_LOCATION=US
export RUN_ID=embedding_smoke_$(date -u +%Y%m%d_%H%M%S)
export IMAGE_NAME=text-embedding-batch-prediction
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME:$RUN_ID
export ARTIFACT_URI=gs://wx-1b759ae5-bd00-4de8-bf4d-427baa30e931/fs/text-embedding-models/embeddinggemma-300m/
export SMOKE_DATASET=$RUN_ID

make push PROJECT_ID="$PROJECT_ID" IMAGE_NAME="$IMAGE_NAME" IMAGE_TAG="$RUN_ID"
```

This uses the existing Cloud Build target; it does not deploy or execute the
Cloud Run Job. Check that the build succeeds and publishes the new image.
The artifact must be the full saved Sentence Transformers directory,
including `modules.json`, tokenizer, pooling/dense modules, and weights.

## 2. Create synthetic BigQuery fixtures

```bash
bq --project_id="$PROJECT_ID" mk --dataset --location="$BQ_LOCATION" \
  "$PROJECT_ID:$SMOKE_DATASET"

bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" query --use_legacy_sql=false <<SQL
CREATE TABLE \`$PROJECT_ID.$SMOKE_DATASET.source\` AS
SELECT text FROM UNNEST([
  'Milk bread eggs',
  'Milk bread eggs',
  '',
  CAST(NULL AS STRING),
  '   ',
  CODE_POINTS_TO_STRING([19990, 30028]),
  REPEAT(CONCAT(CODE_POINTS_TO_STRING([19990, 30028]), ' '), 400),
  CONCAT(REPEAT('milk ', 4096), 'tail alpha'),
  CONCAT(REPEAT('milk ', 4096), 'tail beta')
]) AS text;

CREATE TABLE \`$PROJECT_ID.$SMOKE_DATASET.legacy_input\` AS
SELECT TO_JSON_STRING(STRUCT(COALESCE(text, '') AS text)) AS instances
FROM \`$PROJECT_ID.$SMOKE_DATASET.source\`;

CREATE TABLE \`$PROJECT_ID.$SMOKE_DATASET.object_input\` AS
SELECT COALESCE(text, '') AS text
FROM \`$PROJECT_ID.$SMOKE_DATASET.source\`;
SQL
```

There are nine rows. Duplicates are deliberately retained to check prediction
multiplicity; deduplication is a future pipeline responsibility. NULL becomes
empty text upstream. The two oversized texts have different original keys
but share the token prefix retained after truncation. The Unicode cases guard
against treating byte counts as token counts.

## 3. Register a new model explicitly

```bash
gcloud ai models upload \
  --project="$PROJECT_ID" --region="$LOCATION" \
  --model-id="$RUN_ID" --display-name="$RUN_ID" \
  --artifact-uri="$ARTIFACT_URI" \
  --container-image-uri="$IMAGE_URI" \
  --container-predict-route=/predict \
  --container-health-route=/health \
  --container-ports=8080 \
  --container-env-vars=TRUNCATE_DIM=768,ENCODER_BATCH_SIZE=32

export MODEL_RESOURCE_NAME=projects/$PROJECT_ID/locations/$LOCATION/models/$RUN_ID
```

Do not let the copied runner auto-select an existing model by display name:
that could reuse the old image instead of exercising these changes.

## 4. Test the proven serialized BigQuery format first

```bash
curl --fail-with-body --silent --show-error \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/batchPredictionJobs" \
  --data-binary @- <<JSON
{
  "displayName": "$RUN_ID-legacy",
  "model": "$MODEL_RESOURCE_NAME",
  "inputConfig": {
    "instancesFormat": "bigquery",
    "bigquerySource": {
      "inputUri": "bq://$PROJECT_ID.$SMOKE_DATASET.legacy_input"
    }
  },
  "outputConfig": {
    "predictionsFormat": "bigquery",
    "bigqueryDestination": {
      "outputUri": "bq://$PROJECT_ID.$SMOKE_DATASET"
    }
  },
  "dedicatedResources": {
    "machineSpec": {
      "machineType": "g2-standard-4",
      "acceleratorType": "NVIDIA_L4",
      "acceleratorCount": 1
    },
    "startingReplicaCount": 1,
    "maxReplicaCount": 1
  },
  "manualBatchTuningParameters": {"batchSize": 32},
  "disableContainerLogging": false
}
JSON
```

Copy the returned job `name`. Inspect it in the Vertex batch-prediction
console, or repeat this read-only request as needed:

```bash
export JOB_NAME=projects/REPLACE_PROJECT/locations/us-central1/batchPredictionJobs/REPLACE_JOB_ID
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://$LOCATION-aiplatform.googleapis.com/v1/$JOB_NAME"
```

Require `JOB_STATE_SUCCEEDED`, inspect `completionStats`, `partialFailures`,
and `error`, and record `outputInfo.bigqueryOutputDataset` and
`outputInfo.bigqueryOutputTable`. A succeeded job alone does not establish
complete predictions. Inspect container logs for startup/readiness problems,
CUDA/BF16 errors, OOM, and HTTP failures.
Require the readiness log to report `torch.bfloat16` and a sequence limit no
greater than 2,048 tokens. This logs configuration only, not source text.

The single L4 replica is a smoke-test override, not a production scaling
decision. A nine-row job also tests a partial request smaller than batch size
32. It does not benchmark throughput or prove large-request microbatching.

## 5. Check the actual output schema and results

Set `PREDICTION_TABLE` to the actual fully qualified prediction table reported
by Vertex, without the `bq://` prefix. Do not assume a generated table name.

```bash
export PREDICTION_TABLE=REPLACE_PROJECT.REPLACE_DATASET.REPLACE_TABLE
bq show --schema --format=prettyjson "$PREDICTION_TABLE"
```

The query below uses the copied workflow's `instances`/`prediction` JSON
contract. If Vertex produces a different physical schema, inspect it and
resolve that integration mismatch before proceeding; do not silently discard
text keys or rely on row order.

```bash
bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" query --use_legacy_sql=false <<SQL
CREATE TEMP TABLE checked AS
SELECT
  JSON_VALUE(instances, '$.text') AS text,
  JSON_EXTRACT_ARRAY(prediction) AS elements
FROM \`$PREDICTION_TABLE\`;

ASSERT (SELECT COUNT(*) FROM checked) = 9 AS 'Expected nine predictions';
ASSERT NOT EXISTS (
  SELECT 1 FROM checked
  WHERE text IS NULL OR elements IS NULL OR ARRAY_LENGTH(elements) != 768
    OR EXISTS (
      SELECT 1 FROM UNNEST(elements) AS element
      WHERE SAFE_CAST(element AS FLOAT64) IS NULL
         OR IS_NAN(SAFE_CAST(element AS FLOAT64))
         OR IS_INF(SAFE_CAST(element AS FLOAT64))
    )
) AS 'Expected preserved text and finite 768-dimensional vectors';

ASSERT NOT EXISTS (
  SELECT 1 FROM checked
  WHERE ABS(SQRT((
    SELECT SUM(POW(CAST(element AS FLOAT64), 2))
    FROM UNNEST(elements) AS element
  )) - 1.0) > 0.02
) AS 'Expected approximately normalized encoder vectors';

ASSERT NOT EXISTS (
  SELECT 1 FROM
    (SELECT COALESCE(text, '') AS text, COUNT(*) AS expected
     FROM \`$PROJECT_ID.$SMOKE_DATASET.source\` GROUP BY 1)
  FULL JOIN
    (SELECT text, COUNT(*) AS actual FROM checked GROUP BY 1)
  USING (text)
  WHERE COALESCE(expected, 0) != COALESCE(actual, 0)
) AS 'Original text keys and duplicate multiplicity must be preserved';

SELECT COUNT(*) AS validated_predictions FROM checked;
SQL
```

The encoder norm tolerance allows BF16 rounding; the architecture's tighter
`1e-4` norm requirement applies to the FP32 projection output, not this test.
Compare the two oversized-text vectors: they should be approximately equal
because their retained token prefixes match. Confirm the saved artifact's
tokenizer limit and training convention; numerical comparisons alone do not
prove exact special-token handling.

## 6. Verify named-field input separately

Repeat step 4 with display name `$RUN_ID-object`, input URI ending in
`.object_input`, and this additional top-level field:

```json
"instanceConfig": {"instanceType": "object"}
```

Repeat steps 4-5 for that job's own output table. Both formats must pass.
Do not replace the proven serialized input path until the named-field path is
verified.

## 7. Record results and inspect failures

Record image tag/digest, model resource, artifact version, both job names,
output schemas, row counts, and any errors. This verifies BigQuery
compatibility and basic inference, not recommendation quality or the complete
two-tower pipeline. Compare against training-generated vectors before
production use, particularly because prompt behavior and precision changed.

For a separate optional negative job, create a new input table containing
`{"text":123}` in its serialized `instances` column. Expect failed prediction
records, no fabricated vector, and no input text echoed by the predictor.
Inspect actual Vertex error tables/request retry behavior. An invalid row may
fail its entire request; pipeline isolation/retries are not implemented yet.

No automatic cleanup is included. Retain the smoke resources until results
are reconciled; then remove only the explicitly recorded test model, test
tables/dataset, and test image according to your retention policy. Cancelling
a client command does not cancel an already submitted Vertex job.
