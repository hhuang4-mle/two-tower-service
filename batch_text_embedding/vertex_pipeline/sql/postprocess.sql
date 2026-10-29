CREATE TEMP TABLE embedding_predictions AS
SELECT
  COLLATE(JSON_VALUE(instances, '$.text'), '') AS text,
  JSON_EXTRACT_ARRAY(prediction) AS elements
FROM ({prediction_source});

ASSERT NOT EXISTS (
  SELECT 1 FROM embedding_predictions
  WHERE text IS NULL
    OR elements IS NULL
    OR ARRAY_LENGTH(elements) = 0
    OR EXISTS (
      SELECT 1 FROM UNNEST(elements) AS element
      WHERE SAFE_CAST(element AS FLOAT64) IS NULL
        OR IS_NAN(SAFE_CAST(element AS FLOAT64))
        OR IS_INF(SAFE_CAST(element AS FLOAT64))
    )
) AS 'Predictions must contain text and nonempty finite numeric embeddings';

ASSERT NOT EXISTS (
  SELECT 1 FROM embedding_predictions
  GROUP BY text HAVING COUNT(*) != 1
) AS 'Expected exactly one prediction per text';

ASSERT (
  SELECT COUNT(DISTINCT ARRAY_LENGTH(elements)) <= 1
  FROM embedding_predictions
) AS 'Embedding dimensions must be consistent';

ASSERT NOT EXISTS (
  SELECT 1
  FROM (
    SELECT COLLATE(JSON_VALUE(instances, '$.text'), '') AS text
    FROM `{input_table}`
  ) AS inputs
  RIGHT JOIN embedding_predictions AS predictions
    ON inputs.text = predictions.text
  WHERE inputs.text IS NULL
) AS 'Prediction texts must belong to staged input texts';

CREATE TEMP TABLE embedding_result AS
SELECT
  source.*,
  ARRAY(
    SELECT CAST(element AS FLOAT64)
    FROM UNNEST(predictions.elements) AS element WITH OFFSET AS position
    ORDER BY position
  ) AS embedding
FROM `{source_table}` AS source
LEFT JOIN embedding_predictions AS predictions
  ON COLLATE(COALESCE(CAST(source.`{text_column}` AS STRING), ''), '')
    = predictions.text;

ASSERT (
  (SELECT COUNT(*) FROM embedding_result)
    = (SELECT COUNT(*) FROM `{source_table}`)
) AS 'Result must preserve the source row count';

CREATE OR REPLACE TABLE `{result_table}` AS
SELECT * FROM embedding_result;

SELECT
  (
    SELECT COUNT(*)
    FROM `{input_table}` AS inputs
    LEFT JOIN embedding_predictions AS predictions
      ON COLLATE(JSON_VALUE(inputs.instances, '$.text'), '') = predictions.text
    WHERE predictions.text IS NULL
  ) AS missing_texts,
  (
    SELECT COUNT(*) FROM embedding_result WHERE ARRAY_LENGTH(embedding) = 0
  ) AS missing_rows;
