CREATE TEMP TABLE candidates AS
SELECT COLLATE(JSON_VALUE(instances, '$.text'), '') AS text,
       JSON_EXTRACT_ARRAY(prediction) AS elements
FROM `{prediction_table}`;

ASSERT NOT EXISTS (
  SELECT 1 FROM candidates AS c
  LEFT JOIN `{input_table}` AS i
  ON c.text = COLLATE(JSON_VALUE(i.instances, '$.text'), '')
  WHERE c.text IS NULL OR i.instances IS NULL
) AS 'Vertex returned missing or unexpected text keys; inspect raw outputs';

CREATE TEMP TABLE valid AS
SELECT text, ARRAY(
  SELECT SAFE_CAST(element AS FLOAT64) FROM UNNEST(elements) AS element
  WITH OFFSET AS position ORDER BY position
) AS embedding
FROM candidates
WHERE ARRAY_LENGTH(elements) = 768
AND NOT EXISTS (
  SELECT 1 FROM UNNEST(elements) AS element
  WHERE SAFE_CAST(element AS FLOAT64) IS NULL
     OR IS_NAN(SAFE_CAST(element AS FLOAT64))
     OR IS_INF(SAFE_CAST(element AS FLOAT64))
)
AND EXISTS (
  SELECT 1 FROM UNNEST(elements) AS element
  WHERE ABS(SAFE_CAST(element AS FLOAT64)) > 0
)
AND text IN (SELECT text FROM candidates GROUP BY text HAVING COUNT(*) = 1);

MERGE `{embedding_table}` AS existing
USING valid ON existing.text = valid.text
WHEN NOT MATCHED THEN INSERT (text, embedding) VALUES (valid.text, valid.embedding);
