CREATE TABLE `{input_table}` AS
SELECT TO_JSON_STRING(STRUCT(text AS text)) AS instances
FROM (SELECT DISTINCT text FROM ({text_union}));

CREATE TABLE `{embedding_table}` (
  text STRING, embedding ARRAY<FLOAT64>
);
CREATE TABLE `{manifest_table}` (
  attempt INT64, job_name STRING, state STRING,
  prediction_table STRING, failed_count INT64, incomplete_count INT64
);
